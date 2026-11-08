# EN: Distribution-network topology, AC power-flow coupling, and voltage-sensitivity helper.
# 中文：配电网拓扑映射、交流潮流耦合与节点电压灵敏度计算模块。

from pathlib import Path
import time, warnings
import numpy as np
from records import read, write, record, require, STATIONS, CAPS, digest
from case33_source import create_network, assemble as assemble8, validate_snapshot

class Network:

    def __init__(self, case, source_ref, out, journal, arm):
        from capture_runtime import make_callbacks
        require(case['network_asset'])
        self.case = case
        self.source = source_ref
        self.arm = arm
        self.out = out
        self.journal = journal
        self.net, self.base, self.mapping, self.metadata = create_network(read(case['network_asset']['path']), case)
        self.state = dict(ac=0, lp=0, forbidden_optimizer_attempts=0, context={})
        self.cache = {}
        self.cache_hits = 0
        from snapshot_contract import PF_OPTIONS
        self.static_key = dict(network_asset=case['network_asset'], source_case=source_ref, metadata=self.metadata, PF_OPTIONS=PF_OPTIONS, slack_voltage=1.0, station_PQ_policy='P=imports/1000,Q=0', initialization='flat')
        for x in ('ac', 'lp', 'jacobian'):
            (out / x).mkdir()

        def before(kind, cid, ctx):
            role = 'AC' if kind == 'AC' else 'PROJECTION_LP'
            got = journal.attempt(role, dict(ctx, minute=ctx['minute']))
            if got != cid:
                raise ValueError('Primitive call ID differs')
        self.ac_call, self.lp_call = make_callbacks(out, self.state, before, journal.run, max_ac_calls=10800, max_lp_calls=CAPS[arm]['PROJECTION_LP'])

    def ac(self, second, held_id, imports):
        minute = second // 60
        h = self.case['stations'][0]['hours'][minute // 60]
        factor = h['background_factor'] * self.case['background_peak']
        key = digest(dict(static=self.static_key, background_factor=factor, station_import_kw=imports))
        if key in self.cache:
            self.cache_hits += 1
            cached = self.cache[key]
            require(cached['raw'])
            raw = read(cached['raw']['path'])
            return dict(cached, snapshot=raw['snapshot'], cache_hit=True, cache_key=key, current_second=second)
        for k in ('p_mw', 'q_mvar'):
            self.net.load.loc[self.base.index, k] = self.base[k] * factor
        for sid, i in self.mapping.items():
            self.net.load.loc[i, 'p_mw'] = imports[sid] / 1000.0
            self.net.load.loc[i, 'q_mvar'] = 0.0
        self.state['context'] = dict(arm=self.arm, minute=minute, second=second, model_time_seconds=second, source_step=402 + minute // 60, role='ACTUAL_CURRENT_APPLIED_ACTION', input_sha256=self.source['sha256'], held_command_id=held_id)
        try:
            snap, check, ref = self.ac_call(self.net, 'actual_minute', imports, self.mapping)
            validate_snapshot(snap, self.metadata, factor)
            result = dict(valid=True, snapshot=snap, audit=check, raw=ref, failure=None, cache_key=key, current_second=second, cache_hit=False)
            self.cache[key] = {k: v for k, v in result.items() if k not in ('snapshot', 'audit')}
            return result
        except Exception as e:
            p = self.out / 'ac' / f"{self.state['ac']:03d}.json"
            raw = read(p) if p.exists() else {}
            return dict(valid=False, snapshot=raw.get('snapshot'), audit=None, raw=record(p) if p.exists() else None, failure=dict(type=type(e).__name__, message=str(e)), cache_key=key, current_second=second, cache_hit=False)

    def jacobian(self, minute, snapshot, source_ac, rhs_count):
        import scipy.linalg
        if rhs_count not in (7, 8):
            raise ValueError('Declared J7/J8 primitive')
        J, all_rhs = assemble8(snapshot, self.metadata)
        rhs = all_rhs[:, :rhs_count]
        p = self.out / 'jacobian' / f'{minute:03d}'
        p.mkdir()
        with (p / 'REQUEST.npz').open('xb') as f:
            np.savez_compressed(f, J=J, rhs=rhs)
        req = record(p / 'REQUEST.npz')
        cid = self.journal.attempt('J' + str(rhs_count), dict(minute=minute, request=req, source_ac=source_ac, right_hand_sides=rhs_count))
        start = time.perf_counter()
        returned = False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                K = np.linalg.solve(J, rhs) if rhs_count == 7 else scipy.linalg.solve(J, rhs, assume_a='gen', check_finite=True)
                returned = True
            write(p / 'RETURN_KNOWN.json', dict(call_id=cid, kind='J' + str(rhs_count), minute=minute, returned=True, wall_seconds=time.perf_counter() - start))
            with (p / 'RESPONSE.npz').open('xb') as f:
                np.savez_compressed(f, solution=K)
            residual = float(np.max(np.abs(J @ K - rhs)))
            if not np.isfinite(K).all() or residual > 1e-10:
                raise ValueError('Original current Jacobian equation residual')
            response = record(p / 'RESPONSE.npz')
            write(p / 'RESULT.json', dict(request=req, response=response, source_ac=source_ac, shape=list(K.shape), max_equation_residual=residual, warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught]))
            return (K, req, response)
        except Exception as e:
            write(p / 'FAILURE.json', dict(returned=returned, type=type(e).__name__, message=str(e), no_retry=True))
            raise

    def projection(self, minute, snapshot, K, scalar, caps, source_ac, application_soc):
        from current_jacobian import projection
        from scene_physics import apply_correction
        A, b, pred = projection(snapshot, dict(sensitivity=K[32:, :7]), {s: scalar[s]['grid_import_kw'] for s in STATIONS}, snapshot['station_import_kw'], self.case['voltage_lower_pu'], self.case['voltage_upper_by_bus'])
        self.state['context'] = dict(arm=self.arm, minute=minute, model_time_seconds=60 * minute, source_step=402 + minute // 60, role='NEXT_COMMAND_CURRENT_INFORMATION', source_ac=source_ac, predicted_application_soc=application_soc)
        delta, check, ref = self.lp_call('all32_current_voltage', np.arange(64), -A, b, np.asarray([caps[s]['total_cap_kw'] for s in STATIONS]))
        if np.max(A @ delta - b) > 1e-10:
            raise ValueError('All-node projection residual')
        return (apply_correction(scalar, caps, delta), dict(reference=ref, prediction=pred.tolist(), correction_kw=float(sum(delta)), all_voltage_rows=64))
