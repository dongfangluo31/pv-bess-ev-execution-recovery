# EN: Three-scenario (favorable / nominal / adverse) rolling-horizon MILP scheduler solved deterministically with SciPy/HiGHS.
# 中文：基于 SciPy/HiGHS 确定性求解的三场景（乐观/名义/最不利）滚动时域 MILP 名义调度模型。

from copy import deepcopy
from dataclasses import asdict
import math, time, warnings
import numpy as np
from scipy.sparse import coo_matrix, block_diag, vstack
from station_core.config import BatteryConfig, SolverConfig, ExperimentConfig, Method
from native_block import emit_station_gs_block, HorizonData
from records import write, record, safe

def config(station):
    d = dict(station['core'])
    d['battery'] = BatteryConfig(**d['battery'])
    d['solver'] = SolverConfig(**d['solver'])
    d['scenario_probabilities'] = tuple(d['scenario_probabilities'])
    return ExperimentConfig(**d)

def block(station, hour, soc, point, first_dt):
    cfg = config(station)
    original = station['hours'][hour]['horizon']
    h = deepcopy(original)
    for name in ('ev_lower_kw', 'ev_point_kw', 'ev_upper_kw'):
        h[name][0] = point['actual_ev_kw']
    for name in ('pv_lower_kw', 'pv_point_kw', 'pv_upper_kw'):
        h[name][0] = point['actual_pv_kw']
    native = emit_station_gs_block(Method.GS_F1, HorizonData(**h), soc, cfg, first_step_hours=first_dt, reserve_horizon=HorizonData(**original))
    native['upper'][native['indices']['discharge']] = np.minimum(native['upper'][native['indices']['discharge']], np.maximum(native['ev_s'] - native['pv_s'], 0.0))
    return native

def build(stations, hour, application_soc, packet, first_dt, network=None):
    blocks = []
    offset = 0
    matrices = []
    for s in stations:
        sid = s['profile_id']
        n = block(s, hour, application_soc[sid], packet['stations'][sid], first_dt)
        A = coo_matrix((n['coo']['data'], (n['coo']['row'], n['coo']['col'])), shape=n['coo']['shape']).tocsc()
        matrices.append(A)
        blocks.append(dict(profile_id=sid, offset=offset, indices={k: v + offset for k, v in n['indices'].items()}, native=n))
        offset += len(n['c'])
    c = np.concatenate([b['native']['c'] for b in blocks])
    lo = np.concatenate([b['native']['lower'] for b in blocks])
    hi = np.concatenate([b['native']['upper'] for b in blocks])
    integer = np.concatenate([b['native']['integrality'] for b in blocks])
    A = block_diag(matrices, format='csc')
    rlo = np.concatenate([b['native']['row_lower'] for b in blocks])
    rhi = np.concatenate([b['native']['row_upper'] for b in blocks])
    labels = []
    if network is not None:
        rows = []
        cols = []
        values = []
        upper = []
        for k in range(24):
            base = np.asarray(network['voltage']) - np.asarray(network['Sv']) @ np.asarray(network['imports']) + np.asarray(network['Sb']) * (network['background'][k] - network['background'][0])
            for sign in (-1, 1):
                for bus in range(32):
                    row = len(upper)
                    for j, b in enumerate(blocks):
                        rows.append(row)
                        cols.append(int(b['indices']['grid'][1, k]))
                        values.append(sign * network['Sv'][bus][j])
                    upper.append(base[bus] - network['lower'] if sign < 0 else network['upper'][str(bus + 1)] - base[bus])
                    labels.append(dict(kind='VLOW' if sign < 0 else 'VHIGH', step=k, bus=bus + 1))
            row = len(upper)
            for b in blocks:
                rows.append(row)
                cols.append(int(b['indices']['grid'][1, k]))
                values.append(1.0)
            upper.append(network['PCC'])
            labels.append(dict(kind='PCC', step=k))
        extra = coo_matrix((values, (rows, cols)), shape=(len(upper), offset)).tocsc()
        A = vstack((A, extra), format='csc')
        rlo = np.r_[rlo, np.full(len(upper), -np.inf)]
        rhi = np.r_[rhi, upper]
    return dict(c=c, lower=lo, upper=hi, integrality=integer, A=A, row_lower=rlo, row_upper=rhi, blocks=blocks, network=network, network_labels=labels, first_step_hours=first_dt)

def certificate(m, result):
    x = result.get('x')
    ok = x is not None and result.get('success') is True and (result.get('status') == 0)
    if not ok:
        return dict(valid=False, reason='Native status0 successful incumbent required', status=result.get('status'))
    x = np.asarray(x, float)
    finite = bool(x.shape == m['c'].shape and np.isfinite(x).all())
    if not finite:
        return dict(valid=False, reason='Finite full vector required')
    ax = m['A'] @ x
    rows = float(max(np.max(m['row_lower'] - ax), np.max(ax - m['row_upper']), 0.0))
    bounds = float(max(np.max(m['lower'] - x), np.max(x - m['upper']), 0.0))
    ints = float(np.max(np.abs(x[m['integrality'] != 0] - np.round(x[m['integrality'] != 0]))))
    gap = result.get('mip_gap')
    gap_ok = gap is not None and math.isfinite(gap) and (0 <= gap <= 0.001)
    actions = {b['profile_id']: dict(charge_kw=float(x[b['indices']['charge'][1, 0]]), discharge_kw=float(x[b['indices']['discharge'][1, 0]])) for b in m['blocks']}
    first = 0.0
    for b in m['blocks']:
        a = actions[b['profile_id']]
        n = b['native']
        battery = n['config'].battery
        soc = n['issue_soc_canonical']
        dt = m['first_step_hours']
        maxc = max((battery.soc_max - soc) * battery.energy_kwh / (battery.eta_charge * dt), 0.0)
        maxd = max(battery.eta_discharge * battery.energy_kwh * (soc - battery.soc_min) / dt, 0.0)
        pv_cap = max(float(n['ev_s'][1, 0] - n['pv_s'][1, 0]), 0.0)
        first = max(first, -a['charge_kw'], -a['discharge_kw'], min(a['charge_kw'], a['discharge_kw']), a['charge_kw'] - battery.charge_limit_kw, a['discharge_kw'] - battery.discharge_limit_kw, a['charge_kw'] - maxc, a['discharge_kw'] - maxd, a['discharge_kw'] - pv_cap)
    objective = float(m['c'] @ x)
    fun = result.get('fun')
    dual = result.get('mip_dual_bound')
    objective_tol = 1e-07
    objective_ok = fun is not None and math.isfinite(fun) and (abs(fun - objective) <= objective_tol)
    actual_gap = None if dual is None or not math.isfinite(dual) else max(objective - dual, 0.0) / max(abs(objective), 1e-10)
    dual_ok = gap_ok and actual_gap is not None and (dual <= objective + objective_tol) and (actual_gap <= 0.001 + 1e-10) and (abs(actual_gap - gap) <= 1e-07)
    valid = rows <= 1e-07 and bounds <= 1e-07 and (ints <= 1e-07) and gap_ok and (first <= 1e-09) and objective_ok and dual_ok
    return dict(valid=valid, max_row_residual=rows, max_bound_residual=bounds, max_integrality_residual=ints, first_domain_residual=first, gap=gap, recomputed_gap=actual_gap, objective=objective, solver_fun=fun, dual_bound=dual, objective_consistent=objective_ok, dual_consistent=dual_ok, nominal=actions)

def solve(m, out, journal, kind, minute, real_milp, deadline):
    from scipy.optimize import Bounds, LinearConstraint
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    entered = False
    cid = None
    A = m['A'].tocoo()
    with (out / 'REQUEST.npz').open('xb') as f:
        np.savez_compressed(f, c=m['c'], lower=m['lower'], upper=m['upper'], integrality=m['integrality'], row_lower=m['row_lower'], row_upper=m['row_upper'], A_row=A.row, A_col=A.col, A_data=A.data, A_shape=np.asarray(A.shape))
    meta = dict(shape=list(A.shape), first_step_hours=m['first_step_hours'], blocks=[dict(profile_id=b['profile_id'], offset=b['offset'], indices=b['indices']) for b in m['blocks']], network=m['network'], network_labels=m['network_labels'])
    write(out / 'MODEL.json', meta)
    try:
        kwargs = dict(c=m['c'], integrality=m['integrality'], bounds=Bounds(m['lower'], m['upper']), constraints=LinearConstraint(m['A'], m['row_lower'], m['row_upper']))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('No backend time after request persistence')
        presolve = all((abs(b['native']['issue_soc_canonical'] - b['native']['config'].battery.soc_max) > 1e-12 for b in m['blocks']))
        opts = dict(time_limit=min(60.0, remaining), mip_rel_gap=0.001, mip_feasibility_tolerance=1e-09, presolve=presolve, disp=False, random_seed=20260706, threads=1, parallel=False)
        cid = journal.attempt(kind, dict(minute=minute, request=record(out / 'REQUEST.npz'), options=opts))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('No backend time after durable intent')
        opts['time_limit'] = min(60.0, remaining)
        write(out / 'BACKEND_BOUNDARY.json', dict(call_id=cid, boundary='PERSISTED_IMMEDIATELY_BEFORE_BACKEND_ENTRY', options=opts, known_entered=False, known_returned=False, interpretation='Boundary alone is an unresolved intent; RETURN_KNOWN or FAILURE.backend_entered resolves normal execution'))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('No backend time after boundary persistence')
        opts['time_limit'] = min(60.0, remaining)
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            backend_options = dict(opts)
            entered = True
            ans = real_milp(**kwargs, options=backend_options)
        elapsed = time.perf_counter() - tick
        raw = dict(ans)
        write(out / 'RETURN_KNOWN.json', dict(call_id=cid, backend_returned=True, backend_seconds=elapsed))
        write(out / 'RAW_RESULT.json', dict(result=raw, options=opts, warnings=[str(w.message) for w in caught]))
        check = certificate(m, raw)
        write(out / 'CERTIFICATE.json', check)
        if not check['valid']:
            raise RuntimeError('Native-quality incumbent rejected; retained, no retry')
        if time.perf_counter() > deadline:
            raise TimeoutError('Complete solve/certificate missed effect slot')
        receipt = dict(call_id=cid, kind=kind, backend_seconds=elapsed, whole_seconds=time.perf_counter() - start, request=record(out / 'REQUEST.npz'), result=record(out / 'RAW_RESULT.json'), certificate=record(out / 'CERTIFICATE.json'))
        write(out / 'RECEIPT.json', receipt)
        return (check['nominal'], receipt)
    except BaseException as e:
        write(out / 'FAILURE.json', dict(type=type(e).__name__, message=str(e), call_id=cid, backend_entered=entered, elapsed_seconds=time.perf_counter() - start, retry=False))
        raise
