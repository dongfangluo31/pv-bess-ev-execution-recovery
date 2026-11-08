# EN: Isolated controller companion process serving `HELLO`, `STEP_OPEN`, and `STOP` over line-delimited JSON-RPC.
# 中文：独立控制器子进程：通过行分隔 JSON-RPC 处理握手、步初状态包接收与实时控制指令签发。

from pathlib import Path
from dataclasses import asdict
import json, math, os, sys, time
from protocol import load, write, record, Log, emit
from native_process import own_identity
from native_contract import current_soc
from station_core.config import BatteryConfig, ExperimentConfig, Method, FilterKind
from station_core.filter import apply_execution_filter

def main():
    cfg = load(sys.argv[1])
    out = Path(cfg['output_dir'])
    log = Log(out / 'events.jsonl', 'controller')
    battery = BatteryConfig(**cfg['battery'])
    policy = cfg['prediction']
    arm = cfg['arm']
    nonce = cfg['nonce']
    if cfg['dataset'] == 'opsd':
        from native_contract import forecast
        from opening_state_opsd import validate_opening
    elif cfg['dataset'] == 'sofie':
        from native_contract_sofie import forecast
        from opening_state_sofie import validate_opening
    else:
        raise ValueError('Unknown dataset')
    if arm not in ('F0_FAST', 'F1_FAST'):
        raise ValueError('Unknown frozen arm')
    if cfg['decision_period_s'] != 60 or cfg['command_lifetime_s'] != 60 or cfg['paired_latency_contract'] != 'NEXT_CONTROL_SLOT60':
        raise ValueError('Both arms require the same fixed minute slot')
    if any((k in cfg for k in ('segments', 'native_schedule', 'settlements', 'plant_config', 'input_directory', 'plant_profile', 'expected_forecasts', 'input_references', 'full_input_directory'))):
        raise ValueError('Controller config includes forbidden future input channel')
    core = ExperimentConfig(battery=battery, horizon_steps=96, step_hours=0.25, feeder_safe_kw=cfg['feeder_safe_kw'], pv_rated_kw=cfg['model_pv_rating_kw_unused_for_native_clipping'], reserve_hours=1.0)
    nominal = None
    nominal_epoch = None
    reserve = None
    calls = 0
    failures = []
    last_control = -1
    last_outer = -1
    emit(dict(type='HELLO', role='controller', nonce=nonce, identity=own_identity()))
    log.event('START', 0, arm=arm, mode=cfg['mode'], horizon_hours=24, core_config=asdict(core), pressure_contract='dimensionless1', future_data_in_config=False, static_widths_sha256=policy.get('static_widths_sha256'), demand_semantics='Whole-house demand in legacy ev_* slots, EV submeter separate')

    def solve(packet, soc_m, t, fc):
        nonlocal calls
        import numpy as np
        from station_core import model
        h = model.HorizonData(**{k: np.asarray(v, float) for k, v in fc['horizon'].items()})
        original = model.milp

        def monitored(**kwargs):
            nonlocal calls
            if cfg['mode'] != 'mpc' or calls >= cfg['maximum_MPC_calls']:
                raise RuntimeError('No authorized optimizer call remains')
            call_id = calls + 1
            call_dir = out / f'mpc_{call_id:03d}'
            call_dir.mkdir(exist_ok=False)
            A = kwargs['constraints'].A.tocoo()
            with (call_dir / 'request.npz').open('xb') as f:
                np.savez(f, c=kwargs['c'], integrality=kwargs['integrality'], lower=kwargs['bounds'].lb, upper=kwargs['bounds'].ub, row_lower=kwargs['constraints'].lb, row_upper=kwargs['constraints'].ub, A_row=A.row, A_col=A.col, A_data=A.data, A_shape=A.shape)
            request = record(call_dir / 'request.npz')
            write(call_dir / 'REQUEST.json', dict(call_id=call_id, arm=arm, outer_index=int(t // 900), request=request, options_before_call=dict(kwargs['options']), current_soc=soc_m['soc'], soc_measurement=soc_m, native_opening=packet, forecast=fc, horizon=fc['horizon'], forecast_issued_s=fc['forecast_issued_s'], core_config=asdict(core), method='GS_F1', horizon_start_s=t, horizon_hours=24))
            calls = call_id
            attempt = log.event('MPC_BEFORE_BACKEND', t, arm=arm, call_id=call_id, outer_index=int(t // 900), request=request, observation_id=packet['observation_id'], SOC_used=soc_m['soc'])
            with Path(cfg['attempt_ledger']).open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(run_id=cfg['run_id'], kind='MPC', **attempt), allow_nan=False) + '\n')
                f.flush()
            begun = time.perf_counter()
            try:
                answer = original(**kwargs)
            except BaseException as exc:
                write(call_dir / 'BACKEND_FAILURE.json', dict(error=repr(exc), wall_seconds=time.perf_counter() - begun, request=request))
                log.event('MPC_BACKEND_FAILED', t, call_id=call_id, error=repr(exc))
                raise
            with (call_dir / 'response.npz').open('xb') as f:
                np.savez(f, **{'x': answer.x} if answer.x is not None else {})

            def scalar(v):
                return None if v is None else float(v) if math.isfinite(float(v)) else str(v)
            write(call_dir / 'STATUS.json', dict(request=request, response=record(call_dir / 'response.npz'), status=int(answer.status), message=str(answer.message), fun=scalar(answer.fun), mip_gap=scalar(getattr(answer, 'mip_gap', None)), mip_dual_bound=scalar(getattr(answer, 'mip_dual_bound', None)), mip_node_count=scalar(getattr(answer, 'mip_node_count', None)), options_after_call=dict(kwargs['options']), backend_wall_seconds=time.perf_counter() - begun))
            log.event('MPC_RETURNED', t, call_id=call_id, status=int(answer.status))
            return answer
        model.milp = monitored
        try:
            return model.solve_first_step(Method.GS_F1, h, soc_m['soc'], core)
        finally:
            model.milp = original
    for line in sys.stdin:
        msg = json.loads(line)
        rid = msg['request_id']
        t = msg.get('sim_s', 0)
        if msg['nonce'] != nonce:
            raise RuntimeError('Nonce mismatch')
        if msg['type'] == 'STOP':
            emit(dict(request_id=rid, type='STOPPED', nonce=nonce))
            break
        if msg['type'] != 'DECIDE':
            raise ValueError('Unsupported controller message')
        if t <= last_control or t % cfg['decision_period_s'] != 0 or (not 0 <= t < cfg['duration_s']):
            raise ValueError('Unscheduled control invocation')
        last_control = t
        begun = time.perf_counter()
        packet = msg.get('native_opening')
        soc_m = msg.get('soc_measurement')
        reason = None
        filtered = None
        common = None
        decision = None
        raw_before = nominal
        fc = None
        outer = t % 900 == 0
        try:
            if packet is None or packet['decision_s'] != t:
                raise ValueError('Current scheduled native packet required')
            ev, pv = validate_opening(packet, t)
            soc = current_soc(soc_m, t, battery)
        except Exception as exc:
            reason = 'INPUT_INVALID'
            log.event('INPUT_REJECT_FALLBACK', t, error=repr(exc), native_opening=packet, soc_measurement=soc_m)
        if outer:
            epoch = int(t // 900)
            if epoch <= last_outer:
                raise ValueError('Duplicate outer step')
            last_outer = epoch
            nominal = None
            nominal_epoch = None
            reserve = None
            if reason:
                log.event('MPC_OPPORTUNITY_SKIPPED', t, outer_index=epoch, reason=reason, actual_call=False)
            else:
                fc = forecast(packet, t, policy)
                try:
                    if cfg['mode'] == 'fixture':
                        nominal = tuple(cfg['fixture_nominal'])
                        decision = dict(scope='HAND_COMMAND_ONLY_ZERO_MPC_FIXTURE')
                        log.event('FIXTURE_NOMINAL', t, raw_nominal=nominal, forecast=fc, actual_call=False)
                    elif cfg['mode'] == 'mpc':
                        result = solve(packet, soc_m, t, fc)
                        decision = asdict(result)
                        if result.solver_status != 0 or result.mip_gap is None or result.mip_gap > core.solver.mip_relative_gap:
                            raise RuntimeError('MPC result not certified by frozen status/gap')
                        nominal = (result.charge_kw, result.discharge_kw)
                    else:
                        raise ValueError('Unknown controller mode')
                    nominal_epoch = epoch
                    reserve = fc['reserve_requirement_kwh']
                    log.event('NOMINAL_ACCEPTED', t, raw_nominal=nominal, outer_index=epoch, decision=decision, forecast=fc, SOC_used=soc, soc_measurement=soc_m, native_opening=packet)
                except Exception as exc:
                    nominal = None
                    reason = 'MPC_FAILED'
                    failures.append(dict(t=t, reason=reason, error=repr(exc)))
                    log.event('MPC_FAILURE_NO_RETRY', t, reason=reason, error=repr(exc), next_opportunity_s=t + 900)
        if not reason and nominal is None:
            reason = 'NO_VALID_NOMINAL_FOR_CURRENT_OUTER'
        charge = discharge = 0.0
        if not reason:
            try:
                common = apply_execution_filter(FilterKind.NONE, nominal[0], nominal[1], ev, pv, soc, battery, cfg['feeder_safe_kw'], 1 / 60.0, reserve)
                charge, discharge = (common.charge_kw, common.discharge_kw)
                if arm == 'F1_FAST':
                    filtered = apply_execution_filter(FilterKind.F1_RESERVE, charge, discharge, ev, pv, soc, battery, cfg['feeder_safe_kw'], 1 / 60.0, reserve)
                    charge, discharge = (filtered.charge_kw, filtered.discharge_kw)
            except Exception as exc:
                charge = discharge = 0.0
                reason = 'COMMON_DOMAIN_OR_REF_PHYSICS_REJECTED'
                failures.append(dict(t=t, reason=reason, error=repr(exc)))
                log.event('REF_INPUT_REJECT_FALLBACK', t, error=repr(exc), raw_nominal=nominal, native_opening=packet, soc_measurement=soc_m, common_mapping_in_both_arms=True)
        issue = t + time.perf_counter() - begun
        safe_records = packet.get('records', []) if isinstance(packet, dict) else []
        finite_available = [r.get('available_s') for r in safe_records if isinstance(r.get('available_s'), (int, float)) and math.isfinite(r['available_s'])]
        age_rows = [dict(channel=r.get('channel'), connector_pk=r.get('connector_pk'), age_s=issue - r['reported_s'] if isinstance(r.get('reported_s'), (int, float)) and math.isfinite(r['reported_s']) else None) for r in safe_records]
        command = dict(seq=msg['control_seq'], issue_s=issue, issue_wall_ns=time.perf_counter_ns(), valid_until_s=t + 2 * cfg['decision_period_s'], dispatch_slot_s=t + cfg['decision_period_s'], control_decision_s=t, lifetime_reference='fixed_dispatch_slot', signed_kw=discharge - charge, charge_kw=charge, discharge_kw=discharge, controller_pid=os.getpid(), arm=arm, reason=reason or ('FAST_NOMINAL_HOLD' if arm == 'F0_FAST' else 'PV_AWARE_F1'), native_observation_id=packet.get('observation_id') if isinstance(packet, dict) else None, native_available_max_s=max(finite_available) if finite_available else None, per_channel_age_at_issue=age_rows, SOC_used=None if reason == 'INPUT_INVALID' else soc_m['soc'], soc_measurement_id=None if soc_m is None else soc_m['id'], nominal_epoch=nominal_epoch, raw_nominal=nominal, physical_pre_adaptation_added=True, pv_aware_noexport=True, common_mapping_in_both_arms=True, common_map_output=None if common is None else dict(charge_kw=common.charge_kw, discharge_kw=common.discharge_kw), common_map_discharge_clip_kw=None if common is None else common.common_discharge_clip_kw, original_filter_physical_input=None if filtered is None else dict(charge_kw=filtered.plant_charge_kw, discharge_kw=filtered.plant_discharge_kw), REF_output=None if filtered is None else dict(charge_kw=charge, discharge_kw=discharge), reserve_requirement_kwh=reserve, horizon_hours=24)
        log.event('COMMAND_ISSUED', issue, command=command, native_opening=packet, soc_measurement=soc_m, raw_previous_nominal=raw_before, common_map=None if common is None else asdict(common), filter=None if filtered is None else asdict(filtered), MPC_calls=calls)
        emit(dict(request_id=rid, type='COMMAND', nonce=nonce, command=command))
    log.event('STOP', last_control, MPC_calls=calls, failures=failures)
    log.close()
if __name__ == '__main__':
    main()
