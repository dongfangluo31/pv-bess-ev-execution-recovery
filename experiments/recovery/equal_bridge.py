# EN: Equal-hold midpoint bridge trajectory synthesizer under causal point-persistence load/PV feeds.
# 中文：因果持续性负荷/光伏点预测下的对称保持中点桥接终端恢复轨迹合成器。

import math
import time
from protocol import write, record, digest
from recovery_control import predict_prefix
from queue_summary import summarize_queue
from validated_bridge import solve_bridge, make_command_schedule, json_safe
from catalog_encoding import encode_charge_tail_zero_once
VERSION = 'ORIGINAL_EQUAL_MIDPOINT_NATIVE'
N = 1740
ENERGY_TOL = 1e-08
POWER_TOL = 1e-09

def synthesize_equal(cfg, battery, packet, measurement, queue_state, now_s, empirical, L, pv, reserve, out):
    started = time.perf_counter()
    spec = cfg['recovery_spec']
    T = cfg['duration_s']
    onset = T - 3540
    if now_s != T - 3600:
        raise ValueError('Original equal bridge is selected once at q=T-3600')
    if spec['target_stored_energy_kwh'] != battery.energy_kwh * battery.soc_initial:
        raise ValueError('Original initial shared energy target changed')
    if not isinstance(empirical, dict) or empirical.get('forecast_issued_s') != now_s:
        raise ValueError('Original q is quarter aligned and requires its released forecast')
    if empirical.get('range_kind', 'pointwise') != 'pointwise':
        raise ValueError('Released recovery forecast must be pointwise')
    if empirical['observation_id'] != packet['observation_id']:
        raise ValueError('Forecast observation does not bind current packet')
    if not all((math.isfinite(x) and x >= 0 for x in (L, pv, reserve))):
        raise ValueError('Finite nonnegative current L/PV/reserve required')
    alpha = 1.0 if spec['lag_seconds'] == 0 else 1 - math.exp(-1 / spec['lag_seconds'])
    prefix = predict_prefix(measurement, queue_state, now_s, battery, L, pv, spec['ramp_kw_per_second'], spec['lag_seconds'])
    limits = dict(e_min=battery.soc_min * battery.energy_kwh, e_max=battery.soc_max * battery.energy_kwh, charge_max=battery.charge_limit_kw, discharge_max=battery.discharge_limit_kw, load_min=max(L - pv, 0.0), load_max=L, pv_min=pv, import_max=cfg['feeder_safe_kw'], reserve_min=reserve)
    witness = solve_bridge(prefix['predicted_power_kw'], prefix['predicted_energy_kwh'], spec['target_stored_energy_kwh'], N, 1 / 3600.0, alpha, spec['ramp_kw_per_second'], battery.eta_charge, battery.eta_discharge, limits=limits, command_bounds=(-battery.charge_limit_kw, battery.discharge_limit_kw), energy_tol=ENERGY_TOL, power_tol=POWER_TOL)
    unencoded_witness = witness
    witness = encode_charge_tail_zero_once(witness)
    bridge_status = witness['status']
    zero_tail = []
    conflicts = []
    if bridge_status == 'reachable':
        p, e = (witness['powers'][-1], witness['energies'][-1])
        for j in range(60):
            p += alpha * max(-spec['ramp_kw_per_second'], min(spec['ramp_kw_per_second'], -p))
            e -= (p / battery.eta_discharge if p >= 0 else p * battery.eta_charge) / 3600.0
            slacks = dict(charge=limits['charge_max'] + p, discharge=limits['discharge_max'] - p, absorption=limits['load_min'] - p, energy_lower=e - limits['e_min'], energy_upper=limits['e_max'] - e, reserve=battery.eta_discharge * (e - limits['e_min']) - reserve, grid=limits['import_max'] - max(L - pv - p, 0.0))
            if any((not math.isfinite(v) or v < 0 for v in slacks.values())):
                conflicts.append(dict(step=j, slacks=slacks))
            if abs(p) > POWER_TOL or abs(e - spec['target_stored_energy_kwh']) > ENERGY_TOL:
                conflicts.append(dict(step=j, reason='original_endpoint_tolerance_failed'))
            zero_tail.append(dict(step=j, power_kw=p, energy_kwh=e, target_kw=0.0))
        if conflicts:
            witness = dict(witness, status='constraint_conflict', reason='nominal_zero_tail_failed', bridge_only_status=bridge_status)
    risk = dict(scope='EMPIRICAL_POINTWISE_BOX_NOT_A_NOMINAL_OR_COVERAGE_CERTIFICATE', support_is_certified=empirical.get('support_is_certified') is True, forecast_issued_s=now_s, boxes=[])
    h = empirical['horizon']
    ready = empirical.get('empirical_pointwise_ready')
    for k in sorted(set(((onset + j - now_s) // 900 for j in range(2 * N + 60)))):
        if k < 0 or k >= len(h['ev_lower_kw']) or (ready is not None and (not ready[k])):
            risk['boxes'].append(dict(lead=k, status='UNSUPPORTED'))
            continue
        lower = max(-battery.charge_limit_kw, h['ev_upper_kw'][k] - h['pv_lower_kw'][k] - cfg['feeder_safe_kw'])
        upper = min(battery.discharge_limit_kw, max(h['ev_lower_kw'][k] - h['pv_upper_kw'][k], 0.0))
        risk['boxes'].append(dict(lead=k, lower_kw=lower, upper_kw=upper, status='EMPTY_ROBUST_POWER_BOX' if lower > upper else 'NONEMPTY_NOT_COVERAGE_CERTIFIED'))
    reachable = witness['status'] == 'reachable'
    sign = cfg['positive_inventory_challenge']['offset_sign']
    predicted_delta = prefix['predicted_energy_kwh'] - spec['target_stored_energy_kwh']
    plan = dict(version=VERSION, arm=cfg['arm'], run_id=cfg['run_id'], issue_s=now_s, onset_s=onset, bridge_end_s=T - 60, episode_end_s=T, N=N, initial_shared_energy_target_kwh=spec['target_stored_energy_kwh'], requested_inventory_sign=sign, predicted_bridge_entry_deviation_kwh=predicted_delta, predicted_requested_sign_nontrivial=sign * predicted_delta >= 0.005 * battery.energy_kwh, minimum_actual_nontrivial_error_kwh=0.005 * battery.energy_kwh, actual_nontrivial_entry_status='EVALUATED_AT_ONSET_Q_PLUS_60', telemetry=measurement, known_command_state=summarize_queue(queue_state), released_observation_id=packet['observation_id'], released_packet_sha256=digest(packet), forecast=empirical, held_forecast_sha256=digest(empirical), empirical_pointwise_risk=risk, guard_reserve_requirement_kwh=reserve, nominal_current_point=dict(load_kw=L, pv_kw=pv, reserve_kwh=reserve), predicted_entry=prefix, hold_rule=dict(N=N, alpha=alpha), witness=json_safe(witness), unencoded_witness=json_safe(unencoded_witness), finite_precision_encoding_rule='CHARGE_TAIL_ZERO_ENCODING_ONCE', calendar=make_command_schedule(witness, onset, N, zero_until_step=T) if reachable else [], zero_tail=zero_tail, zero_tail_conflicts=conflicts, bridge_only_status=bridge_status, conditional_nominal_witness=reachable, full_bridge_and_tail_certificate=False, robust_recovery_certificate=False, certificate_scope='ONE_SHOT_ORIGINAL_EQUAL_BRIDGE_CONDITIONAL_CURRENT_POINT_PERSISTENCE', support_is_certified=False, actual_future_consumed=False, synthesis_wall_seconds=time.perf_counter() - started, actual_new_optimizer_calls=0, bridge_function_evaluations=witness.get('function_evaluations', 0), bridge_formula_step_evaluations=2 * N * unencoded_witness.get('function_evaluations', 0), original_encoded_replay_steps=len(unencoded_witness['powers']), finite_precision_encoding_replay_steps=2 * N if witness.get('encoding_adjustment') else 0, finite_precision_encoding_formula_steps=2 * N if witness.get('encoding_adjustment') else 0, prefix_prediction_steps=len(prefix['rows']), encoded_bridge_prediction_steps=len(witness['powers']), zero_tail_prediction_steps=len(zero_tail), refusal_policy='Keep ordinary F1 after refusal; final raw zero commands at T-120 and T-60. Original guard abort latches and subsequent commands are zero through the unchanged queue. No later recovery synthesis.')
    path = out / 'RECOVERY_PLAN.json'
    write(path, plan)
    return (plan, record(path))
