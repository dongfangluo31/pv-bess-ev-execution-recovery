# EN: Second-by-second physical plant execution of the terminal recovery suffix with ramp, first-order lag, and zero-export.
# 中文：计及斜率限制、一阶滞后与防逆流的终端恢复尾段逐秒物理执行仿真器。

from bisect import bisect_right
from copy import deepcopy
from pathlib import Path
import gzip, json, math, sys, time
from protocol import write

def imports():
    from station_core.config import BatteryConfig
    from station_core.plant import realize_no_export
    from station_core.filter import advance_soc_checked
    from recovery_control import safe_target
    return (BatteryConfig, realize_no_export, advance_soc_checked, safe_target)

def map_at_entry(public, target):
    BatteryConfig, _, _, safe_target = imports()
    start = time.perf_counter_ns()
    p = public['public_packet']
    R = public['held_nominal_reserve']['value_kwh']
    c, d, detail = safe_target(target, p['ev_point_kw'], p['pv_point_kw'], public['telemetry']['soc'], BatteryConfig(**public['battery']), public['feeder_safe_kw'], R)
    return dict(charge_kw=c, discharge_kw=d, details=detail, map_wall_ns=time.perf_counter_ns() - start)

def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def totals():
    return dict(seconds=0, grid_kwh=0.0, curtail_kwh=0.0, charge_kwh=0.0, discharge_kwh=0.0, throughput_kwh=0.0, PCC_excess_kwh=0.0, PCC_excess_seconds=0, reserve_fixed_shortage_kwh_h=0.0, reserve_current_shortage_kwh_h=0.0, reserve_fixed_seconds=0, reserve_current_seconds=0, physical_projection_seconds=0, ramp_override_seconds=0, SOC_violation_seconds=0, rating_violation_seconds=0, mutex_violation_seconds=0, noexport_violation_seconds=0, fallback_seconds=0, zero_hold_state_mismatch_seconds=0, zero_hold_command_mismatch_seconds=0)

def execute(public, interface, out, controls, ready, first_mapping, branch):
    BatteryConfig, realize_no_export, advance_soc_checked, safe_target = imports()
    b = BatteryConfig(**public['battery'])
    clock = public['clock']
    q = clock['issue_s']
    T = clock['episode_end_s']
    onset = clock['onset_s']
    end = clock['bridge_end_s']
    G = public['feeder_safe_kw']
    R0 = public['held_nominal_reserve']['value_kwh']
    packets = read_gz(interface['role2_later_public_feeds']['packets']['path'])
    packet_at = {r['decision_s']: r['packet'] for r in packets}
    releases = json.loads(Path(interface['role2_later_public_feeds']['reserve_timeline']['path']).read_text(encoding='utf-8'))['entries']
    public_R = [(r['event']['forecast_issued_s'], r['event']['released_planner_reserve_requirement_kwh']) for r in releases]
    source = read_gz(interface['role3_hidden_plant']['file']['path'])
    starts = [s['start_s'] for s in source]
    if set(packet_at) != set(range(q, T, 60)) or source[0]['start_s'] != q or source[-1]['end_s'] != T:
        raise ValueError('Frozen suffix feeds missing')
    if controls is not None and (len(controls) != 58 or any((not math.isfinite(float(u)) or not -15 <= u <= 15 for u in controls))):
        raise ValueError('Invalid accepted minute-command vector')
    old = public['known_command_state']
    active = deepcopy(old['last_delivered_command'])
    queue = deepcopy(old['pending_commands'])
    highest = active['seq'] if active else 0
    soc = public['telemetry']['soc']
    power = public['telemetry']['tracked_power_kw']
    alpha = 1 - math.exp(-1 / public['dynamics']['lag_seconds'])
    ramp = public['dynamics']['ramp_kw_per_second']
    dt = 1 / 3600.0
    PTOL = 1e-09
    ETOL = 1e-08
    target = public['terminal_service']['target_energy_kwh']
    total = totals()
    phases = {p: totals() for p in ('immutable_prefix', 'bridge', 'zero_tail')}
    endpoints = {}
    control_log = []
    max_balance = 0.0
    nominal_states = []
    started = time.perf_counter()
    write(out / 'TAIL_START.json', dict(branch=branch, initial_soc=soc, initial_energy_kwh=soc * b.energy_kwh, initial_power_kw=power, issue_s=q, end_s=T, expected_actual_suffix_steps=3600, historical_pre_q_prefix_steps=0, guard_diagnostic_only=True, plan_accepted=controls is not None))
    path = out / 'actual_tail.jsonl.gz'
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as z:
            for t in range(q, T):
                R = max((entry for entry in public_R if entry[0] <= t), key=lambda entry: entry[0])[1]
                if (t - q) % 60 == 0:
                    j = (t - q) // 60
                    p = packet_at[t]
                    if p['decision_s'] != t or any((r['available_s'] > t for r in p['records'])):
                        raise ValueError('Unreleased execution observation')
                    raw_target = 0.0 if controls is None or j >= 58 else float(controls[j])
                    map_error = None
                    mapping = None
                    issue = dict(sim_s=t, decision_index=j, raw_target_kw=raw_target, public_packet_observation_id=p['observation_id'], public_R_kwh=R, soc_at_decision=soc, tracked_kw_at_decision=power, guard_feedback_used=False)
                    if j == 0 and controls is None:
                        issue.update(command_scheduled=False, reason='Initial planner refused/late: preserve already-issued queue')
                    else:
                        begun = time.perf_counter_ns()
                        try:
                            if j == 0 and first_mapping is not None:
                                mapping = first_mapping
                                c = mapping['charge_kw']
                                d = mapping['discharge_kw']
                                wall = mapping['map_wall_ns']
                            else:
                                c, d, detail = safe_target(raw_target, p['ev_point_kw'], p['pv_point_kw'], soc, b, G, R)
                                wall = time.perf_counter_ns() - begun
                                mapping = dict(charge_kw=c, discharge_kw=d, details=detail, map_wall_ns=wall)
                        except Exception as exc:
                            c = d = 0.0
                            wall = time.perf_counter_ns() - begun
                            map_error = repr(exc)
                        elapsed = ready['ready_wall_seconds'] if j == 0 else wall / 1000000000.0
                        cmd = dict(seq=public['next_new_command_seq'] + j, control_decision_s=t, issue_s=t + elapsed, effective_s=t + 60, valid_until_s=t + 120, signed_kw=d - c, charge_kw=c, discharge_kw=d)
                        timely = elapsed <= 60.0
                        if timely:
                            queue.append(cmd)
                        issue.update(mapping=mapping, mapping_error=map_error, command=cmd, command_scheduled=timely, reason='SCHEDULED' if timely else 'MAPPING_MISSED_FIXED_SLOT', map_wall_ns=wall)
                    control_log.append(issue)
                for cmd in sorted(list(queue), key=lambda c: (c['effective_s'], c['seq'])):
                    if cmd['effective_s'] <= t:
                        queue.remove(cmd)
                        if cmd['valid_until_s'] >= t and cmd['seq'] > highest:
                            active = cmd
                            highest = cmd['seq']
                fallback = active is None or active['valid_until_s'] < t
                desired = 0.0 if fallback else active['signed_kw']
                previous = power
                before = soc
                ramped = min(max(desired, previous - ramp), previous + ramp)
                lagged = previous + alpha * (ramped - previous)
                maxc = max(0.0, (b.soc_max - soc) * b.energy_kwh / (b.eta_charge * dt))
                maxd = max(0.0, b.eta_discharge * b.energy_kwh * (soc - b.soc_min) / dt)
                c = min(-lagged, b.charge_limit_kw, maxc) if lagged < 0 else 0.0
                d = min(lagged, b.discharge_limit_kw, maxd) if lagged >= 0 else 0.0
                segment = source[bisect_right(starts, t) - 1]
                if not segment['start_s'] <= t < segment['end_s']:
                    raise ValueError('Hidden source coverage gap')
                L, pv = (segment['ev_kw'], segment['pv_kw'])
                a = realize_no_export(c, d, L, pv)
                power = a.discharge_kw - a.charge_kw
                soc = advance_soc_checked(soc, a.charge_kw, a.discharge_kw, b, dt)
                E = soc * b.energy_kwh
                deliverable = b.eta_discharge * (E - b.soc_min * b.energy_kwh)
                projection = abs(power - lagged)
                ramp_override = max(abs(power - previous) - ramp, 0.0)
                balance = abs(a.grid_import_kw + pv - a.pv_curtail_kw + a.discharge_kw - L - a.charge_kw)
                max_balance = max(max_balance, balance)
                row = dict(sim_s=t, end_s=t + 1, source_segment_id=segment['segment_id'], soc_pre=before, soc_post=soc, energy_pre_kwh=before * b.energy_kwh, energy_post_kwh=E, previous_tracked_kw=previous, desired_signed_kw=desired, ramped_kw=ramped, lagged_kw=lagged, charge_kw=a.charge_kw, discharge_kw=a.discharge_kw, tracked_kw=power, grid_kw=a.grid_import_kw, pv_curtail_kw=a.pv_curtail_kw, ev_kw=L, pv_kw=pv, applied_command_seq=None if active is None else active['seq'], fallback=fallback, physical_clip_kw=projection, ramp_override_kw=ramp_override, fixed_R0_kwh=R0, current_public_R_kwh=R, deliverable_energy_kwh=deliverable, guard_diagnostic_only=True, source_quality=segment)
                z.write((json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode())
                phase = 'immutable_prefix' if t < onset else 'bridge' if t < end else 'zero_tail'
                for summary in (total, phases[phase]):
                    summary['seconds'] += 1
                    for key, value in [('grid_kwh', a.grid_import_kw), ('curtail_kwh', a.pv_curtail_kw), ('charge_kwh', a.charge_kw), ('discharge_kwh', a.discharge_kw), ('throughput_kwh', a.charge_kw + a.discharge_kw)]:
                        summary[key] += value * dt
                    summary['PCC_excess_kwh'] += max(a.grid_import_kw - G, 0.0) * dt
                    summary['PCC_excess_seconds'] += int(a.grid_import_kw > G + PTOL)
                    for suffix, value in [('fixed', R0), ('current', R)]:
                        summary['reserve_' + suffix + '_shortage_kwh_h'] += max(value - deliverable, 0.0) * dt
                        summary['reserve_' + suffix + '_seconds'] += int(deliverable < value - PTOL)
                    summary['physical_projection_seconds'] += int(projection > PTOL)
                    summary['ramp_override_seconds'] += int(ramp_override > PTOL)
                    summary['SOC_violation_seconds'] += int(soc < b.soc_min - PTOL or soc > b.soc_max + PTOL)
                    summary['rating_violation_seconds'] += int(a.charge_kw > b.charge_limit_kw + PTOL or a.discharge_kw > b.discharge_limit_kw + PTOL)
                    summary['mutex_violation_seconds'] += int(min(a.charge_kw, a.discharge_kw) > PTOL)
                    summary['noexport_violation_seconds'] += int(a.grid_import_kw < -PTOL or balance > PTOL or a.discharge_kw > max(L - pv + a.charge_kw, 0.0) + PTOL)
                    summary['fallback_seconds'] += int(fallback)
                    if t >= end:
                        summary['zero_hold_state_mismatch_seconds'] += int(abs(E - target) > ETOL or abs(power) > PTOL)
                        summary['zero_hold_command_mismatch_seconds'] += int(abs(desired) > PTOL)
                if t + 1 in (onset, end, T):
                    endpoints[str(t + 1)] = dict(energy_kwh=E, power_kw=power, energy_error_kwh=E - target)
    for summary in [total, *phases.values()]:
        summary['common_cost_yuan'] = 0.6 * summary['grid_kwh'] + 0.08 * summary['throughput_kwh']
        summary['operating_cost_with_curtail_yuan'] = summary['common_cost_yuan'] + 0.02 * summary['curtail_kwh']
    write(out / 'ACTUAL_CONTROL_EVENTS.json', dict(branch=branch, decisions=control_log, pending_after_T=queue, initial_deadline_miss=ready['deadline_missed'], later_mapping_missed_slots=sum((not x['command_scheduled'] for x in control_log[1:])), mapper_wall_seconds=sum((x.get('map_wall_ns', 0) for x in control_log)) / 1000000000.0))
    final = dict(branch=branch, status='ACTUAL_SUFFIX_RECORDED', actual_suffix_steps=total['seconds'], historical_prefix_replay_steps=0, new_MPC=0, initial_energy_kwh=public['stored_energy_kwh'], final_energy_kwh=soc * b.energy_kwh, final_power_kw=power, final_energy_error_kwh=soc * b.energy_kwh - target, totals=total, phase_totals=phases, endpoints=endpoints, terminal_energy_and_power_ok=abs(soc * b.energy_kwh - target) <= ETOL and abs(power) <= PTOL, bridge_endpoint_ok=abs(endpoints[str(end)]['energy_error_kwh']) <= ETOL and abs(endpoints[str(end)]['power_kw']) <= PTOL, final_zero_hold_ok=total['zero_hold_state_mismatch_seconds'] == total['zero_hold_command_mismatch_seconds'] == 0, maximum_power_balance_residual_kw=max_balance, wall_seconds=time.perf_counter() - started, actual_guard_feedback_to_commands=False)
    write(out / 'ACTUAL_TAIL_RESULT.json', final)
    return final
