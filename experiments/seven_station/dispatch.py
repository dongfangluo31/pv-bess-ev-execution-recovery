# EN: Closed-loop BESS dispatch policy and real-time constraint enforcement.
# 中文：闭环储能调度策略与实时物理约束执行模块。

import time
SCRIPT_STARTED_PERF = time.perf_counter()
from dataclasses import asdict, replace
from pathlib import Path
import argparse, json, math, os, time, warnings
from scene_io import ROOT, LIB, BATCH, ARMS, MAXIMUM, STATIONS, BUS_MAP, load, write, record, require, digest, safe, utc
from scene_core import core
from scene_physics import DT, common_action, scalar_target, apply_correction

def verify():
    if not __debug__:
        raise RuntimeError('Normal non-O interpreter required')
    p = load(BATCH / 'PREPARED.json')
    for r in p['source_files'] + p['input_files'] + p['external_dependencies']:
        require(r)
    return p

def request_snapshot(kwargs):
    A = kwargs['constraints'].A.tocoo()
    order = sorted(range(len(A.data)), key=lambda i: (int(A.row[i]), int(A.col[i])))
    return safe(dict(c=kwargs['c'], integrality=kwargs['integrality'], lower_bounds=kwargs['bounds'].lb, upper_bounds=kwargs['bounds'].ub, A=dict(shape=list(A.shape), row=[int(A.row[i]) for i in order], col=[int(A.col[i]) for i in order], data=[float(A.data[i]) for i in order]), row_lower=kwargs['constraints'].lb, row_upper=kwargs['constraints'].ub, options=dict(kwargs['options'])))

def new_network(data):
    import pandapower as pp
    from pandapower.networks.power_system_test_cases import sorted_from_json
    require(data['network_asset'])
    net = sorted_from_json(data['network_asset']['path'])
    base = net.load[['p_mw', 'q_mvar']].copy()
    if list(net.bus.index) != list(range(33)) or len(net.line) != 37 or len(net.trafo) != 0:
        raise ValueError('Original case33 topology domain')
    station_loads = {s: int(pp.create_load(net, bus=BUS_MAP[s], p_mw=0.0, q_mvar=0.0, name='network_replay_' + s)) for s in STATIONS}
    return (net, base, station_loads)

def execute(rd, counts, attempts, state, begun, attempt):
    import numpy as np
    from station_core import model
    from station_core.config import Method
    from capture_runtime import make_callbacks
    from current_jacobian import assemble, solve_assembled, verify_solution, projection
    real_milp = model.milp
    ac_call, lp_call = make_callbacks(rd, state, attempt, BATCH.name, max_ac_calls=360, max_lp_calls=180)
    startup = time.perf_counter() - begun
    arm_results = []
    active_station = None
    active_context = None

    def captured_milp(**kwargs):
        snap = request_snapshot(kwargs)
        cid = counts['MPC'] + 1
        context = dict(active_context, profile_id=active_station['profile_id'], current_soc=active_station['current_soc'], station_input_sha256=digest(active_station), request_sha256=digest(snap))
        attempt('MPC', cid, context)
        tick = time.perf_counter()
        error = None
        result = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = real_milp(**kwargs)
                write(rd / 'mpc' / f'{cid:03d}_RETURN_KNOWN.json', dict(run_id=BATCH.name, call_id=cid, context=context, backend_returned=True, backend_seconds=time.perf_counter() - tick, snapshot_state='PENDING'))
                snap['result'] = {k: v for k, v in result.items()}
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
                snap['result'] = None
        write(rd / 'mpc' / f'{cid:03d}.json', dict(run_id=BATCH.name, call_id=cid, context=context, station_input=active_station, solver_wall_seconds=time.perf_counter() - tick, exception=error, warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught], snapshot=snap))
        if error is not None:
            raise RuntimeError('MPC exception retained')
        if result.status != 0 or result.x is None or result.mip_gap > 0.001 + 1e-12:
            raise RuntimeError('Dispatch solve did not meet its quality limit')
        point = np.asarray(result.x)
        values = kwargs['constraints'].A @ point
        if not np.isfinite(point).all() or np.max(kwargs['constraints'].lb - values) > 1e-7 or np.max(values - kwargs['constraints'].ub) > 1e-7:
            raise RuntimeError('Dispatch constraint residual')
        return result
    model.milp = captured_milp
    global_error = None
    try:
        for arm in ARMS:
            arm_start = time.perf_counter()
            from runtime_gate import arm_begin
            arm_begin(arm, arm_start)
            verify()
            input_ref = load(BATCH / 'INPUT_REFERENCE.json')
            require(input_ref)
            data = load(input_ref['path'])
            stations = data['stations']
            cores = {s['profile_id']: core(s['profile_scale']) for s in stations}
            soc = {s['profile_id']: s['initial_soc'] for s in stations}
            nominal = {s['profile_id']: dict(s['initial_held_action']) for s in stations}
            held = {s: dict(v) for s, v in nominal.items()}
            held_id = 'FIXED_SOURCE_STEP401_ACTUAL'
            held_valid_until = 60
            queue = {}
            trace = (rd / 'arms' / f'{arm}_minutes.jsonl').open('x', encoding='utf8')
            timing = (rd / 'arms' / f'{arm}_timing.jsonl').open('x', encoding='utf8')
            net, background, station_loads = new_network(data)
            done = 0
            failure = None
            fallbacks = []
            nominal_rounds = []
            first_effect = None
            linear_solves = 0
            initial_soc = dict(soc)
            before_counts = dict(counts)
            try:
                for minute in range(180):
                    if time.perf_counter() - arm_start >= 600:
                        raise TimeoutError('600s arm work budget reached; remaining dependent minutes NOT_RUN')
                    t = minute * 60
                    hour = minute // 60
                    frame_start = time.perf_counter()
                    from runtime_gate import round_begin
                    round_begin(minute, frame_start)
                    active_context = dict(arm=arm, minute=minute, model_time_seconds=t, source_step=402 + hour)
                    if minute in queue:
                        q = queue.pop(minute)
                        held = q['commands']
                        held_id = q['command_id']
                        held_valid_until = q['valid_until_seconds']
                        if q['network_correction_kw'] > 1e-09 and first_effect is None:
                            first_effect = t
                    elif t >= held_valid_until:
                        held = {s: dict(charge_kw=0.0, discharge_kw=0.0) for s in STATIONS}
                        held_id = 'COMMON_EXPIRED_COMMAND_ZERO'
                        held_valid_until = t + 60
                        fallbacks.append(dict(minute=minute, kind='COMMON_COMMAND_EXPIRED_ZERO'))
                    physical = {}
                    next_soc = {}
                    imports = {}
                    for s in stations:
                        sid = s['profile_id']
                        h = s['hours'][hour]
                        a = common_action(held[sid], soc[sid], h['actual_ev_kw'], h['actual_pv_kw'], cores[sid].battery)
                        physical[sid] = a
                        next_soc[sid] = a['soc_post']
                        imports[sid] = a['grid_import_kw']
                    factor = stations[0]['hours'][hour]['background_factor'] * data['background_peak']
                    for col in ('p_mw', 'q_mvar'):
                        net.load.loc[background.index, col] = background[col] * factor
                    for sid, i in station_loads.items():
                        net.load.loc[i, 'p_mw'] = imports[sid] / 1000.0
                        net.load.loc[i, 'q_mvar'] = 0.0
                    state['context'] = dict(active_context, role='ACTUAL_CURRENT_APPLIED_ACTION', input_sha256=record(input_ref['path'])['sha256'], held_command_id=held_id)
                    snap, ac_review, ac_ref = ac_call(net, 'actual_minute', imports, station_loads)
                    voltage = {int(x['bus']): float(x['vm_pu']) for x in snap['res_bus']}
                    lower_def = max((max(data['voltage_lower_pu'] - voltage[i], 0.0) for i in range(1, 33)))
                    upper_exc = max((max(voltage[i] - data['voltage_upper_by_bus'][str(i)], 0.0) for i in range(1, 33)))
                    round_status = None
                    if minute % 60 == 0:
                        proposal = {}
                        outcomes = []
                        round_status = 'ACCEPTED'
                        for s in stations:
                            if time.perf_counter() - frame_start >= 60:
                                round_status = 'LATE_NOT_ACCEPTED'
                                break
                            sid = s['profile_id']
                            active_station = dict(profile_id=sid, core=asdict(cores[sid]), horizon=s['hours'][hour]['horizon'], current_soc=soc[sid])
                            cid = counts['MPC'] + 1
                            try:
                                decision = model.solve_first_step(Method.GS_F1, model.HorizonData(**active_station['horizon']), soc[sid], cores[sid])
                                proposal[sid] = dict(charge_kw=decision.charge_kw, discharge_kw=decision.discharge_kw)
                                outcomes.append(dict(profile_id=sid, call_id=cid, status='ACCEPTED', decision=asdict(decision)))
                            except Exception as exc:
                                outcomes.append(dict(profile_id=sid, call_id=cid if counts['MPC'] >= cid else None, status='REJECTED', exception=dict(type=type(exc).__name__, message=str(exc))))
                                round_status = 'FAILED_KEEP_PREVIOUS_NOMINAL'
                                break
                        if len(proposal) == 7 and time.perf_counter() - frame_start < 60:
                            nominal = proposal
                        else:
                            if round_status == 'ACCEPTED':
                                round_status = 'LATE_NOT_ACCEPTED'
                            fallbacks.append(dict(minute=minute, kind=round_status))
                        nominal_rounds.append(dict(minute=minute, status=round_status, results=outcomes, not_run_profiles=[s for s in STATIONS if s not in {r['profile_id'] for r in outcomes}]))
                    if time.perf_counter() - frame_start >= 60:
                        raise TimeoutError('Control round missed its fixed 60s effect slot; dependent future stopped')
                    current_stations = [dict(profile_id=s['profile_id'], observation={key: s['hours'][hour][key] for key in ('actual_ev_kw', 'actual_pv_kw', 'reserve_requirement_kwh')}) for s in stations]
                    scalar, caps, scalar_log = scalar_target(current_stations, cores, nominal, next_soc, data['shared_beta'])
                    target = {s: dict(charge_kw=r['charge_kw'], discharge_kw=r['discharge_kw']) for s, r in scalar.items()}
                    network_status = 'NOT_APPLICABLE_SCALAR_ARM'
                    lp_ref = None
                    jac_ref = None
                    correction = 0.0
                    prediction = None
                    if arm == 'NETWORK_JOINT':
                        J, jrhs = assemble(snap)
                        jfile = rd / 'jacobian' / f'{minute:03d}_REQUEST.npz'
                        with jfile.open('xb') as f:
                            np.savez_compressed(f, J=J, rhs=jrhs)
                        from runtime_gate import given_j7_before
                        linear_solves = given_j7_before(attempts, arm, minute, record(jfile), ac_ref, BATCH.name)
                        state['J7'] += 1
                        solution = solve_assembled(J, jrhs)
                        jresult = rd / 'jacobian' / f'{minute:03d}_RETURN.npz'
                        with jresult.open('xb') as f:
                            np.savez_compressed(f, solution=solution)
                        j = verify_solution(J, jrhs, solution)
                        jac_ref = dict(request=record(jfile), response=record(jresult), source_ac=ac_ref, max_linear_residual=j['max_linear_residual'], evaluation_point=j['evaluation_point'])
                        A, b, pred = projection(snap, j, {s: r['grid_import_kw'] for s, r in scalar.items()}, imports, data['voltage_lower_pu'], data['voltage_upper_by_bus'])
                        prediction = pred.tolist()
                        state['context'] = dict(active_context, role='NEXT_COMMAND_CURRENT_INFORMATION', source_ac=ac_ref, jacobian=jac_ref, jacobian_evaluation=j['evaluation_point'], library_returned_J_used=False, row_order=[dict(bus=i, sense=sense) for sense in ('LOWER', 'UPPER') for i in range(1, 33)], predicted_application_soc=next_soc, actual_source_is_current_hour_only=True)
                        try:
                            delta, review, lp_ref = lp_call('all32_lower_and_upper_current_J', np.arange(64), -A, b, np.array([caps[s]['total_cap_kw'] for s in STATIONS]))
                            if np.max(A @ delta - b) > 1e-10:
                                raise ValueError('Full32 upper/lower linear gate failed')
                            target = apply_correction(scalar, caps, delta)
                            correction = float(np.sum(np.maximum(delta, 0.0)))
                            network_status = 'LINEAR_MODEL_ACCEPTED_NOT_AC_FUTURE_GUARANTEE'
                        except Exception as exc:
                            cid = state['lp']
                            p = rd / 'lp' / f'{cid:03d}.json'
                            raw = load(p) if p.exists() else None
                            status = (raw or {}).get('snapshot', {}).get('result', {})
                            status = status.get('status') if isinstance(status, dict) else None
                            if status not in (1, 2, 3, 4):
                                raise
                            network_status = 'SCALAR_FALLBACK_UNCERTIFIED_LP'
                            fallbacks.append(dict(minute=minute, kind=network_status, native_status=status, exception=dict(type=type(exc).__name__, message=str(exc))))
                            lp_ref = record(p)
                            correction = 0.0
                    for s in stations:
                        sid = s['profile_id']
                        h = s['hours'][hour]
                        a = common_action(target[sid], next_soc[sid], h['actual_ev_kw'], h['actual_pv_kw'], cores[sid].battery)
                        if a['common_charge_clip_kw'] > 1e-09 or a['common_discharge_total_clip_kw'] > 1e-09:
                            raise ValueError('Issued target requires undeclared extra clip')
                    command_id = f'{arm}:{minute:03d}'
                    elapsed = time.perf_counter() - frame_start
                    late = elapsed > 60.0
                    issued = dict(command_id=command_id, issue_model_seconds=t, ready_model_seconds=t + elapsed, effect_seconds=t + 60, valid_until_seconds=t + 120, commands=target, network_correction_kw=correction, late=late)
                    if not late:
                        queue[minute + 1] = issued
                    else:
                        fallbacks.append(dict(minute=minute, kind='LATE_COMMAND_DROPPED'))
                    frame = dict(run_id=BATCH.name, arm=arm, minute=minute, model_seconds=t, source_hour=hour, source_step=402 + hour, soc_pre=soc, held_command_id=held_id, held_command=held, actual=physical, soc_post=next_soc, current_AC=ac_ref, minimum_voltage_pu=min(voltage.values()), lower_voltage_deficit_pu=lower_def, upper_voltage_excess_pu=upper_exc, network_status=network_status, scalar_target=scalar, scalar_details=scalar_log, network_caps=caps, jacobian=jac_ref, linear_prediction_before_correction=prediction, LP=lp_ref, nominal=nominal, nominal_round_status=round_status, issued=issued, control_seconds_through_command=elapsed, current_scalar_information=current_stations, grid_cost_yuan=sum((physical[s['profile_id']]['grid_import_kw'] * s['hours'][hour]['price_yuan_per_kwh'] * DT for s in stations)), throughput_kwh=sum(((a['charge_kw'] + a['discharge_kw']) * DT for a in physical.values())))
                    trace.write(json.dumps(safe(frame), separators=(',', ':')) + '\n')
                    trace.flush()
                    os.fsync(trace.fileno())
                    soc = next_soc
                    done += 1
                    full_round = time.perf_counter() - frame_start
                    final_late = full_round > 60.0
                    if minute + 1 in queue:
                        if final_late:
                            queue.pop(minute + 1)
                        else:
                            queue[minute + 1] = dict(queue[minute + 1], ready_model_seconds=t + full_round, late=False)
                    timing.write(json.dumps(dict(minute=minute, frame_seconds_through_journal=full_round, ready_model_seconds=t + full_round, command_admitted=not final_late, preliminary_late=late, final_late=final_late, own_timing_row_write_after_measurement=True), separators=(',', ':')) + '\n')
                    timing.flush()
                    os.fsync(timing.fileno())
                    if final_late:
                        raise TimeoutError('Late command retained/dropped; stop dependent future rather than overlapping processing')
            except Exception as exc:
                failure = dict(type=type(exc).__name__, message=str(exc), minute=done)
            finally:
                trace.close()
                timing.close()
            result = dict(arm=arm, completed_minutes=done, planned_minutes=180, remaining_minutes_not_run=list(range(done, 180)), failure=failure, fallbacks=fallbacks, initial_soc=initial_soc, final_soc=soc, unexecuted_end_queue=queue, nominal_rounds=nominal_rounds, first_positive_network_command_application_seconds=first_effect, call_counts={k: counts[k] - before_counts[k] for k in counts}, current_J_linear_systems=linear_solves, current_J_right_hand_sides=7 * linear_solves, arm_seconds_before_result_write=time.perf_counter() - arm_start, thermal_gate='OPEN_UNRATED', scope='Ideal one-minute constant-power digital plant; no ramp/HIL/field claim')
            write(rd / 'arms' / f'{arm}_RESULT.json', result)
            result['arm_full_seconds_through_result_write'] = time.perf_counter() - arm_start
            arm_results.append(result)
        verify()
    except Exception as exc:
        global_error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        model.milp = real_milp
        attempts.close()
        write(rd / 'RUN_COMPLETE.json', dict(run_id=BATCH.name, status='COMPLETED' if global_error is None and len(arm_results) == 2 else 'FAILED', global_failure=global_error, maximum_calls=MAXIMUM, actual_calls=counts, arms=arm_results, planned_arm_count=2, planned_minutes_per_arm=180, actual_J7=state['J7'], maximum_J7=180, startup_seconds=startup, whole_parent_seconds_before_completion_write=time.perf_counter() - begun, old_network_37AC_36LP_and_feedback_7MPC_1LP_1AC_preserved=True, new_source_lp_calls=0, old_method_static_or_CIL_counts_not_reclassified=True, files=[record(p) for p in sorted(rd.rglob('*')) if p.is_file()]))
    return 0 if global_error is None else 1
