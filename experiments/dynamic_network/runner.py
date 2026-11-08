# EN: End-to-end closed-loop station simulation runner emitting step traces, shadow-filter counterfactuals, and solver diagnostics.
# 中文：端到端场站闭环仿真运行器，输出步级轨迹、同状态影子修正对照与求解器残差诊断。

import time
SCRIPT_START = time.perf_counter_ns()
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import argparse, ast, json, os, math, shutil, ctypes
from bootstrap import ROOT, WORK, wire
wire()
from records import read, write, record, require, Journal, STATIONS, ARMS, CAPS, encoded
from dispatch import Dispatch

def background(case, hour, shape_source):
    shape = json.loads(require(shape_source).read_text(encoding='utf-8'))
    start = datetime.fromisoformat(case['stations'][0]['hours'][hour]['timestamp']).hour
    return [float(shape[(start + k) % 24]) * case['background_peak'] for k in range(24)]

def voltage(case, ac):
    if not ac['valid']:
        return None
    import numpy as np
    V = np.hypot(ac['snapshot']['V']['real'], ac['snapshot']['V']['imag'])
    low = [max(case['voltage_lower_pu'] - float(V[i]), 0.0) for i in range(1, 33)]
    high = [max(float(V[i]) - case['voltage_upper_by_bus'][str(i)], 0.0) for i in range(1, 33)]
    return dict(V=V.tolist(), lower=low, upper=high, positive_bus_count=sum((l > 0 or u > 0 for l, u in zip(low, high))), minimum=float(min(V[1:])), thermal='UNAVAILABLE_99999')

def arm_run(batch, plan, pref, authref):
    import numpy as np
    import scipy.optimize as opt
    import pandapower as pp
    from station_core import model as old_model
    from model import config, build, solve
    from physical import step, observation, prefix60
    from scene_physics import scalar_target, common_action
    from network_runtime import Network
    saved = (old_model.milp, opt.milp, opt.linprog, opt.minimize, pp.runpp)
    arm = plan['arm']
    out = batch / 'execution' / arm
    out.mkdir()
    log = Journal(out / 'events.jsonl', batch.name, arm)
    trace = (out / 'SECONDS.jsonl').open('xb')
    case = read(plan['source_case']['path'])
    configs = {s['profile_id']: config(s) for s in case['stations']}
    state = {s['profile_id']: s['initial_soc'] for s in case['stations']}
    initial = dict(state)
    held = {s['profile_id']: dict(s['initial_held_action']) for s in case['stations']}
    tracked = {sid: a['discharge_kw'] - a['charge_kw'] for sid, a in held.items()}
    dispatch = Dispatch(held)
    nominal = None
    nominal_hour = None
    failures = []
    decisions = []
    done = 0
    fatal = None
    net = None
    begin = time.perf_counter()
    control_seconds = 0.0
    evaluation_seconds = 0.0
    offline_ac_seconds = 0.0
    aggregate = dict(grid_kwh=0.0, charge_kwh=0.0, discharge_kwh=0.0, curtail_kwh=0.0, common_cost_yuan=0.0, throughput_kwh=0.0, reserve_shortage_exposure_kwh_h=0.0, PCC_excess_kwh=0.0, voltage_lower_deficit_pu_h=0.0, voltage_upper_excess_pu_h=0.0, voltage_unknown_seconds=0, positive_voltage_bus_seconds=0, physical_clip_kwh=0.0, minimum_voltage_pu=None)
    write(out / 'RUN_STARTED.json', dict(run_id=batch.name, arm=arm, prepared=pref, authorization=authref, source_case=plan['source_case'], initial_soc=initial, initial_tracked_power_kw=tracked, physical_seconds=10800, plant_data='FROZEN_HOUR_CONSTANT_DIGITAL_SCENARIO_NOT_SECOND_MEASUREMENT', hard_realtime=False))
    try:
        init_start = time.perf_counter()
        net = Network(case, plan['source_case'], out, log, arm)
        initialization_seconds = time.perf_counter() - init_start
        for second in range(10800):
            eval_start = time.perf_counter()
            minute = second // 60
            hour = second // 3600
            is_decision = second % 60 == 0
            packet = observation(case, second)
            if is_decision:
                held, held_info = dispatch.action(minute)
            before = dict(state)
            prior_power = dict(tracked)
            physical = {sid: step(held[sid], state[sid], tracked[sid], packet['stations'][sid]['actual_ev_kw'], packet['stations'][sid]['actual_pv_kw'], cfg.battery) for sid, cfg in configs.items()}
            imports = {sid: a['grid_import_kw'] for sid, a in physical.items()}
            ac_start = time.perf_counter()
            ac = net.ac(second, held_info['held_command_id'], imports)
            ac_elapsed = time.perf_counter() - ac_start
            v = voltage(case, ac)
            if not is_decision:
                offline_ac_seconds += ac_elapsed
            if is_decision:
                tick_ns = time.perf_counter_ns() - int((time.perf_counter() - eval_start) * 1000000000.0)
                opening = dispatch.opening(minute)
                reason = None
                proposal = None
                planner = []
                fast = None
                prefix = None
                Jref = None
                if second % 3600 == 0:
                    nominal = None
                    nominal_hour = None
                    log.event('CURRENT_POINT_AND_HORIZON_RELEASE', second, packet=packet, future_actual_available=False)
                if not opening['control_opportunity_available']:
                    reason = 'BUSY_PREVIOUS_WORK'
                elif control_seconds >= 600.0:
                    reason = 'CONTROLLER_BUDGET_EXHAUSTED'
                elif not ac['valid']:
                    reason = 'CURRENT_AC_FAILED'
                remaining = second + 60 - opening['engine_start_model_ns'] / 1000000000.0 - (time.perf_counter_ns() - tick_ns) / 1000000000.0
                deadline = time.perf_counter() + max(remaining, 0.0)
                if reason is None and remaining <= 0:
                    reason = 'OBSERVATION_LATE'
                if reason is None:
                    try:
                        prefix = prefix60(second, before, prior_power, held, held_info['valid_until_s'], packet, configs)
                        prefix_ref = write(out / f'PREFIX_{minute:03d}.json.gz', prefix)
                        app = prefix['predicted_soc']
                        if second % 3600 == 0:
                            information = dict(second=second, packet=packet, source_case=plan['source_case'], prefix=prefix_ref, initial_state=before, initial_tracked_kw=prior_power, application_soc=app, first_step_hours=59 / 60, conditioning='CURRENT_POINT_PERSISTENCE', preserved_forecast_reserve=True, fixed_prefix_common_cost_yuan=prefix['fixed_prefix_common_cost_yuan'], fixed_prefix_resources=prefix['fixed_prefix_resources'], optimizer_objective_scope='Original weighted GS suffix24-index objective; fixed prefix independent of decision; actual episode cost counts prefix once')
                            write(out / f'INFORMATION_{minute:03d}.json', information)
                            if arm == 'JOINT_H24_NETWORK_MPC':
                                K, req, resp = net.jacobian(minute, ac['snapshot'], ac['raw'], 8)
                                Jref = dict(request=req, response=resp)
                                snap = ac['snapshot']
                                V = np.hypot(snap['V']['real'], snap['V']['imag'])[1:]
                                network = dict(voltage=V.tolist(), Sv=K[32:, :7].tolist(), Sb=K[32:, 7].tolist(), imports=[imports[s] for s in STATIONS], background=background(case, hour, plan['background_shape_source_ref']), lower=case['voltage_lower_pu'], upper=case['voltage_upper_by_bus'], PCC=case['shared_beta'] * sum((s['core']['feeder_safe_kw'] for s in case['stations'])), source_ac=ac['raw'], J=Jref)
                                m = build(case['stations'], hour, app, packet, 59 / 60, network)
                                nominal, r = solve(m, out / 'mpc' / f'{minute:03d}_joint', log, 'JOINT_MIP', minute, saved[1], deadline)
                                planner.append(r)
                            else:
                                accepted = {}
                                for s in case['stations']:
                                    m = build([s], hour, app, packet, 59 / 60)
                                    x, r = solve(m, out / 'mpc' / f"{minute:03d}_{s['profile_id']}", log, 'MPC', minute, saved[1], deadline)
                                    accepted.update(x)
                                    planner.append(r)
                                nominal = accepted
                            nominal_hour = hour
                        if nominal is None or nominal_hour != hour:
                            raise RuntimeError('NO_ACCEPTED_CURRENT_HOUR_PLAN')
                        if arm == 'JOINT_H24_NETWORK_MPC':
                            mapped = {sid: common_action(nominal[sid], app[sid], packet['stations'][sid]['actual_ev_kw'], packet['stations'][sid]['actual_pv_kw'], configs[sid].battery) for sid in STATIONS}
                            proposal = {sid: {k: mapped[sid][k] for k in ('charge_kw', 'discharge_kw')} for sid in STATIONS}
                            fast = dict(common=mapped, REF=False)
                        else:
                            obs = [dict(profile_id=sid, observation=packet['stations'][sid]) for sid in STATIONS]
                            scalar, caps, details = scalar_target(obs, configs, nominal, app, case['shared_beta'])
                            proposal = {sid: {k: scalar[sid][k] for k in ('charge_kw', 'discharge_kw')} for sid in STATIONS}
                            fast = dict(scalar=details, caps=caps)
                            if arm == 'NETWORK_JOINT':
                                K, req, resp = net.jacobian(minute, ac['snapshot'], ac['raw'], 7)
                                Jref = dict(request=req, response=resp)
                                proposal, projection = net.projection(minute, ac['snapshot'], K, scalar, caps, ac['raw'], app)
                                fast['projection'] = projection
                        if time.perf_counter() > deadline:
                            raise TimeoutError('Complete controller missed fixed effect slot')
                    except Exception as e:
                        reason = 'CONTROL_FAILURE'
                        failure = dict(second=second, type=type(e).__name__, message=str(e))
                        failures.append(failure)
                        log.event('CONTROL_FAILURE_RETAINED', second, failure=failure)
                        proposal = None
                        if second % 3600 == 0 and nominal_hour != hour:
                            nominal = None
                elif reason is not None:
                    failures.append(dict(second=second, type=reason))
                log.event('CONTROL_DECISION', second, packet=packet, nominal=nominal, planner=planner, fast=fast, prefix_reference=None if prefix is None else prefix_ref, J=Jref, proposal=proposal, reason=reason, opening=opening)
                elapsed_ns = time.perf_counter_ns() - tick_ns
                admission = dispatch.finish(minute, opening, elapsed_ns, proposal, reason)
                log.event('COMMAND_ADMISSION', second, admission=admission)
                tail = time.perf_counter_ns() - tick_ns
                dispatch.busy_until_ns = opening['engine_start_model_ns'] + tail
                admission['busy_after_receipt_ns'] = dispatch.busy_until_ns
                decisions.append(admission)
                control_seconds += tail / 1000000000.0
            for sid, a in physical.items():
                b = configs[sid].battery
                o = packet['stations'][sid]
                through = (a['charge_kw'] + a['discharge_kw']) / 3600
                grid = a['grid_import_kw'] / 3600
                short = max(o['reserve_requirement_kwh'] - b.eta_discharge * b.energy_kwh * (a['soc_post'] - b.soc_min), 0.0)
                aggregate['grid_kwh'] += grid
                aggregate['throughput_kwh'] += through
                aggregate['common_cost_yuan'] += grid * o['price_yuan_per_kwh'] + 0.08 * through
                aggregate['charge_kwh'] += a['charge_kw'] / 3600
                aggregate['discharge_kwh'] += a['discharge_kw'] / 3600
                aggregate['curtail_kwh'] += a['pv_curtail_kw'] / 3600
                aggregate['reserve_shortage_exposure_kwh_h'] += short / 3600
                aggregate['physical_clip_kwh'] += a['physical_clip_kw'] / 3600
                state[sid] = a['soc_post']
                tracked[sid] = a['tracked_post_kw']
            pcc = case['shared_beta'] * sum((s['core']['feeder_safe_kw'] for s in case['stations']))
            aggregate['PCC_excess_kwh'] += max(sum(imports.values()) - pcc, 0.0) / 3600
            if v is None:
                aggregate['voltage_unknown_seconds'] += 1
            else:
                aggregate['voltage_lower_deficit_pu_h'] += sum(v['lower']) / 3600
                aggregate['voltage_upper_excess_pu_h'] += sum(v['upper']) / 3600
                aggregate['positive_voltage_bus_seconds'] += v['positive_bus_count']
                aggregate['minimum_voltage_pu'] = v['minimum'] if aggregate['minimum_voltage_pu'] is None else min(aggregate['minimum_voltage_pu'], v['minimum'])
            frame = dict(second=second, hour=hour, physical=physical, command=held, held_info=held_info, packet=packet, current_ac=ac['raw'], ac_valid=ac['valid'], ac_failure=ac['failure'], ac_cache_hit=ac['cache_hit'], ac_cache_key=ac['cache_key'], voltage=v)
            trace.write(encoded(frame) + b'\n')
            trace.flush()
            done = second + 1
            if second % 60 == 59:
                os.fsync(trace.fileno())
            evaluation_seconds += time.perf_counter() - eval_start
    except BaseException as e:
        fatal = dict(type=type(e).__name__, message=str(e), completed_seconds=done)
    finally:
        old_model.milp, opt.milp, opt.linprog, opt.minimize, pp.runpp = saved
        trace.flush()
        os.fsync(trace.fileno())
        trace.close()
        counts = dict(log.counts)
        log.close()
        terminal_delta = sum((configs[s].battery.energy_kwh * (state[s] - initial[s]) for s in STATIONS))
        result = dict(status='COMPLETED' if done == 10800 and fatal is None else 'FAILED', run_id=batch.name, arm=arm, physical_seconds=done, planned_physical_seconds=10800, fatal=fatal, control_failures=failures, command_admissions=decisions, call_intents=counts, maximum_calls=CAPS[arm], AC_cache_hits=0 if net is None else net.cache_hits, aggregate_actual=aggregate, initial_soc=initial, final_soc=state, final_tracked_kw=tracked, terminal_stored_energy_delta_kwh=terminal_delta, terminal_value_corrected_cost_yuan=aggregate['common_cost_yuan'] - 0.57 * terminal_delta, timing=dict(controller_chain_seconds=control_seconds, network_initialization_seconds=locals().get('initialization_seconds'), all_physics_evaluation_and_controller_seconds=evaluation_seconds, offline_nondecision_AC_seconds=offline_ac_seconds, arm_wall_seconds=time.perf_counter() - begin, latency_scope='Online current AC+prefix+MPC+REF+J+projection+certificates+persistence; network construction and per-second ex-post AC evaluation separately counted; digital not CIL'), complete_controller_service=done == 10800 and fatal is None and (not failures) and all((x['admitted'] for x in decisions)), thermal_qualified=None, actual3h_return_required=False, linear_steady_horizon_not_transient_AC_guarantee=True, files=[record(p) for p in sorted(out.rglob('*')) if p.is_file()])
        write(out / 'ARM_COMPLETE.json', result)
    return result
