# EN: Closed-loop episode relay stepping causal packets between plant and controller subprocesses.
# 中文：在物理对象子进程与控制器子进程之间按因果时序中继数据包与控制指令的闭环调度器。

from pathlib import Path
from dataclasses import asdict
import argparse, ast, gzip, heapq, json, os, sys, time, uuid, copy, shutil
from protocol import load, write, record, digest, Log
from case_io import load_openings, load_profile
from slot_contract import fixed_slot_delivery
HERE = Path(__file__).resolve().parent
CASES = {'opsd24h': ('opsd', 86400), 'sofie875h': ('sofie', 31500), 'opsd67h': ('opsd', 241200)}
ARMS = ('F0_FAST', 'F1_FAST')

def count_attempts(path, arm=None):
    if not path.exists():
        return 0
    with path.open(encoding='utf-8') as f:
        return sum((r.get('kind') == 'MPC' and (arm is None or r['arm'] == arm) for r in map(json.loads, f)))

def episode(arm, batch, execution, ledger, Peer):
    out = execution / arm
    out.mkdir(exist_ok=False)
    log = Log(out / 'relay_events.jsonl', 'relay')
    plan = load(batch / 'batch_plan.json')
    duration = plan['duration_s']
    cap = plan['per_arm_MPC_cap']
    cfg = load(batch / 'CONTROLLER_BASE.json')
    pcfg = load(batch / 'PLANT_BASE.json')
    cfg.update(arm=arm, mode='mpc', decision_period_s=60, command_lifetime_s=60, maximum_MPC_calls=cap, output_dir=str(out / 'controller'), nonce=uuid.uuid4().hex, attempt_ledger=str(ledger))
    pcfg.update(output_dir=str(out / 'plant'), nonce=uuid.uuid4().hex)
    source = load(batch / 'NATIVE_RELAY_SOURCE.json')['openings']
    packets = {r['decision_s']: r for r in load_openings(source, duration)}
    controller = plant = None
    dead = False
    commands = []
    control_seq = 0
    steps = 0
    errors = []
    failure = None
    final_soc = None
    begun = time.perf_counter()
    write(out / 'RUN_START.json', dict(run_id=batch.name, arm=arm, expected_plant_steps=duration, maximum_MPC=cap, controller_config=cfg, plant_config=pcfg, native_schedule=source))
    log.event('EPISODE_BEGIN', 0, arm=arm, expected_plant_steps=duration, expected_outer_opportunities=cap, arm_wall_seconds=plan['arm_wall_seconds'], clock_origin_logger_time=cfg['clock_origin_logger_time'], protocol='accelerated event-time; actual RPC duration contributes to ready_s; common fixed t+60s dispatch slot; late responses dropped; validity ends t+120s')
    try:
        plant = Peer('plant', pcfg, log)
        controller = Peer('controller', cfg, log)
        if plant.identity['pid'] == controller.identity['pid']:
            raise RuntimeError('Same process')
        for t in range(duration):
            if time.perf_counter() - begun > plan['arm_wall_seconds']:
                raise TimeoutError('Frozen arm watchdog')
            if t % 60 == 0:
                control_seq += 1
                answer, _ = plant.rpc(dict(type='OBSERVE_SOC', sim_s=t))
                soc = answer['measurement']
                packet = packets[t]
                log.event('NATIVE_INPUT_RELEASED', t, opening=packet, soc_measurement=soc, current_settlement_read=False, controller_future_profile_read=False)
                if dead:
                    log.event('CONTROLLER_OPPORTUNITY_NOT_RUN', t, control_seq=control_seq, outer_opportunity=t % 900 == 0, reason='Prior failure; no restart')
                else:
                    try:
                        answer, wall = controller.rpc(dict(type='DECIDE', sim_s=t, control_seq=control_seq, native_opening=packet, soc_measurement=soc), limit=90.0)
                        c, late = fixed_slot_delivery(answer['command'], t, wall)
                        c['native_age_at_effective'] = [dict(channel=r['channel'], connector_pk=r['connector_pk'], age_s=c['effective_s'] - r['reported_s']) for r in packet['records']]
                        if late:
                            log.event('COMMAND_MISSED_FIXED_SLOT', t, command=c, RPC_wall_seconds=wall, reason='Late actual response dropped')
                        else:
                            heapq.heappush(commands, (c['effective_s'], c['seq'], c))
                            log.event('COMMAND_SCHEDULED', t, command=c, RPC_wall_seconds=wall)
                    except Exception as exc:
                        errors.append(dict(sim_s=t, event='CONTROLLER_FAILURE', error=repr(exc)))
                        log.event('CONTROLLER_FAILURE_FALLBACK', t, error=repr(exc))
                        controller.close(forced=True)
                        dead = True
            while commands and commands[0][0] <= t:
                _, _, c = heapq.heappop(commands)
                answer, _ = plant.rpc(dict(type='COMMAND', sim_s=t, command=c))
                log.event('COMMAND_DELIVERED', t, command_seq=c['seq'], rejected=answer['rejected'])
            plant.rpc(dict(type='ADVANCE', sim_s=t))
            steps += 1
        final, _ = plant.rpc(dict(type='OBSERVE_SOC', sim_s=duration))
        final_soc = final['measurement']
        log.event('TERMINAL_STATE', duration, measurement=final_soc)
    except Exception as exc:
        failure = repr(exc)
        log.event('EPISODE_FAILURE', steps, error=failure)
    finally:
        for peer in (controller, plant):
            if peer is not None:
                try:
                    peer.close(forced=failure is not None)
                except Exception as exc:
                    errors.append(dict(event='SHUTDOWN_ERROR', error=repr(exc)))
                    log.event('SHUTDOWN_ERROR', steps, error=repr(exc))
    log.event('EPISODE_END', steps, steps=steps, parent_failure=failure, errors=errors, pending_commands_after_window=len(commands), wall_seconds=time.perf_counter() - begun)
    log.close()
    result = dict(run_id=batch.name, arm=arm, status='COMPLETED' if steps == duration and failure is None else 'FAILED', actual_MPC_calls=count_attempts(ledger, arm), LP_calls=0, AC_calls=0, plant_steps=steps, expected_plant_steps=duration, final_SOC_measurement=final_soc, errors=errors, parent_failure=failure, wall_seconds=time.perf_counter() - begun, files=[record(p) for p in sorted(out.rglob('*')) if p.is_file()])
    write(out / 'RUN_COMPLETE.json', result)
    if result['actual_MPC_calls'] > cap:
        raise RuntimeError('Per-arm actual call cap exceeded')
    return result
