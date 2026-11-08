# EN: Isolated physical plant companion process integrating sub-step BESS/PCC dynamics and enforcing zero-export bounds.
# 中文：独立物理受控对象子进程：积分步内储能与并网点功率平衡并施加防逆流与 SOC 边界约束。

from bisect import bisect_right
from pathlib import Path
import json, math, os, sys, time
from protocol import load, emit, Log, digest, record
from native_process import own_identity
from station_core.config import BatteryConfig
from station_core.plant import realize_no_export
from station_core.filter import advance_soc_checked
from case_io import load_profile, step_source_quality

def main():
    cfg = load(sys.argv[1])
    out = Path(cfg['output_dir'])
    log = Log(out / 'events.jsonl', 'plant')
    battery = BatteryConfig(**cfg['battery'])
    nonce = cfg['nonce']
    profile = cfg['plant_profile']
    if record(profile['path']) != profile:
        raise ValueError('Plant profile identity mismatch')
    segments = load_profile(profile, cfg['dataset'], cfg['duration_s'])
    starts = [s['start_s'] for s in segments]
    if starts[0] != 0 or segments[-1]['end_s'] != cfg['duration_s']:
        raise ValueError('Frozen native window changed')
    soc = battery.soc_initial
    tracked = 0.0
    queue = []
    active = None
    highest_seq = 0
    step = 1
    next_s = 0
    emit(dict(type='HELLO', role='plant', nonce=nonce, identity=own_identity()))
    log.event('START', 0, initial_soc=soc, step_s=step, profile=profile, initial_tracked_kw=tracked, source_scope='fixed OPSD counter-support-uniform offline estimate; virtual BESS/PCC, not measured seconds')

    def actual(t):
        i = bisect_right(starts, t) - 1
        s = segments[i]
        if not s['start_s'] <= t < s['end_s']:
            raise ValueError('No true-model segment at time')
        return (float(s['ev_kw']), float(s['pv_kw']), s['segment_id'], step_source_quality(s, t, cfg['dataset']), s.get('EV_submeter_kw'))
    for line in sys.stdin:
        m = json.loads(line)
        rid = m['request_id']
        kind = m['type']
        t = m.get('sim_s', 0)
        if m['nonce'] != nonce:
            raise ValueError('Wrong nonce')
        if kind == 'STOP':
            emit(dict(request_id=rid, type='STOPPED', nonce=nonce))
            break
        if t != next_s:
            raise ValueError('Plant clock must be current sequential state')
        if kind == 'OBSERVE_SOC':
            measurement = dict(kind='PLANT_SOC_MEASUREMENT', id=f'soc:{t}', sample_s=t, available_s=t, soc=soc, plant_pid=os.getpid(), source='independent_simulated_tracked_SOC', age_seconds=0)
            log.event('SOC_OBSERVATION', t, measurement=measurement, true_SOC=soc)
            emit(dict(request_id=rid, type='SOC_OBSERVATION', nonce=nonce, measurement=measurement))
            continue
        if kind == 'COMMAND':
            c = m['command']
            flags = []
            if c['issue_s'] > t:
                flags.append('FUTURE_ISSUE')
            if c['effective_s'] > t:
                flags.append('EARLY_DELIVERY')
            if c['valid_until_s'] < t:
                flags.append('EXPIRED')
            if c['seq'] <= highest_seq:
                flags.append('OUT_OF_ORDER')
            if not math.isfinite(c['signed_kw']):
                flags.append('NONFINITE')
            if flags:
                log.event('COMMAND_REJECT', t, command=c, reasons=flags)
            else:
                queue.append(c)
                queue.sort(key=lambda x: (x['effective_s'], x['seq']))
                log.event('COMMAND_ENQUEUED', t, command=c)
            emit(dict(request_id=rid, type='COMMAND_ACK', nonce=nonce, rejected=bool(flags), reasons=flags))
            continue
        if kind != 'ADVANCE' or t >= cfg['duration_s']:
            raise ValueError('Unknown or past-end advance')
        for c in list(queue):
            if c['effective_s'] <= t:
                queue.remove(c)
                if c['valid_until_s'] < t or c['seq'] <= highest_seq:
                    log.event('QUEUED_COMMAND_REJECT', t, command=c)
                else:
                    active = c
                    highest_seq = c['seq']
                    log.event('COMMAND_APPLIED', t, command=c)
        fallback = active is None or active['valid_until_s'] < t
        desired = 0.0 if fallback else active['signed_kw']
        previous = tracked
        ramp = cfg['ramp_kw_per_second'] * step
        ramped = min(max(desired, previous - ramp), previous + ramp)
        alpha = 1.0 if cfg['lag_seconds'] == 0 else 1 - math.exp(-step / cfg['lag_seconds'])
        lagged = previous + alpha * (ramped - previous)
        dt = step / 3600.0
        maxc = max(0.0, (battery.soc_max - soc) * battery.energy_kwh / (battery.eta_charge * dt))
        maxd = max(0.0, battery.eta_discharge * battery.energy_kwh * (soc - battery.soc_min) / dt)
        charge = min(-lagged, battery.charge_limit_kw, maxc) if lagged < 0 else 0.0
        discharge = min(lagged, battery.discharge_limit_kw, maxd) if lagged >= 0 else 0.0
        ev, pv, segment_id, source_quality, ev_submeter = actual(t)
        a = realize_no_export(charge, discharge, ev, pv)
        tracked = a.discharge_kw - a.charge_kw
        before = soc
        soc = advance_soc_checked(soc, a.charge_kw, a.discharge_kw, battery, dt)
        log.event('PLANT_STEP', t, end_s=t + 1, source_segment_id=segment_id, soc_pre=before, soc_post=soc, previous_tracked_kw=previous, desired_signed_kw=desired, ramped_kw=ramped, lagged_kw=lagged, charge_kw=a.charge_kw, discharge_kw=a.discharge_kw, tracked_kw=tracked, pre_noexport_charge_kw=charge, pre_noexport_discharge_kw=discharge, noexport_discharge_clip_kw=a.discharge_clip_kw, grid_kw=a.grid_import_kw, pv_curtail_kw=a.pv_curtail_kw, ev_kw=ev, pv_kw=pv, demand_kw=ev, EV_submeter_kw=ev_submeter, demand_includes_EV_once=True, fallback=fallback, applied_command_seq=None if active is None else active['seq'], applied_issue_s=None if active is None else active['issue_s'], applied_effective_s=None if active is None else active['effective_s'], command_expired=active is not None and active['valid_until_s'] < t, physical_clip_kw=abs(tracked - lagged), ramp_override_kw=max(abs(tracked - previous) - ramp, 0.0), source_quality=source_quality)
        next_s = t + 1
        emit(dict(request_id=rid, type='ADVANCED', nonce=nonce, end_s=next_s))
    log.event('STOP', next_s, final_soc=soc, actual_plant_steps=next_s)
    log.close()
if __name__ == '__main__':
    main()
