from copy import deepcopy
from dataclasses import asdict
import math
from station_core.plant import realize_no_export
from station_core.filter import advance_soc_checked
DT = 1 / 3600

def step(command, soc, tracked, ev, pv, battery):
    c = float(command['charge_kw'])
    d = float(command['discharge_kw'])
    if min(c, d, ev, pv) < -1e-09 or min(c, d) > 1e-09 or (not all((math.isfinite(x) for x in (c, d, soc, tracked, ev, pv)))):
        raise ValueError('Physical command domain')
    desired = max(d, 0.0) - max(c, 0.0)
    ramp = max(battery.charge_limit_kw, battery.discharge_limit_kw) / 30.0
    alpha = 1 - math.exp(-1 / 5.0)
    ramped = min(max(desired, tracked - ramp), tracked + ramp)
    lagged = tracked + alpha * (ramped - tracked)
    maxc = max((battery.soc_max - soc) * battery.energy_kwh / (battery.eta_charge * DT), 0.0)
    maxd = max(battery.eta_discharge * battery.energy_kwh * (soc - battery.soc_min) / DT, 0.0)
    cc = min(-lagged, battery.charge_limit_kw, maxc) if lagged < 0 else 0.0
    dd = min(lagged, battery.discharge_limit_kw, maxd) if lagged >= 0 else 0.0
    a = realize_no_export(cc, dd, ev, pv, tolerance=1e-09)
    post = advance_soc_checked(soc, a.charge_kw, a.discharge_kw, battery, DT, tolerance=1e-09)
    return dict(**asdict(a), soc_pre=soc, soc_post=post, tracked_pre_kw=tracked, tracked_post_kw=a.discharge_kw - a.charge_kw, desired_kw=desired, ramped_kw=ramped, lagged_kw=lagged, physical_clip_kw=abs(a.discharge_kw - a.charge_kw - lagged), ramp_override_kw=max(abs(a.discharge_kw - a.charge_kw - tracked) - ramp, 0.0))

def observation(case, second):
    hour = second // 3600
    return dict(sample_s=hour * 3600, available_s=hour * 3600, read_s=second, source_hour=hour, source_availability='ASSUMED_OPENING_RELEASED_CROSS_SOURCE_HOURLY_MEAN_ZOH_MODEL_POINT', measured_second_data=False, stations={s['profile_id']: {k: s['hours'][hour][k] for k in ('actual_ev_kw', 'actual_pv_kw', 'reserve_requirement_kwh', 'price_yuan_per_kwh')} for s in case['stations']})

def prefix60(second, state, tracked, held, valid_until, packet, configs):
    if packet['available_s'] > second or packet['sample_s'] > second:
        raise ValueError('Future packet')
    x = dict(state)
    power = dict(tracked)
    rows = []
    for offset in range(60):
        at = second + offset
        row = {}
        for sid, cfg in configs.items():
            obs = packet['stations'][sid]
            cmd = held[sid] if at < valid_until else dict(charge_kw=0.0, discharge_kw=0.0)
            a = step(cmd, x[sid], power[sid], obs['actual_ev_kw'], obs['actual_pv_kw'], cfg.battery)
            row[sid] = a
            x[sid] = a['soc_post']
            power[sid] = a['tracked_post_kw']
        rows.append(dict(second=at, stations=row))
    fixed = {sid: dict(grid_kwh=sum((r['stations'][sid]['grid_import_kw'] for r in rows)) / 3600, throughput_kwh=sum((r['stations'][sid]['charge_kw'] + r['stations'][sid]['discharge_kw'] for r in rows)) / 3600) for sid in configs}
    for sid, v in fixed.items():
        v['common_cost_yuan'] = v['grid_kwh'] * packet['stations'][sid]['price_yuan_per_kwh'] + 0.08 * v['throughput_kwh']
    return dict(predicted_soc=x, predicted_tracked_kw=power, rows=rows, source_packet=deepcopy(packet), prediction='CURRENT_POINT_PERSISTENCE_NOT_FUTURE_OBSERVATION', held_valid_until_s=valid_until, fixed_prefix_resources=fixed, fixed_prefix_common_cost_yuan=sum((v['common_cost_yuan'] for v in fixed.values())), prefix_cost_scope='Known held action cost independent of new decision; excluded from suffix optimizer objective; actual seconds counted once in episode cost')
