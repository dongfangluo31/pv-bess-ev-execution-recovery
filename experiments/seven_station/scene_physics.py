# EN: Balanced AC distribution-network step physics coupling station BESS/PV/EV injections with `pandapower`.
# 中文：将各场站光储充有功/无功注入耦合至 `pandapower` 交流配电网潮流求解的步级物理引擎。

from dataclasses import asdict
import math
from scene_io import STATIONS
DT = 1 / 60

def common_action(command, soc, ev, pv, battery):
    from station_core.plant import realize_no_export
    from station_core.filter import advance_soc_checked
    c = float(command['charge_kw'])
    d = float(command['discharge_kw'])
    if not all((math.isfinite(x) for x in (c, d, soc, ev, pv))) or min(c, d, ev, pv) < -1e-09 or min(c, d) > 1e-09:
        raise ValueError('Invalid requested ideal actuator command')
    cc = min(max(c, 0.0), battery.charge_limit_kw, max((battery.soc_max - soc) * battery.energy_kwh / (battery.eta_charge * DT), 0.0))
    dd = min(max(d, 0.0), battery.discharge_limit_kw, max(battery.eta_discharge * battery.energy_kwh * (soc - battery.soc_min) / DT, 0.0))
    a = realize_no_export(cc, dd, ev, pv, tolerance=1e-09)
    post = advance_soc_checked(soc, a.charge_kw, a.discharge_kw, battery, DT, tolerance=1e-09)
    return dict(**asdict(a), soc_pre=soc, soc_post=post, requested_charge_kw=c, requested_discharge_kw=d, common_charge_clip_kw=c - a.charge_kw, common_discharge_total_clip_kw=d - a.discharge_kw)

def scalar_target(stations, cores, nominal, application_soc, beta):
    from station_core.config import FilterKind
    from station_core.filter import apply_execution_filter
    from station_core.validation import CoupledProjectionInput, coupled_greedy_projection
    local = []
    ch = []
    dh = []
    for s in stations:
        sid = s['profile_id']
        cfg = cores[sid]
        b = cfg.battery
        h = s['observation']
        soc = application_soc[sid]
        common = common_action(nominal[sid], soc, h['actual_ev_kw'], h['actual_pv_kw'], b)
        f = apply_execution_filter(FilterKind.F1_RESERVE, common['charge_kw'], common['discharge_kw'], h['actual_ev_kw'], h['actual_pv_kw'], soc, b, cfg.feeder_safe_kw, DT, h['reserve_requirement_kwh'])
        usable = max(b.eta_discharge * b.energy_kwh * (soc - b.soc_min) - h['reserve_requirement_kwh'], 0.0)
        ccap = min(f.charge_kw, f.final_grid_kw)
        dcap = max(0.0, min(b.discharge_limit_kw - f.discharge_kw, max(usable - f.discharge_kw * DT, 0.0) / DT, max(f.final_grid_kw - ccap, 0.0)))
        ch.append(ccap)
        dh.append(dcap)
        local.append(dict(profile_id=sid, common_nominal=common, local_ref=asdict(f)))
    shared_limit = beta * sum((c.feeder_safe_kw for c in cores.values()))
    excess = max(sum((r['local_ref']['final_grid_kw'] for r in local)) - shared_limit, 0.0)
    coupled = coupled_greedy_projection(CoupledProjectionInput(excess, tuple(ch), tuple(dh)))
    result = {}
    caps = {}
    for i, (s, row) in enumerate(zip(stations, local)):
        sid = s['profile_id']
        b = cores[sid].battery
        h = s['observation']
        f = row['local_ref']
        command = dict(charge_kw=f['charge_kw'] - coupled.reduce_charge_kw[i], discharge_kw=f['discharge_kw'] + coupled.extra_discharge_kw[i])
        a = common_action(command, application_soc[sid], h['actual_ev_kw'], h['actual_pv_kw'], b)
        if a['common_charge_clip_kw'] > 1e-07 or a['common_discharge_total_clip_kw'] > 1e-07:
            raise ValueError('Shared correction requires unaccounted clip')
        result[sid] = dict(charge_kw=a['charge_kw'], discharge_kw=a['discharge_kw'], grid_import_kw=a['grid_import_kw'])
        reserve = h['reserve_requirement_kwh']
        usable = max(b.eta_discharge * b.energy_kwh * (application_soc[sid] - b.soc_min) - reserve, 0.0)
        cc = min(a['charge_kw'], a['grid_import_kw'])
        dd = max(0.0, min(b.discharge_limit_kw - a['discharge_kw'], max(usable - a['discharge_kw'] * DT, 0.0) / DT, max(a['grid_import_kw'] - cc, 0.0)))
        caps[sid] = dict(charge_cap_kw=cc, discharge_cap_kw=dd, total_cap_kw=min(a['grid_import_kw'], cc + dd))
    return (result, caps, dict(local=local, coupled=asdict(coupled), shared_limit_kw=shared_limit))

def apply_correction(base, caps, delta):
    result = {}
    for sid, x in zip(STATIONS, delta):
        x = float(x)
        cc = caps[sid]['charge_cap_kw']
        if not math.isfinite(x) or x < -1e-09 or x > caps[sid]['total_cap_kw'] + 1e-09:
            raise ValueError('Network delta outside original physical box')
        x = max(x, 0.0)
        u = min(x, cc)
        v = max(x - u, 0.0)
        result[sid] = dict(charge_kw=base[sid]['charge_kw'] - u, discharge_kw=base[sid]['discharge_kw'] + v)
    return result
