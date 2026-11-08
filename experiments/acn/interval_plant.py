from fractions import Fraction as F
import math
from control import BATTERY, PTOL, CAP
from pv_bess_ev.common.plant import realize_no_export
from pv_bess_ev.common.filter import advance_soc_checked

def integrate_interval(soc, c, d, plant):
    if not all((math.isfinite(x) for x in (soc, c, d))) or min(c, d) < 0 or c > 250 + PTOL or (d > 250 + PTOL) or (c > PTOL and d > PTOL):
        raise ValueError('Invalid executed command')
    cursor = F(plant['start_s'])
    before = soc
    trace = []
    totals = dict(demand_kwh=0.0, grid_kwh=0.0, charge_kwh=0.0, discharge_kwh=0.0, throughput_kwh=0.0, PCC_excess_kwh=0.0, PCC_excess_seconds=0.0, noexport_clip_kwh=0.0, SOC_clip_kwh=0.0)
    maxgrid = 0.0
    for segment in plant['segments']:
        a = F(segment['start_s_exact'])
        b = F(segment['end_s_exact'])
        L = float(F(segment['demand_kw_exact']))
        if a != cursor or b <= a:
            raise ValueError('Noncontiguous exact service segments')
        duration = float((b - a) / 3600)
        e0 = soc * 600.0
        actual_c = min(c, max((0.9 - soc) * 600.0 / (0.95 * duration), 0.0))
        actual_d = min(d, max(0.95 * 600.0 * (soc - 0.1) / duration, 0.0))
        act = realize_no_export(actual_c, actual_d, L, 0.0)
        next_soc = advance_soc_checked(soc, act.charge_kw, act.discharge_kw, BATTERY, duration)
        delta = dict(demand_kwh=L * duration, grid_kwh=act.grid_import_kw * duration, charge_kwh=act.charge_kw * duration, discharge_kwh=act.discharge_kw * duration, throughput_kwh=(act.charge_kw + act.discharge_kw) * duration, PCC_excess_kwh=max(act.grid_import_kw - CAP, 0.0) * duration, PCC_excess_seconds=float(b - a) if act.grid_import_kw > CAP + PTOL else 0.0, noexport_clip_kwh=act.discharge_clip_kw * duration, SOC_clip_kwh=(c - actual_c + d - actual_d) * duration)
        for k, v in delta.items():
            totals[k] += v
        maxgrid = max(maxgrid, act.grid_import_kw)
        trace.append(dict(start_s_exact=str(a), end_s_exact=str(b), demand_kw=L, charge_kw=act.charge_kw, discharge_kw=act.discharge_kw, grid_kw=act.grid_import_kw, energy_pre_kwh=e0, energy_post_kwh=next_soc * 600.0, SOC_pre=soc, SOC_post=next_soc, pre_noexport_charge_kw=actual_c, pre_noexport_discharge_kw=actual_d, noexport_discharge_clip_kw=act.discharge_clip_kw, SOC_charge_clip_kw=c - actual_c, SOC_discharge_clip_kw=d - actual_d, EV_shed_kwh=0.0))
        soc = next_soc
        cursor = b
    if cursor != F(plant['end_s']) or not F(0) < F(plant['end_s']) - F(plant['start_s']) <= 60:
        raise ValueError('Incomplete minute support')
    if abs(totals['demand_kwh'] - float(F(plant['energy_kwh_exact']))) > 1e-09:
        raise ValueError('Source energy mismatch')
    totals['common_cost_yuan'] = 0.6 * totals['grid_kwh'] + 0.08 * totals['throughput_kwh']
    return dict(SOC_pre=before, SOC_post=soc, energy_pre_kwh=before * 600.0, energy_post_kwh=soc * 600.0, segments=trace, totals=totals, maximum_grid_kw=maxgrid, EV_shed_kwh=0.0)
