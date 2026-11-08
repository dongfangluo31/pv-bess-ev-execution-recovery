# EN: Closed-loop BESS dispatch policy and real-time constraint enforcement.
# 中文：闭环储能调度策略与实时物理约束执行模块。

import math, time
from dataclasses import asdict
from validated_bridge import solve_bridge, make_command_schedule, json_safe
from station_core.config import FilterKind
from station_core.filter import apply_execution_filter
from protocol import write, record, digest

def predict_prefix(measurement, queue_state, now_s, battery, L, pv, ramp, lag_seconds):
    if measurement['sample_s'] != now_s or measurement['available_s'] > now_s:
        raise ValueError('Illegal recovery telemetry clock')
    if queue_state['as_of_s'] != now_s:
        raise ValueError('Queue is not current')
    p = float(measurement['tracked_power_kw'])
    soc = float(measurement['soc'])
    if not math.isfinite(p):
        raise ValueError('Nonfinite current tracked power')
    active = queue_state.get('last_delivered_command')
    pending = sorted(queue_state['pending_commands'], key=lambda c: (c['effective_s'], c['seq']))
    if any((c['issue_s'] > now_s for c in pending)):
        raise ValueError('Future command in known queue')
    alpha = 1.0 if lag_seconds == 0 else 1 - math.exp(-1 / lag_seconds)
    rows = []
    for t in range(now_s, now_s + 60):
        for c in pending:
            if c['effective_s'] <= t and (active is None or c['seq'] > active['seq']) and (c['valid_until_s'] >= t):
                active = c
        u = 0.0 if active is None or active['valid_until_s'] < t else active['signed_kw']
        z = min(max(u, p - ramp), p + ramp)
        v = p + alpha * (z - p)
        h = 1 / 3600.0
        c = min(-v, battery.charge_limit_kw, max(0.0, (battery.soc_max - soc) * battery.energy_kwh / (battery.eta_charge * h))) if v < 0 else 0.0
        d = min(v, battery.discharge_limit_kw, max(0.0, battery.eta_discharge * battery.energy_kwh * (soc - battery.soc_min) / h)) if v >= 0 else 0.0
        d = min(d, max(L - pv + c, 0.0))
        before = p
        p = d - c
        soc += battery.eta_charge * c * h / battery.energy_kwh - d * h / (battery.eta_discharge * battery.energy_kwh)
        rows.append(dict(sim_s=t, previous_power_kw=before, target_kw=u, predicted_power_kw=p, predicted_energy_kwh=battery.energy_kwh * soc, assumed_current_load_kw=L, assumed_current_pv_kw=pv))
    return dict(predicted_power_kw=p, predicted_energy_kwh=battery.energy_kwh * soc, rows=rows, conditional_current_point_persistence=True, actual_future_consumed=False)

def safe_target(target, L, pv, soc, battery, G, reserve):
    charge = max(-target, 0.0)
    discharge = max(target, 0.0)
    common = apply_execution_filter(FilterKind.NONE, charge, discharge, L, pv, soc, battery, G, 1 / 60.0, reserve)
    filtered = apply_execution_filter(FilterKind.F1_RESERVE, common.charge_kw, common.discharge_kw, L, pv, soc, battery, G, 1 / 60.0, reserve)
    return (filtered.charge_kw, filtered.discharge_kw, dict(common=asdict(common), filter=asdict(filtered), requested_target_kw=target, changed=abs(filtered.discharge_kw - filtered.charge_kw - target) > 1e-09))
