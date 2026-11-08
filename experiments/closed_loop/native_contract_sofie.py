# EN: Causal step-opening packet schema and physical admissibility contract.
# 中文：步初因果数据包结构校验与物理可行域契约定义模块。

import math
from opening_state_sofie import validate_opening

def forecast(packet, now_s, policy):
    ev, pv = validate_opening(packet, now_s)
    if policy['horizon_steps'] != 96 or policy['step_hours'] != 0.25:
        raise ValueError('24-hour horizon required')
    if policy['pressure_dimensionless'] != 1.0:
        raise ValueError('Frozen dimensionless phi must equal one')
    a, b = (policy['ev_additive_uncertainty_kw'], policy['pv_additive_uncertainty_kw'])
    if not all((math.isfinite(x) and x >= 0 for x in (a, b))):
        raise ValueError('Invalid fixed band')
    pv_lower = max(pv - b, 0.0)
    values = dict(ev_lower_kw=max(ev - a, 0.0), ev_point_kw=ev, ev_upper_kw=ev + a, pv_lower_kw=pv_lower, pv_point_kw=pv, pv_upper_kw=pv + b, price_yuan_per_kwh=policy['illustrative_price_yuan_per_kwh'], pressure=1.0)
    return dict(horizon={k: [v] * 96 for k, v in values.items()}, observation_id=packet['observation_id'], forecast_issued_s=now_s, horizon_hours=24.0, calibration_samples=0, ACI=False, pressure_scope='fixed dimensionless phi=1, uncalibrated', reserve_requirement_kwh=policy['reserve_hours'] * max(ev + a - pv_lower - policy['virtual_import_limit_kw'], 0.0))

def current_soc(measurement, now_s, battery):
    if measurement is None or measurement.get('kind') != 'PLANT_SOC_MEASUREMENT':
        raise ValueError('Independent plant SOC measurement required')
    sample, available, soc = (measurement['sample_s'], measurement['available_s'], measurement['soc'])
    if not all((math.isfinite(x) for x in (sample, available, soc))):
        raise ValueError('Nonfinite SOC measurement')
    if not sample <= available <= now_s or now_s - sample > 1.0:
        raise ValueError('Future or stale independent SOC')
    if not battery.soc_min <= soc <= battery.soc_max:
        raise ValueError('SOC outside original domain')
    return soc
