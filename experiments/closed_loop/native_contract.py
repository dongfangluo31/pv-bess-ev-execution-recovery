# EN: Causal step-opening packet schema and physical admissibility contract.
# 中文：步初因果数据包结构校验与物理可行域契约定义模块。

import math
from opening_state_opsd import validate_opening
from static_forecast import forecast as frozen_forecast
from protocol import digest

def forecast(packet, now_s, policy):
    validate_opening(packet, now_s)
    if digest(policy['static_widths']) != policy['static_widths_sha256']:
        raise ValueError('Embedded static width identity changed')
    out = frozen_forecast(packet['original_packet'], packet['decision_utc'], policy['static_widths'], policy['static_parameters'])
    out.update(observation_id=packet['observation_id'], forecast_issued_s=now_s, horizon_hours=24.0, calibration_scope='Frozen per-lead four-sided alpha.025; no within-episode update or guaranteed temporal coverage', input_binding='Legacy ev_* = total house demand, EV submeter separately reported', static_widths_sha256=policy['static_widths_sha256'])
    return out

def current_soc(measurement, now_s, battery):
    if measurement is None or measurement.get('kind') != 'PLANT_SOC_MEASUREMENT':
        raise ValueError('Independent own plant SOC required')
    sample, available, soc = (measurement['sample_s'], measurement['available_s'], measurement['soc'])
    if not all((math.isfinite(x) for x in (sample, available, soc))):
        raise ValueError('Nonfinite SOC')
    if not sample <= available <= now_s or now_s - sample > 1.0:
        raise ValueError('Future/stale own plant SOC')
    if not battery.soc_min <= soc <= battery.soc_max:
        raise ValueError('SOC outside original domain')
    return soc
