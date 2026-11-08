# EN: Causal step-opening packet schema and physical admissibility contract.
# 中文：步初因果数据包结构校验与物理可行域契约定义模块。

from datetime import datetime
import math
PORTS = ('32', '1571', '1353', '1345', '1108', '1696', '3620', '1768')
EXPECTED = {('PV', '')} | {('EV', p) for p in PORTS}

def validate_opening(message, now_s):
    if message.get('kind') != 'NATIVE_OPENING_OBSERVATION':
        raise ValueError('Only native opening messages are controller inputs')
    if not math.isfinite(now_s) or now_s < message['decision_s']:
        raise ValueError('Message not yet released at this decision')
    records = message['records']
    if len(records) != 9 or {(r['channel'], r['connector_pk']) for r in records} != EXPECTED:
        raise ValueError('Exactly PV plus eight distinct native EV ports required')
    if not message['valid_complete_pair']:
        raise ValueError('Incomplete input; common explicit fallback required')
    for r in records:
        if not r['valid'] or not r['source_records']:
            raise ValueError('Invalid or unbound native sample')
        sample, available = (r['reported_s'], r['available_s'])
        if not all((math.isfinite(x) for x in (sample, available, r['power_kw']))):
            raise ValueError('Nonfinite native input')
        if not sample <= available <= message['decision_s'] <= now_s:
            raise ValueError('Future or invalid availability')
        age = now_s - sample
        cap = 10.0 if r['channel'] == 'PV' else 900.0
        if age > cap or age < 0 or r['power_kw'] < 0:
            raise ValueError('Stale or out-of-domain native input')
        if abs(r['age_seconds'] - (message['decision_s'] - sample)) > 1e-09:
            raise ValueError('Per-channel age was rewritten')
    pv = next((r['power_kw'] for r in records if r['channel'] == 'PV'))
    ev = sum((r['power_kw'] for r in records if r['channel'] == 'EV'))
    if abs(pv - message['pv_point_kw']) > 1e-09 or abs(ev - message['ev_point_kw']) > 1e-09:
        raise ValueError('Aggregate mismatch')
    return (ev, pv)

def persistence(message, now_s, policy):
    ev, pv = validate_opening(message, now_s)
    if policy['horizon_steps'] != 96 or policy['step_hours'] != 0.25:
        raise ValueError('Preserve the 24-hour prediction duration')
    h = policy['horizon_steps']
    ev_margin = policy['ev_additive_uncertainty_kw']
    pv_margin = policy['pv_additive_uncertainty_kw']
    if any((not math.isfinite(x) or x < 0 for x in (ev_margin, pv_margin))):
        raise ValueError('Invalid fixed uncertainty contract')
    lower_pv = max(0.0, pv - pv_margin)
    values = dict(ev_lower_kw=max(0.0, ev - ev_margin), ev_point_kw=ev, ev_upper_kw=ev + ev_margin, pv_lower_kw=lower_pv, pv_point_kw=pv, pv_upper_kw=pv + pv_margin, price_yuan_per_kwh=policy['illustrative_price_yuan_per_kwh'], pressure=ev_margin + pv - lower_pv)
    reserve = policy['reserve_hours'] * max(values['ev_upper_kw'] - lower_pv - policy['virtual_import_limit_kw'], 0.0)
    return dict(horizon={k: [v] * h for k, v in values.items()}, forecast_issued_s=now_s, observation_id=message['observation_id'], horizon_hours=24, method='CAUSAL_PERSISTENCE_FIXED_UNCALIBRATED_BANDS', calibration_samples_used=0, ACI_enabled=False, reserve_requirement_kwh=reserve, pressure_scope='fixed band adverse deviation in kW, not a calibrated statistic', uncertainty_coverage_guarantee=False)

def released_settlements(rows, now_s):
    if not math.isfinite(now_s):
        raise ValueError('Finite time required')
    return [r for r in rows if r['logical_available_s'] <= now_s]
