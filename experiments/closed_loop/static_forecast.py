# EN: Pre-step causal persistence forecast builder without future lookahead.
# 中文：无未来信息泄漏的步前因果持续性预测序列构造器。

from fractions import Fraction as F
from datetime import datetime
import math
CHANNELS = ('grid_import', 'grid_export', 'pv', 'ev')

def epoch(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def forecast(packet, now_utc, widths, parameters):
    now = epoch(now_utc)
    records = {r['channel']: r for r in packet['source_records']}
    if set(records) != set(CHANNELS):
        raise ValueError('Exactly four raw channels required')
    value = {}
    for k, r in records.items():
        if not r['qualified'] or r['causal_quality_mask'] or (not epoch(r['modeled_available_at_utc']) <= now) or (not 0 <= now - r['timestamp_seconds'] <= 900):
            raise ValueError('Unqualified, future or over-age report')
        if not epoch(r['native_support_start_utc']) < epoch(r['native_support_end_utc']) <= epoch(r['modeled_available_at_utc']):
            raise ValueError('Bad support/release clock')
        if r['native_support_seconds'] > 900:
            raise ValueError('Overlong original support')
        v = r['recorded_power_estimate_W'] / 1000
        if not math.isfinite(v) or v < 0:
            raise ValueError('Invalid original power estimate')
        value[k] = v
    demand = value['grid_import'] - value['grid_export'] + value['pv']
    pv = value['pv']
    if demand - value['ev'] < 0:
        raise ValueError('Negative derived nonEV background; no clipping')
    if len(widths) != 384 or parameters['horizon_steps'] != 96 or parameters['step_hours'] != 0.25:
        raise ValueError('Frozen H96 required')
    if parameters['pressure_dimensionless'] != 1.0:
        raise ValueError('Dimensionless pressure fixed at one')
    ws = {(r['lead_zero_based'], r['side']): r for r in widths}
    if len(ws) != 384:
        raise ValueError('Duplicate/missing widths')
    d = F(demand)
    p = F(pv)
    out = {k: [] for k in ('ev_lower_kw', 'ev_point_kw', 'ev_upper_kw', 'pv_lower_kw', 'pv_point_kw', 'pv_upper_kw', 'price_yuan_per_kwh', 'pressure')}
    exact = []
    for lead in range(96):
        w = {s: ws[lead, s] for s in ('demand_lower', 'demand_upper', 'pv_lower', 'pv_upper')}
        if any((not x['finite_bound_ready'] or x['alpha_exact'] != '1/40' or x['included_samples'] < 39 for x in w.values())):
            raise ValueError('Lead lacks frozen finite .025 calibration')
        lo = max(F(0), d - F(w['demand_lower']['width_kw_exact']))
        hi = d + F(w['demand_upper']['width_kw_exact'])
        plo = max(F(0), p - F(w['pv_lower']['width_kw_exact']))
        phi = p + F(w['pv_upper']['width_kw_exact'])
        ex = dict(demand_lower=str(lo), demand_point=str(d), demand_upper=str(hi), pv_lower=str(plo), pv_point=str(p), pv_upper=str(phi))
        exact.append(ex)
        for k, v in [('ev_lower_kw', lo), ('ev_point_kw', d), ('ev_upper_kw', hi), ('pv_lower_kw', plo), ('pv_point_kw', p), ('pv_upper_kw', phi)]:
            out[k].append(float(v))
        out['price_yuan_per_kwh'].append(parameters['illustrative_price_yuan_per_kwh'])
        out['pressure'].append(1.0)
    return dict(issued_utc=now_utc, horizon=out, exact_rational_bounds=exact, EV_submeter_kw=value['ev'], demand_slot_semantics='TOTAL HOUSE = import - export + PV, counted once; not EV-only demand', reserve_requirement_kwh=parameters['reserve_hours'] * max(out['ev_upper_kw'][0] - out['pv_lower_kw'][0] - parameters['virtual_import_limit_kw'], 0.0), widths_frozen_before_episode=True, ACI=False, future_profile_or_label_consumed=False)
