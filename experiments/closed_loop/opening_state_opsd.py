# EN: Causal step-opening packet schema and physical admissibility contract.
# 中文：步初因果数据包结构校验与物理可行域契约定义模块。

from datetime import datetime
import math
from protocol import digest
ORIGIN = 1452871800
ORIGIN_UTC = '2016-01-15T15:30:00Z'
CHANNELS = {'grid_import', 'grid_export', 'pv', 'ev'}

def epoch(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def validate_opening(m, now_s):
    if m.get('kind') != 'NATIVE_OPENING_OBSERVATION' or m.get('clock_origin_utc') != ORIGIN_UTC:
        raise ValueError('Fixed native OPSD current packet required')
    if not isinstance(now_s, (int, float)) or not math.isfinite(now_s) or m['decision_s'] != now_s or (not 0 <= now_s < 241200):
        raise ValueError('Only current scheduled nonterminal message in original 67h block')
    if epoch(m['decision_utc']) != ORIGIN + now_s or not m['valid_complete_pair']:
        raise ValueError('Invalid current UTC')
    if m['includes_counter_or_settlement'] or m['includes_SOC'] or m['includes_future_profile']:
        raise ValueError('Forbidden input channel')
    p = m['original_packet']
    if digest(p) != m['original_packet_sha256']:
        raise ValueError('Original packet hash mismatch')
    if p['relative_s'] != now_s or p['decision_utc'] != m['decision_utc'] or p['terminal_only'] or (p['actual_SOC'] is not None) or (p['control_or_solver_result'] is not None):
        raise ValueError('Wrong original packet clock or supplied result')
    raw = {r['channel']: r for r in p['source_records']}
    rr = m['records']
    if len(rr) != 4 or len(p['source_records']) != 4 or set(raw) != CHANNELS or ({r['channel'] for r in rr} != CHANNELS):
        raise ValueError('Exactly four unique channels')
    for r in rr:
        x = raw[r['channel']]
        report = x['timestamp_seconds'] - ORIGIN
        available = epoch(x['modeled_available_at_utc']) - ORIGIN
        if epoch(x['timestamp_utc']) - ORIGIN != report or r['reported_s'] != report or r['available_s'] != available:
            raise ValueError('Report timestamp rewritten')
        if not report <= available <= now_s or not 0 <= now_s - report <= 900:
            raise ValueError('Future/stale native report')
        if r['age_seconds'] != now_s - report or p['reported_age_s'][r['channel']] != now_s - report:
            raise ValueError('Age was rejuvenated')
        if not x['qualified'] or not r['valid'] or x['causal_quality_mask'] or r['causal_quality_mask']:
            raise ValueError('Original quality failure')
        if r['source_id'] != x['source_id'] or not x['source_member_sha256']:
            raise ValueError('Source binding changed')
        start = epoch(x['native_support_start_utc']) - ORIGIN
        end = epoch(x['native_support_end_utc']) - ORIGIN
        if not start < end == report or end - start != x['native_support_seconds'] or end - start > 900 or (r['native_support_start_s'] != start) or (r['native_support_end_s'] != end):
            raise ValueError('Native past support changed')
        v = x['recorded_power_estimate_W'] / 1000
        if not math.isfinite(v) or v < 0 or r['power_kw'] != v:
            raise ValueError('Invalid/rescaled native power')
    vals = {r['channel']: r['power_kw'] for r in rr}
    d = vals['grid_import'] - vals['grid_export'] + vals['pv']
    bg = d - vals['ev']
    if bg < 0 or m['demand_point_kw'] != d or m['pv_point_kw'] != vals['pv'] or (m['EV_submeter_kw'] != vals['ev']) or (m['derived_nonEV_background_kw'] != bg):
        raise ValueError('Whole-house binding or background invalid')
    return (d, vals['pv'])
