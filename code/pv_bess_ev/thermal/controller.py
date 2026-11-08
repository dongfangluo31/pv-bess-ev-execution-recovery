# EN: High-precision 64-digit Decimal thermal feasibility predicate and 60-bisection dynamic transformer current-cap selector.
# 中文：64 位十进制高精度热可行性判定谓词与 60 次二分搜索变压器动态电流限幅控制器。

import math
from dataclasses import asdict
from decimal import Decimal, localcontext
from .model import Parameters, State, advance
from .interval_control import curves, interval_peaks, interval_statistics, reference_filter

# EN: Evaluate 1-minute endpoint oil temperature and positive-time hot-spot peak in 64-digit Decimal against guarded limits.
# 中文：采用 64 位有效数字 Decimal 精确评估 1 分钟末油温及正时刻绕组热点峰值是否满足带保护裕度的热限值。
def decimal_predicate(common, pre, ambient, current_pu):
    with localcontext() as ctx:
        ctx.prec = 64
        D = lambda value: Decimal.from_float(float(value))
        p = common['thermal_parameters']
        k, one, zero = (D(current_pu), Decimal(1), Decimal(0))
        to = D(p['k11']) * D(p['oil_time_constant_min'])
        tw = D(p['k22']) * D(p['winding_time_constant_min'])
        if p['k21'] != 1 or pre['h2_k'] != 0:
            raise ValueError('Guarded predicate requires k21=1 and h2=0')
        ratio = D(p['load_to_no_load_loss_ratio'])
        oil_goal = D(ambient) + D(p['oil_rise_rated_k']) * (D(p['oil_exponent']) * ((one + ratio * k * k) / (one + ratio)).ln()).exp()
        winding_goal = zero if k == 0 else D(p['hotspot_gradient_rated_k']) * (D(p['winding_exponent']) * k.ln()).exp()
        a, b = (D(pre['oil_c']) - oil_goal, D(pre['h1_k']) - winding_goal)
        times = [one]
        if a * b < 0 and to != tw:
            star = (-b * to / (a * tw)).ln() / (one / tw - one / to)
            if zero < star < one:
                times.append(star)

        def oil(t):
            return oil_goal + a * (-t / to).exp()

        def hotspot(t):
            return oil(t) + winding_goal + b * (-t / tw).exp()
        endpoint = oil(one)
        positive_peak = max((hotspot(t) for t in times))
        olim, hlim = (D(common['limits']['oil_c']), D(common['limits']['hotspot_c']))
        guard = D(common['controller']['positive_time_guard_c'])
        initial_ok = D(pre['oil_c']) <= olim and D(pre['oil_c']) + D(pre['h1_k']) <= hlim
        feasible = initial_ok and endpoint <= olim - guard and (positive_peak <= hlim - guard)
        return dict(feasible=bool(feasible), precision=64, input_interpretation='Exact binary64 via Decimal.from_float', oil_endpoint_c=str(endpoint), positive_time_hotspot_peak_c=str(positive_peak), oil_guard_limit_c=str(olim - guard), hotspot_guard_limit_c=str(hlim - guard), initial_state_feasible=bool(initial_ok), positive_time_candidates=len(times), precision_scope='64-digit arithmetic predicate; not a formal interval-arithmetic certificate')

def predict(common, pre, ambient, current_pu):
    p = Parameters(**common['thermal_parameters'])
    s = State(**pre)
    c = curves(p, s, current_pu, ambient)
    peaks = interval_peaks(p, s, current_pu, ambient)
    end = advance(p, s, current_pu=current_pu, ambient_c=ambient, seconds=60)
    positive_hot = [advance(p, s, current_pu=current_pu, ambient_c=ambient, seconds=t * 60).hotspot_c for t in c['candidates'] if t > 0]
    oil, hot = (common['limits']['oil_c'], common['limits']['hotspot_c'])
    guard = common['controller']['positive_time_guard_c']
    decimal = decimal_predicate(common, pre, ambient, current_pu)
    return dict(current_pu=current_pu, feasible=decimal['feasible'], **peaks, positive_time_hotspot_peak_c=max(positive_hot), oil_endpoint_c=end.oil_c, positive_time_guard_c=guard, decimal_predicate=decimal, internal_work=dict(float_peak_evaluations=1, decimal_guard_evaluations=1, float_positive_candidates=len(positive_hot), decimal_positive_candidates=decimal['positive_time_candidates']))

def select_cap(evaluate):
    records = []

    def feasible(k):
        r = evaluate(k, len(records))
        records.append(r)
        return r['feasible']
    if not feasible(0.0):
        return dict(K_cap=0.0, thermal_feasible=False, reason='NO_FEASIBLE_CURRENT', iterations=0, prediction_count=len(records), predictions=records)
    if feasible(1.5):
        return dict(K_cap=1.5, thermal_feasible=True, reason='MAX_CURRENT_CAP', iterations=0, prediction_count=len(records), predictions=records)
    lo, hi, used = (0.0, 1.5, 0)
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        used += 1
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return dict(K_cap=lo, K_upper_infeasible=hi, thermal_feasible=True, reason='THERMAL_INTERVAL_BOUND', iterations=used, prediction_count=len(records), predictions=records)

# EN: Convert per-unit transformer current cap k in [0, 1.5] to three-phase 0.38 kV active power cap (capped at 300 kW).
# 中文：将标幺电流上限 k ∈ [0, 1.5] 折算为三相 0.38 kV 低压侧有功功率上限（最高不超过 300 kW）。
def power_cap(common, k):
    if not math.isfinite(k) or not 0 <= k <= 1.5:
        raise ValueError('Current cap out of frozen controller domain')
    a = common['asset']
    return min(300.0, math.sqrt(3) * 0.38 * a['rated_lv_current_ka'] * 1000 * k)

def action(common, requested_kw, k, soc):
    out = reference_filter(requested_kw, power_cap(common, k), soc, common['battery'])
    out['free_shed_kw'] = 0.0
    out['delivered_service_kw'] = requested_kw
    return out

def thermal_step(common, pre, ambient, current_pu):
    p, s = (Parameters(**common['thermal_parameters']), State(**pre))
    post = advance(p, s, current_pu=current_pu, ambient_c=ambient, seconds=60)
    stats = interval_statistics(p, s, current_pu, ambient, oil_limit=105.0, hotspot_limit=120.0)
    return dict(thermal_post=asdict(post), interval=stats, hotspot_post_c=post.hotspot_c, actual_current_pu=current_pu)
