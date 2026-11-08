# EN: Closed-form intra-minute hot-spot peak locator, binary-searched thermal current cap K_cap, and degree-minute exceedance integrator.
# 中文：分钟内绕组热点解析极值定位、二分搜索动态热电流上限 K_cap 及超温（度·分）积分统计。

import math
from .model import Parameters, State, equilibrium, advance

def curves(parameters, state, K, ambient, seconds=60.0):
    if parameters.k21 != 1 or state.h2_k != 0:
        raise ValueError('Two-exponential interval contract requires k21=1 and actual h2=0')
    if not math.isfinite(K) or K < 0 or seconds <= 0:
        raise ValueError('Finite nonnegative actual current/time domain')
    goal = equilibrium(parameters, K, ambient)
    to, tw, _ = parameters.time_constants_min
    A = state.oil_c - goal.oil_c
    B = state.h1_k - goal.h1_k
    T = seconds / 60.0
    candidates = [0.0, T]
    stationary = None
    # EN: When oil and winding transients move in opposite directions (A * B < 0), check interior stationary point `stationary` inside (0, T).
    # 中文：当顶层油温与绕组梯度暂态方向相反（A * B < 0）时，解析求解区间 (0, T) 内的内部驻点极值时刻。
    if A * B < 0 and to != tw:
        stationary = (math.log(abs(B)) + math.log(to) - math.log(abs(A)) - math.log(tw)) / (1 / tw - 1 / to)
        if 0 < stationary < T:
            candidates.append(stationary)
    oil = lambda t: goal.oil_c + A * math.exp(-t / to)
    hot = lambda t: goal.hotspot_c + A * math.exp(-t / to) + B * math.exp(-t / tw)
    return dict(T=T, oil=oil, hot=hot, goal=goal, A=A, B=B, to=to, tw=tw, candidates=sorted(candidates), stationary_min=stationary)

def interval_peaks(parameters, state, K, ambient, seconds=60.0):
    c = curves(parameters, state, K, ambient, seconds)
    end = advance(parameters, state, current_pu=K, ambient_c=ambient, seconds=seconds)
    oil = [(state.oil_c, 0.0), (end.oil_c, c['T'])]
    hot = [(advance(parameters, state, current_pu=K, ambient_c=ambient, seconds=t * 60).hotspot_c, t) for t in c['candidates']]
    po = max(oil)
    ph = max(hot)
    return dict(oil_peak_c=po[0], oil_peak_offset_s=60 * po[1], hotspot_peak_c=ph[0], hotspot_peak_offset_s=60 * ph[1], stationary_offset_s=60 * c['stationary_min'] if c['stationary_min'] is not None else None, interior_candidate=any((0 < t < c['T'] for t in c['candidates'])))

def current_cap(parameters, state, ambient, oil_limit=105.0, hotspot_limit=120.0, max_K=1.5, iterations=60, temperature_guard_c=1e-09, before_prediction=None):
    count = 0

    def feasible(K):
        nonlocal count
        if before_prediction is not None:
            before_prediction(K, count + 1)
        count += 1
        c = curves(parameters, state, K, ambient)
        peaks = interval_peaks(parameters, state, K, ambient)
        future = advance(parameters, state, current_pu=K, ambient_c=ambient, seconds=60.0)
        positive_hot = [advance(parameters, state, current_pu=K, ambient_c=ambient, seconds=t * 60).hotspot_c for t in c['candidates'] if t > 0]
        ok = state.oil_c <= oil_limit and state.hotspot_c <= hotspot_limit and (future.oil_c <= oil_limit - temperature_guard_c) and (max(positive_hot) <= hotspot_limit - temperature_guard_c)
        return ok
    if not feasible(0.0):
        return dict(K_cap=0.0, thermal_feasible=False, reason='NO_FEASIBLE_CURRENT_INCLUDING_INITIAL_STATE', prediction_evaluations=count, iterations=0)
    if feasible(max_K):
        return dict(K_cap=max_K, thermal_feasible=True, reason='NORMAL_CYCLIC_CURRENT_CAP', prediction_evaluations=count, iterations=0)
    lo, hi = (0.0, max_K)
    used = 0
    for _ in range(iterations):
        middle = (lo + hi) / 2
        if middle == lo or middle == hi:
            break
        used += 1
        if feasible(middle):
            lo = middle
        else:
            hi = middle
    return dict(K_cap=lo, thermal_feasible=True, reason='INTERVAL_THERMAL_BOUND', prediction_evaluations=count, iterations=used, K_upper_infeasible=hi)

def exceedance(curve, primitive, limit, breaks):
    duration = 0.0
    integral = 0.0
    last_end = None
    for a, b in zip(breaks, breaks[1:]):
        fa, fb = (curve(a) - limit, curve(b) - limit)
        if fa <= 0 and fb <= 0:
            continue
        left, right = (a, b)
        if (fa > 0) != (fb > 0):
            lo, hi = (a, b)
            for _ in range(60):
                m = (lo + hi) / 2
                if m == lo or m == hi:
                    break
                if (curve(m) - limit > 0) == (fa > 0):
                    lo = m
                else:
                    hi = m
            crossing = (lo + hi) / 2
            if fa > 0:
                right = crossing
            else:
                left = crossing
        if right > left:
            duration += right - left
            integral += max(primitive(right) - primitive(left) - limit * (right - left), 0.0)
            last_end = right
    return dict(exceed_seconds=duration * 60.0, exceed_degree_minutes=integral, last_exceed_end_offset_s=last_end * 60.0 if last_end is not None else None)

def interval_statistics(parameters, state, K, ambient, seconds=60.0, oil_limit=105.0, hotspot_limit=120.0):
    c = curves(parameters, state, K, ambient, seconds)
    goal = c['goal']
    A, B, to, tw = (c[k] for k in ('A', 'B', 'to', 'tw'))
    io = lambda t: goal.oil_c * t - A * to * math.exp(-t / to)
    ih = lambda t: goal.hotspot_c * t - A * to * math.exp(-t / to) - B * tw * math.exp(-t / tw)
    peaks = interval_peaks(parameters, state, K, ambient, seconds)
    return dict(**peaks, oil_peak_positive_residual_k=max(peaks['oil_peak_c'] - oil_limit, 0.0), hotspot_peak_positive_residual_k=max(peaks['hotspot_peak_c'] - hotspot_limit, 0.0), oil=exceedance(c['oil'], io, oil_limit, [0.0, c['T']]), hotspot=exceedance(c['hot'], ih, hotspot_limit, c['candidates']))

# EN: Discharge BESS up to converter and SOC_min limits whenever EV request_kw exceeds thermally safe transformer power grid_cap_kw.
# 中文：当充电站需求 request_kw 超过变压器热安全折算功率上限 grid_cap_kw 时，在变流器与 SOC_min 范围内放电削峰。
def reference_filter(request_kw, grid_cap_kw, soc, battery, seconds=60.0):
    dt = seconds / 3600.0
    available = max(battery['eta_discharge'] * battery['energy_kwh'] * (soc - battery['soc_min']), 0.0)
    wanted = max(request_kw - grid_cap_kw, 0.0)
    discharge = min(wanted, battery['discharge_limit_kw'], available / dt, request_kw)
    grid = request_kw - discharge
    post = soc - discharge * dt / (battery['eta_discharge'] * battery['energy_kwh'])
    return dict(charge_kw=0.0, discharge_kw=discharge, grid_kw=grid, soc_post=post, requested_correction_kw=wanted, unmet_correction_kw=max(wanted - discharge, 0.0), unserved_kw=0.0)
