import math, time
from dataclasses import asdict
from queue_summary import summarize_queue
from frontload_cruise import step as physical_step
from nominal_environment import environment
VERSION = 'FRONTLOAD_COMPLETION_TAIL'
PTOL = 1e-09
ETOL = 1e-08
HOURS = 1 / 3600.0
BISECTION = 48

def lag_step(p, u, r, a):
    return p + a * max(-r, min(r, u - p))

def e_step(e, p, b):
    return e - (p / b['eta_discharge'] if p >= 0 else b['eta_charge'] * p) * HOURS

def ceil60(n):
    return 60 * math.ceil(n / 60)

def stop_bound(P, r, a):
    P = max(float(P), 0.0)
    sat = max(0, math.ceil(max(P - r, 0.0) / (a * r)))
    after = max(P - sat * a * r, 0.0)
    decay = 0 if after <= PTOL else 1 if a == 1 else max(0, math.ceil(math.log(PTOL / after) / math.log1p(-a)))
    return dict(absolute_power_upper_kw=P, to_r_steps=sat, brake_to_r_seconds=ceil60(sat), decay_steps=decay, joint_zero_steps=sat + decay, joint_zero_seconds=ceil60(sat + decay), bound_kind='R_DOMAIN_AND_ZERO_TOLERANCE_ARE_DIFFERENT')

def conservative_brake(P, r, a, n):
    if not all((math.isfinite(x) for x in (P, r, a))) or P < 0 or r <= 0 or (not 0 < a <= 1) or (n < 0) or n % 60:
        raise ValueError('Invalid fixed brake domain')
    evaluations = 0
    steps = 0

    def endpoint(u):
        nonlocal evaluations, steps
        p = P
        evaluations += 1
        for _ in range(n):
            p = lag_step(p, u, r, a)
            steps += 1
        return p
    limit = r - PTOL
    if n == 0:
        return dict(status='ACCEPTED' if P <= r else 'REFUSED', target_magnitude_kw=0.0, seconds=0, entering_absolute_power_upper_kw=P, endpoint_upper_kw=P, construction_limit_kw=r, construction_inside_margin_kw=0.0, evaluations=0, steps=0, reason='NO_BRAKE_PHASE_ALREADY_IN_R_DOMAIN' if P <= r else 'ZERO_LENGTH_OUTSIDE_R_DOMAIN')
    zero = endpoint(0.0)
    if limit < 0 or zero > limit:
        return dict(status='REFUSED', target_magnitude_kw=0.0, seconds=n, entering_absolute_power_upper_kw=P, endpoint_upper_kw=zero, construction_limit_kw=limit, construction_inside_margin_kw=PTOL, evaluations=evaluations, steps=steps, reason='ZERO_TARGET_CANNOT_REACH_INSIDE_BOUNDARY_IN_ORIGINAL_DURATION')
    lo = 0.0
    hi = r
    end_lo = zero
    end_hi = endpoint(hi)
    if end_hi <= limit:
        lo = hi
        end_lo = end_hi
    else:
        for _ in range(BISECTION):
            mid = (lo + hi) / 2
            end = endpoint(mid)
            if end <= limit:
                lo = mid
                end_lo = end
            else:
                hi = mid
                end_hi = end
    return dict(status='ACCEPTED', target_magnitude_kw=lo, seconds=n, entering_absolute_power_upper_kw=P, endpoint_upper_kw=end_lo, construction_limit_kw=limit, construction_inside_margin_kw=PTOL, root_lower_kw=lo, root_upper_kw=hi, upper_endpoint_kw=end_hi, evaluations=evaluations, steps=steps, reason='FIXED_DURATION_MAXIMUM_SAFE_SIDE_NONZERO_BRAKE')

def signed_limits(E, b, L, pv, G, reserve):
    low = max(-b['charge_limit_kw'], -max(b['soc_max'] * b['energy_kwh'] - E, 0.0) * 60 / b['eta_charge'], L - pv - G)
    high = min(b['discharge_limit_kw'], max(L - pv, 0.0), max(b['eta_discharge'] * (E - b['soc_min'] * b['energy_kwh']) - reserve, 0.0) * 60)
    return (low, high)

def select_completion_tail(*, measurement, queue_state, now_s, episode_end_s, battery, L, pv, G, reserve, ramp, lag, forecast, previous_plan=None, accounting=None):
    started = time.perf_counter()
    b = asdict(battery)
    now = now_s
    T = episode_end_s
    r = ramp
    a = 1.0 if lag == 0 else 1 - math.exp(-1 / lag)
    accounting = {} if accounting is None else accounting
    accounting.update(complete=False, unprojected_evaluations=0, unprojected_steps=0, physical_validation_steps=0, immutable_prefix_physical_steps=0, absolute_envelope_steps=0, current_map_checks=0, brake_root_evaluations=0, brake_root_steps=0)
    if now % 60 or T % 900 or (not T - 3600 <= now < T) or (measurement['sample_s'] != now) or (measurement['available_s'] > now) or (queue_state['as_of_s'] != now):
        raise ValueError('Original last-hour/current telemetry/queue contract required')
    if not all((math.isfinite(v) for v in (measurement['soc'], measurement['tracked_power_kw'], L, pv, G, reserve, r, lag))) or min(L, pv, G, reserve, lag) < 0 or r <= 0:
        raise ValueError('Finite original domain required')
    if not b['soc_min'] <= measurement['soc'] <= b['soc_max']:
        raise ValueError('Actual SOC domain changed')
    target = b['energy_kwh'] * b['soc_initial']
    E0 = measurement['soc'] * b['energy_kwh']
    P0 = measurement['tracked_power_kw']
    if not -b['charge_limit_kw'] - PTOL <= P0 <= b['discharge_limit_kw'] + PTOL:
        raise ValueError('Actual entering power outside original ratings')
    env = environment(forecast, now, T, L, pv)
    onset = now + 60
    bridge = T - 60
    zero_info = stop_bound(r, r, a)
    zero_seconds = max(60, zero_info['joint_zero_seconds'])
    zero_start = bridge - zero_seconds
    active = queue_state.get('last_delivered_command')
    pending = sorted(queue_state.get('pending_commands', []), key=lambda c: (c['effective_s'], c['seq']))
    if any((c['issue_s'] > now for c in pending)):
        raise ValueError('Unknown future command in queue')
    prefix = []
    p = P0
    E = E0
    upper = abs(P0)
    envelope_steps = 0
    for t in range(now, min(onset, T)):
        for c in pending:
            if c['effective_s'] <= t <= c['valid_until_s'] and (active is None or c['seq'] > active['seq']):
                active = c
        u = 0.0 if active is None or active['valid_until_s'] < t else active['signed_kw']
        before = p
        row = physical_step(p, E, u, b, *env[t - now], r, a)
        p = row['p']
        E = row['e']
        upper = max(0.0, lag_step(upper, abs(u), r, a))
        envelope_steps += 1
        accounting['immutable_prefix_physical_steps'] += 1
        accounting['absolute_envelope_steps'] += 1
        prefix.append(dict(sim_s=t, previous_power_kw=before, target_kw=u, predicted_power_kw=p, predicted_energy_kwh=E, predicted_grid_kw=row['grid'], predicted_ramp_override_kw=row['ramp_override'], predicted_projection_kw=row['projection']))
    entry = dict(predicted_power_kw=p, predicted_energy_kwh=E, rows=prefix, conditional_current_point_persistence=True, actual_future_consumed=False)
    old_state = None
    if previous_plan and previous_plan.get('version') == VERSION:
        candidate_state = previous_plan.get('policy_state')
        if candidate_state and candidate_state['episode_end_s'] == T and (candidate_state['brake_end_s'] > onset):
            old_state = dict(candidate_state)
    lock_end = max(onset, old_state['brake_end_s'] if old_state else onset)

    def locked_target(t):
        if old_state and t < old_state['pulse_end_s']:
            return old_state['pulse_target_kw']
        if old_state and t < old_state['brake_target_end_s']:
            return old_state['brake_target_kw']
        return 0.0
    free_upper = upper
    for t in range(onset, min(lock_end, T)):
        free_upper = max(0.0, lag_step(free_upper, abs(locked_target(t)), r, a))
        envelope_steps += 1
        accounting['absolute_envelope_steps'] += 1
    pre_brake = stop_bound(free_upper, r, a)
    free = lock_end + pre_brake['brake_to_r_seconds']
    for _ in range(pre_brake['brake_to_r_seconds']):
        free_upper = max(0.0, lag_step(free_upper, 0.0, r, a))
        envelope_steps += 1
        accounting['absolute_envelope_steps'] += 1
    evaluations = 0
    tail_steps = 0
    attempts = []

    def evaluate(A, H, B, brake, keep=False, brake_target=0.0):
        nonlocal evaluations, tail_steps
        evaluations += 1
        accounting['unprojected_evaluations'] += 1
        pp = p
        ee = E
        energies = {now: E0}
        energies.update({x['sim_s'] + 1: x['predicted_energy_kwh'] for x in prefix})
        bad = []
        powers = []
        es = [E]
        commands = []
        calendar = []
        bridge_state = dict(energy=E0, power=P0) if now == bridge else None
        for x in prefix:
            if x['sim_s'] + 1 == bridge:
                bridge_state = dict(energy=x['predicted_energy_kwh'], power=x['predicted_power_kw'])
        max_grid = 0.0
        min_res = float('inf')
        max_projection = 0.0
        max_ramp = 0.0

        def flag(t, why):
            if len(bad) < 8:
                bad.append(dict(sim_s=t, kind=why))
        for x in prefix:
            sl = b['eta_discharge'] * (x['predicted_energy_kwh'] - b['soc_min'] * b['energy_kwh']) - reserve
            min_res = min(min_res, sl)
            max_grid = max(max_grid, x['predicted_grid_kw'] - G)
            max_ramp = max(max_ramp, x['predicted_ramp_override_kw'])
            if x['predicted_grid_kw'] > G + PTOL or x['predicted_ramp_override_kw'] > PTOL or sl < -PTOL:
                flag(x['sim_s'], 'IMMUTABLE_PREFIX_PHYSICAL_CONFLICT')
            if x['sim_s'] >= bridge - 1 and (abs(x['predicted_power_kw']) > PTOL or abs(x['predicted_energy_kwh'] - target) > ETOL or (x['sim_s'] >= bridge and x['target_kw'] != 0.0)):
                flag(x['sim_s'], 'IMMUTABLE_PREFIX_FINAL_ZERO_TAIL_CONFLICT')
        for t in range(onset, T):
            if t < lock_end:
                u = locked_target(t)
            elif t < free:
                u = 0.0
            elif t >= zero_start:
                u = 0.0
            elif H and t < free + H:
                u = A
            elif H and t < free + H + brake:
                u = brake_target
            else:
                u = B
            if t % 60 == 0:
                decision = t - 60
                lo, hi = signed_limits(energies[decision], b, *env[decision - now], G, reserve)
                accounting['current_map_checks'] += 1
                if u < lo - PTOL or u > hi + PTOL:
                    flag(decision, 'ORIGINAL_CURRENT_MAP_OR_TOTAL_RESERVE_CONFLICT')
                if keep:
                    calendar.append(dict(issue_step=decision, effective_step=t, valid_until_step=decision + 120, target_kw=u))
            previous = pp
            pp = lag_step(pp, u, r, a)
            ee = e_step(ee, pp, b)
            tail_steps += 1
            energies[t + 1] = ee
            accounting['unprojected_steps'] += 1
            ll, vv = env[t - now]
            real = physical_step(previous, es[-1] if keep else energies[t], u, b, ll, vv, r, a)
            accounting['physical_validation_steps'] += 1
            projection = abs(real['p'] - pp)
            grid = max(ll - vv - pp, 0.0)
            slack = b['eta_discharge'] * (ee - b['soc_min'] * b['energy_kwh']) - reserve
            max_grid = max(max_grid, grid - G)
            min_res = min(min_res, slack)
            max_projection = max(max_projection, projection)
            max_ramp = max(max_ramp, real['ramp_override'])
            if projection > PTOL or not b['soc_min'] * b['energy_kwh'] <= ee <= b['soc_max'] * b['energy_kwh'] or grid > G + PTOL or (slack < -PTOL):
                flag(t, 'UNPROJECTED_CANDIDATE_FAILS_ORIGINAL_PHYSICAL_GATE')
            if t >= bridge - 1 and (abs(pp) > PTOL or abs(ee - target) > ETOL or (t >= bridge and u != 0.0)):
                flag(t, 'FINAL_ZERO_TAIL_POINTWISE_CONFLICT')
            if t + 1 == bridge:
                bridge_state = dict(energy=ee, power=pp)
            if keep:
                powers.append(pp)
                es.append(ee)
                commands.append(u)
        residual = ee - target
        if bridge_state is None or abs(bridge_state['energy'] - target) > ETOL or abs(bridge_state['power']) > PTOL or (abs(residual) > ETOL) or (abs(pp) > PTOL):
            flag(T, 'BRIDGE_OR_FINAL_ENDPOINT_CONFLICT')
        if keep:
            calendar.append(dict(issue_step=T - 60, effective_step=T, valid_until_step=T + 60, target_kw=0.0))
        if now == T - 60:
            lo, hi = signed_limits(E0, b, L, pv, G, reserve)
            if not lo - PTOL <= 0 <= hi + PTOL:
                flag(now, 'FINAL_ZERO_NOT_CURRENT_PCC_SAFE')
            if abs(E0 - target) > ETOL or abs(P0) > PTOL:
                flag(now, 'ACTUAL_BRIDGE_ENTRY_NOT_RECOVERED')
        out = dict(A=A, H=H, B=B, brake_seconds=brake, brake_target_kw=brake_target, energy_residual_kwh=residual, power_kw=pp, maximum_grid_excess_kw=max_grid, minimum_reserve_slack_kwh=min_res, maximum_physical_projection_kw=max_projection, maximum_ramp_override_kw=max_ramp, conflicts=bad, passed=not bad, bridge=bridge_state)
        if keep:
            out.update(powers=powers, energies=es, commands=commands, calendar=calendar)
        return out

    def scalar(z):
        return {k: v for k, v in z.items() if k not in ('powers', 'energies', 'commands', 'calendar')}

    def solve(lo, hi, H, B, brake, cruise=False, brake_target=0.0):
        for _ in range(BISECTION):
            m = (lo + hi) / 2
            z = evaluate(m, H, m if cruise else B, brake, brake_target=brake_target)
            if z['energy_residual_kwh'] > 0:
                lo = m
            else:
                hi = m
        choices = [evaluate(x, H, x if cruise else B, brake, True, brake_target) for x in (lo, hi)]
        return min(choices, key=lambda z: (not z['passed'], abs(z['energy_residual_kwh'])))
    chosen = None
    selected_kind = 'REFUSED'
    selected_k = None
    selected_bound = None
    z = evaluate(0.0, 0, 0.0, 0, True)
    attempts.append(dict(kind='ZERO_CONTINUATION', **scalar(z)))
    if z['passed']:
        chosen = z
        selected_kind = 'ZERO_CONTINUATION'
    elif free <= zero_start:
        left = evaluate(-r, 0, -r, 0)
        right = evaluate(r, 0, r, 0)
        attempts.extend([dict(kind='SAFE_CRUISE_LOWER', **scalar(left)), dict(kind='SAFE_CRUISE_UPPER', **scalar(right))])
        if left['energy_residual_kwh'] >= 0 >= right['energy_residual_kwh']:
            z = solve(-r, r, 0, 0.0, 0, True)
            attempts.append(dict(kind='SAFE_CRUISE_REPLAY', **scalar(z)))
            if z['passed']:
                chosen = z
                selected_kind = 'SAFE_CRUISE'
        else:
            direction = 1 if right['energy_residual_kwh'] > 0 else -1
            for k in range(1, max(0, (zero_start - free) // 60) + 1):
                H = k * 60
                limits = []
                for t in range(free, free + H, 60):
                    ll, vv = env[t - 60 - now]
                    limits.append(min(b['discharge_limit_kw'], max(ll - vv, 0.0)) if direction > 0 else min(b['charge_limit_kw'], max(G - ll + vv, 0.0)))
                cap = min(limits) if limits else 0.0
                if cap <= r:
                    attempts.append(dict(kind='INTEGER_CAPACITY', k=k, amplitude_cap=cap, refused='no_capacity_above_r'))
                    continue
                pulse_upper = free_upper
                for _ in range(H):
                    pulse_upper = max(0.0, lag_step(pulse_upper, cap, r, a))
                    envelope_steps += 1
                    accounting['absolute_envelope_steps'] += 1
                bound = stop_bound(pulse_upper, r, a)
                brake = bound['brake_to_r_seconds']
                if free + H + brake > zero_start:
                    attempts.append(dict(kind='INTEGER_CAPACITY', k=k, amplitude_cap=cap, refused='insufficient_state_bound_brake_time'))
                    continue
                certificate = conservative_brake(pulse_upper, r, a, brake)
                bound['nonzero_brake_certificate'] = certificate
                envelope_steps += certificate['steps']
                accounting['absolute_envelope_steps'] += certificate['steps']
                accounting['brake_root_evaluations'] += certificate['evaluations']
                accounting['brake_root_steps'] += certificate['steps']
                if certificate['status'] != 'ACCEPTED':
                    attempts.append(dict(kind='INTEGER_CAPACITY', k=k, amplitude_cap=cap, frozen_brake_bound=bound, refused=certificate['reason']))
                    continue
                brake_target = direction * certificate['target_magnitude_kw']
                lo, hi = (r, cap) if direction > 0 else (-cap, -r)
                B = direction * r
                l = evaluate(lo, H, B, brake, brake_target=brake_target)
                h = evaluate(hi, H, B, brake, brake_target=brake_target)
                attempts.append(dict(kind='INTEGER_CAPACITY', k=k, H=H, amplitude_cap=cap, absolute_end_pulse_upper_kw=pulse_upper, frozen_brake_bound=bound, brake_target_kw=brake_target, low_residual=l['energy_residual_kwh'], high_residual=h['energy_residual_kwh']))
                if l['energy_residual_kwh'] >= 0 >= h['energy_residual_kwh']:
                    selected_k = k
                    selected_bound = bound
                    z = solve(lo, hi, H, B, brake, brake_target=brake_target)
                    attempts.append(dict(kind='SELECTED_INTEGER_REPLAY', k=k, **scalar(z)))
                    if z['passed']:
                        chosen = z
                        selected_kind = 'POSITIVE_PULSE' if direction > 0 else 'NEGATIVE_PULSE'
                    break
    policy_state = None
    if chosen is not None:
        if old_state and lock_end > onset:
            policy_state = dict(old_state)
            policy_state['brake_end_s'] = free
        elif free > onset:
            policy_state = dict(episode_end_s=T, pulse_target_kw=0.0, pulse_end_s=onset, brake_target_kw=0.0, brake_target_end_s=free, brake_end_s=free, committed_issue_s=now, kind='ACTUAL_QUEUE_BOUND_PRE_BRAKE')
        elif chosen['H']:
            policy_state = dict(episode_end_s=T, pulse_target_kw=chosen['A'], pulse_end_s=free + chosen['H'], brake_target_kw=chosen['brake_target_kw'], brake_target_end_s=free + chosen['H'] + chosen['brake_seconds'], brake_end_s=free + chosen['H'] + chosen['brake_seconds'], committed_issue_s=now, kind='NEW_CURRENT_PULSE')
    accounting.update(complete=True, physical_validation_evaluations=evaluations, total_model_recurrences=len(prefix) + 2 * tail_steps + envelope_steps, synthesis_wall_seconds=time.perf_counter() - started)
    result = dict(version=VERSION, issue_s=now, onset_s=onset, bridge_end_s=bridge, episode_end_s=T, initial_shared_energy_target_kwh=target, predicted_entry=entry, telemetry=measurement, known_command_state=summarize_queue(queue_state), forecast=forecast, nominal_current_point=dict(load_kw=L, pv_kw=pv, reserve_kwh=reserve), guard_reserve_requirement_kwh=reserve, hold_rule=dict(N=zero_seconds, alpha=a, rule='FIXED_DURATION_SIGNED_SAFE_SIDE_BRAKE_TO_R_MINUS_PTOL_THEN_CAP_R'), zero_settle_seconds=zero_seconds, final_zero_seconds=60, prefix_absolute_power_upper_kw=upper, free_absolute_power_upper_kw=free_upper, pre_brake_bound=pre_brake, free_phase_start_s=free, selected_k=selected_k, selected_pulse_stop_bound=selected_bound, policy_state=policy_state, old_lock_retained=old_state is not None and chosen is not None, old_lock_cancelled=old_state is not None and chosen is None, future_free_pulse_not_committed=chosen is not None and chosen['H'] > 0 and (free > onset), attempts=attempts, candidate_evaluations=evaluations, prefix_prediction_steps=len(prefix), candidate_tail_prediction_steps=tail_steps, absolute_envelope_steps=envelope_steps, physical_prediction_steps=len(prefix) + tail_steps, active_hold_seconds=0 if chosen is None else chosen['H'], unprojected_prediction_steps=tail_steps, evaluation_accounting=dict(accounting), fixed_bisection_budget=BISECTION, actual_new_optimizer_calls=0, minimum_actual_nontrivial_error_kwh=0.005 * b['energy_kwh'], structural_candidate_evaluation_cap=165, structural_maximum_integer_holds=56, prefix_maximum_physical_projection_kw=max((x['predicted_projection_kw'] for x in prefix), default=0.0), conditional_nominal_witness=chosen is not None, robust_recovery_certificate=False, actual_future_consumed=False, scope='UNPROJECTED_MONOTONE_ROOT_THEN_ORIGINAL_PHYSICAL_GATE; CURRENT_MEASUREMENT_AND_RELEASED_POINT_FORECAST_ONLY', refusal_is_global_infeasibility=False, refusal_policy='Cancel unissued policy lock, preserve immutable queue and original ordinary/MPC/margin/F1 action; do not force unsafe terminal zero.', synthesis_wall_seconds=time.perf_counter() - started)
    if chosen is None:
        result.update(status='REFUSED', calendar=[], witness=dict(status='constraint_conflict', reason='bounded_family_no_verified_tail'))
    else:
        result.update(status='CONDITIONAL_TAIL', selected_kind=selected_kind, pending_zero_renewal_only=onset >= T, calendar=chosen.pop('calendar'), witness=dict(status='reachable', reason='complete_original_gate_nominal_witness', **chosen), predicted_entry_error_kwh=E - target)
    return result
