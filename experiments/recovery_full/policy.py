# EN: Closed-loop BESS dispatch policy and real-time constraint enforcement.
# 中文：闭环储能调度策略与实时物理约束执行模块。

from station_core.config import BatteryConfig
from recovery_control import safe_target
from tail_core import select_completion_tail, stop_bound, lag_step, PTOL, ETOL

def current_nominal(request):
    now = request['now_s']
    p = request['public_packet']
    m = request['measurement']
    q = request['known_command_state']
    r = request['held_reserve']
    if p['decision_s'] != now or m['sample_s'] != now or m['available_s'] > now or (q['as_of_s'] != now) or (r['issued_s'] > now):
        raise ValueError('Unreleased or misdated current data')
    if any((x['available_s'] > now for x in p['records'])):
        raise ValueError('Future actual sample forbidden')
    L = p['demand_point_kw'] if 'demand_point_kw' in p else p['ev_point_kw']
    pv = p['pv_point_kw']
    forecast = dict(method='CURRENT_PUBLIC_POINT_PERSISTENCE', forecast_issued_s=now, available_s=now, load_kw=L, pv_kw=pv, horizon=dict(ev_point_kw=[L] * 96, pv_point_kw=[pv] * 96), horizon_is_same_point_derived_shape=True, support_is_certified=False, coverage_guarantee=False, future_profile_or_label_consumed=False)
    return (L, pv, forecast)

def propose(request, previous_policy_state=None, constructor=select_completion_tail, accounting=None):
    entry = request['entry']
    now = request['now_s']
    if now < entry['handoff_s']:
        raise ValueError('Preparation unchanged; brake tail starts only at original q')
    L, pv, forecast = current_nominal(request)
    b = BatteryConfig(**entry['battery'])
    accounting = {} if accounting is None else accounting
    previous = None if previous_policy_state is None else dict(version='FRONTLOAD_COMPLETION_TAIL', policy_state=previous_policy_state)
    candidate = constructor(measurement=request['measurement'], queue_state=request['known_command_state'], now_s=now, episode_end_s=entry['end_s'], battery=b, L=L, pv=pv, G=entry['feeder_safe_kw'], reserve=request['held_reserve']['R'], ramp=entry['dynamics']['ramp_kw_per_second'], lag=entry['dynamics']['lag_seconds'], forecast=forecast, previous_plan=previous, accounting=accounting)
    result = dict(method='PREPARATION_BASELINE_PLUS_CURRENT_POINT_CAUSAL_BRAKE', accepted=False, normal_return=True, independent_numerical_baseline=False, reference_stored_plan_consumed=False, future_actual_consumed=False, new_forecast_information_channel=False, robust_unknown_load_guarantee=False, nominal_forecast=forecast, candidate=candidate, accounting=accounting, policy_state=None, raw_target_kw=None, mapping=None, intermediate_service_preserved=True)
    if candidate['witness']['status'] == 'reachable':
        if not candidate['calendar'] or candidate['calendar'][0]['issue_step'] != now:
            raise ValueError('Wrong first causal action')
        target = candidate['calendar'][0]['target_kw']
        c, d, detail = safe_target(target, L, pv, request['measurement']['soc'], b, entry['feeder_safe_kw'], request['held_reserve']['R'])
        if abs(d - c - target) > PTOL:
            result['refusal'] = 'Current common map changes the candidate'
            return result
        result.update(accepted=True, policy_state=candidate['policy_state'], raw_target_kw=target, mapping=dict(charge_kw=c, discharge_kw=d, signed_kw=d - c, details=detail))
    else:
        result['refusal'] = 'Bounded pulse/brake family found no current nominal witness; preserve queue; not global infeasibility'
    return result

def zero_absorption_diagnostic(previous_power_kw, ramp_kw_per_s):
    lower = previous_power_kw - ramp_kw_per_s
    return dict(ramp_lower_kw=lower, noexport_upper_kw=0.0, compatible=lower <= 0.0, positive_conflict_kw=max(lower, 0.0))

def safe_cruise_capacity(entry):
    b = entry['battery']
    r = entry['dynamics']['ramp_kw_per_second']
    seconds = entry['end_s'] - 60 - (entry['handoff_s'] + 60)
    needed = entry['mid_target_kwh'] - entry['final_target_kwh']
    optimistic = r * seconds / (3600 * b['eta_discharge'])
    return dict(seconds=seconds, required_energy_drop_kwh=needed, optimistic_drop_at_r_kwh=optimistic, shortfall_kwh=max(needed - optimistic, 0.0), necessary_energy_condition=needed <= optimistic, does_not_certify_actual_absorption=True)
