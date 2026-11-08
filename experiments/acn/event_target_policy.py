# EN: Closed-loop BESS dispatch policy and real-time constraint enforcement.
# 中文：闭环储能调度策略与实时物理约束执行模块。

from control import common_ref, END, TARGET, ETOL, PTOL, CAP

def choose_event_target(*, arm, t, soc, load_kw, nominal, reserve):
    if arm not in ('F0', 'REF', 'REF_RECOVERY') or not 0 <= t < END:
        raise ValueError('Fixed arm/time required')
    c, d = nominal
    errors = []
    try:
        oc, od, ordinary = common_ref(c, d, load_kw, soc, reserve, arm != 'F0')
    except Exception as exc:
        oc = od = 0.0
        ordinary = dict(failure=repr(exc))
        errors.append('ORDINARY_COMMON_OR_REF_INPUT_REFUSED')
    result = dict(charge_kw=oc, discharge_kw=od, ordinary_charge_kw=oc, ordinary_discharge_kw=od, ordinary=ordinary, phase='ORDINARY', recovery=None, errors=errors)
    if arm != 'REF_RECOVERY' or t < END - 7200:
        return result
    phase = 'PREPARATION' if t < END - 3600 else 'RECOVERY'
    deadline = END - 3600 if phase == 'PREPARATION' else END
    remaining_h = (deadline - t) / 3600.0
    e = soc * 600.0
    delta = TARGET - e
    if abs(delta) <= ETOL:
        requested_c = requested_d = 0.0
        reason = 'TARGET_REACHED_REQUEST_ZERO_HOLD'
    elif delta > 0:
        requested_c = min(delta / (0.95 * remaining_h), 250.0, max(CAP - load_kw, 0.0), max((0.9 - soc) * 600.0 / (0.95 / 60.0), 0.0))
        requested_d = 0.0
        reason = 'FINITE_RATE_CHARGE_TOWARD_TARGET'
    else:
        requested_d = min(-delta * 0.95 / remaining_h, 250.0, max(load_kw, 0.0), max(0.95 * 600.0 * (soc - 0.1) * 60.0, 0.0), max((0.95 * (e - 60.0) - reserve) * 60.0, 0.0))
        requested_c = 0.0
        reason = 'FINITE_RATE_DISCHARGE_TOWARD_TARGET'
    trace = dict(policy='ACN_IDEAL_FINITE_RATE_TARGET_FEEDBACK', phase=phase, target_kwh=TARGET, target_time_s=deadline, energy_pre_kwh=e, delta_to_target_kwh=delta, remaining_hours=remaining_h, requested_charge_kw=requested_c, requested_discharge_kw=requested_d, reason=reason, only_current_SOC_load_and_shared_reserve=True, future_events_read=False, hard_terminal_guarantee=False, original_ramp_lag_proof_reused=False)
    try:
        rc, rd, mapped = common_ref(requested_c, requested_d, load_kw, soc, reserve, True)
        trace.update(status='TARGET_ACTION_COMMON_REF_MAPPED', mapping=mapped, REF_modified_target_request=abs(rc - requested_c) > PTOL or abs(rd - requested_d) > PTOL)
        result.update(charge_kw=rc, discharge_kw=rd)
    except Exception as exc:
        trace.update(status='TARGET_ACTION_REFUSED_RETAIN_ORDINARY', error=repr(exc))
    result.update(phase=phase, recovery=trace)
    return result
