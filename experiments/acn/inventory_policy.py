# EN: Paired terminal battery inventory valuation (`delta_stock_kwh`) and adjusted operating-cost accounting.
# 中文：配对终端储能库存差（`delta_stock_kwh`）估值与等效运行成本核算模块。

from control import common_ref, ETOL, END, TARGET
BUFFER_AC_KWH = 25.0 / 6.0
DT = 1.0 / 60.0
ETA = 0.95

def select(*, t, soc, load_kw, nominal, reserve):
    E = soc * 600.0
    A = ETA * (E - 60.0)
    B = BUFFER_AC_KWH
    terminal = t >= END - 86400
    phase = 'NORMAL_BUFFER' if not terminal else 'H24_PREPARATION' if t < END - 3600 else 'TERMINAL_RECOVERY'
    protected = reserve if terminal else reserve + B
    target_conflict = terminal and reserve > ETA * (TARGET - 60.0)
    cmax = min(250.0, max((0.9 - soc) * 600.0 / (ETA * DT), 0.0))
    dmax = min(250.0, max(ETA * 600.0 * (soc - 0.1) / DT, 0.0), max((A - protected) / DT, 0.0))
    headroom = max(200.0 - load_kw, 0.0)
    trace = dict(policy='FIXED_AC_BUFFER_H24_TERMINAL', phase=phase, energy_pre_kwh=E, available_AC_kwh=A, original_R_kwh=reserve, buffer_AC_kwh=B, fixed_guard_hours=DT, terminal_may_use_buffer=terminal, original_reserve_never_lowered=True, protected_nominal_AC_kwh=protected, nominal_discharge_cap_kw=dmax, charge_SOC_power_cap_kw=cmax, current_PCC_charge_headroom_kw=headroom, buffer_deficit_pre_AC_kwh=max(reserve + B - A, 0.0), buffer_goal_exceeds_physical_capacity=reserve + B > 456.0, original_R_goal_exceeds_physical_capacity=reserve > 456.0, terminal_target_conflicts_with_original_R=target_conflict, terminal_buffer_priority='Original R restoration may exceed300; target may use B; only auxiliary B restoration is capped at300', future_plant_information_used=False, new_optimizer_calls=0)
    errors = []
    try:
        oc, od, ordinary = common_ref(*nominal, load_kw, soc, reserve, True)
    except Exception as exc:
        oc = od = 0.0
        ordinary = dict(failure=repr(exc))
        errors.append('ORDINARY_MAP_REFUSED')
    reserve_fill = 0.0
    if terminal:
        deadline = END - 3600 if phase == 'H24_PREPARATION' else END
        left = (deadline - t) / 3600.0
        delta = TARGET - E
        trace.update(target_kwh=TARGET, target_time_s=deadline, remaining_hours=left, delta_target_kwh=delta)
        if abs(delta) <= ETOL:
            rawc = rawd = 0.0
            reason = 'TARGET_REACHED_ZERO_HOLD_NOT_ORDINARY_NOMINAL'
        elif delta > 0:
            rawc = min(delta / (ETA * left), cmax, headroom)
            rawd = 0.0
            reason = 'EARLY_FINITE_RATE_CHARGE'
        else:
            rawc = 0.0
            rawd = min(-delta * ETA / left, dmax, max(load_kw, 0.0))
            reason = 'EARLY_FINITE_RATE_DISCHARGE_BUFFER_MAY_BE_SPENT'
        fill = max(reserve + B - A, 0.0) / (ETA * ETA * DT)
        fill = min(fill, cmax, headroom, max((TARGET - E) / (ETA * DT), 0.0))
        if fill > 0:
            rawc = max(rawc, fill)
            rawd = 0.0
            reason += ';BUFFER_REFILL_BELOW_TARGET_ONLY'
        reserve_fill = min(max(reserve - A, 0.0) / (ETA * ETA * DT), cmax, headroom)
        if reserve_fill > 0:
            rawc = max(rawc, reserve_fill)
            rawd = 0.0
            reason += ';ORIGINAL_R_RESTORATION_HAS_PRIORITY_OVER_TARGET'
    else:
        rawc = min(max(nominal[0], 0.0), cmax)
        rawd = min(max(nominal[1], 0.0), dmax)
        fill = min(max(reserve + B - A, 0.0) / (ETA * ETA * DT), cmax, headroom)
        reason = 'NOMINAL_PROJECTED_TO_ORIGINAL_R_PLUS_FIXED_BUFFER'
        if fill > 0:
            rawc = max(rawc, fill)
            rawd = 0.0
            reason = 'REFILL_BUFFER_WITH_CURRENT_HEADROOM_NO_DISCHARGE'
    trace.update(requested_charge_kw=rawc, requested_discharge_kw=rawd, buffer_refill_kw=fill, original_R_refill_kw=reserve_fill, reason=reason)
    try:
        c, d, mapped = common_ref(rawc, rawd, load_kw, soc, reserve, True)
        trace.update(status='GOVERNED_ACTION_ORIGINAL_COMMON_REF', mapping=mapped, REF_uses_original_R_only=True)
    except Exception as exc:
        c = d = 0.0
        trace.update(status='GOVERNED_ACTION_REFUSED_ZERO_RETAINED', failure=repr(exc))
        errors.append('GOVERNED_MAP_REFUSED')
    return dict(charge_kw=c, discharge_kw=d, ordinary_charge_kw=oc, ordinary_discharge_kw=od, ordinary=ordinary, phase=phase, recovery=trace, errors=errors)
