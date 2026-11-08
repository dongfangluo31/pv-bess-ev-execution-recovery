# EN: Event-driven Reference Governor for ACN workplace charging with 24-hour causal opportunity-based terminal SOC recovery.
# 中文：面向 ACN 事件驱动调度的参考调节器（Reference Governor），含末 24 小时因果机会型终端电量恢复机制。

from .station_physics import common_ref, BATTERY, END, ETOL
DT = 1.0 / 60.0
ETA_C = BATTERY.eta_charge
ETA_D = BATTERY.eta_discharge
REFERENCE = BATTERY.energy_kwh * BATTERY.soc_reference
TARGET = BATTERY.energy_kwh * BATTERY.soc_initial
EMIN = BATTERY.energy_kwh * BATTERY.soc_min

def select(*, t, soc, load_kw, nominal, reserve):
    E = soc * BATTERY.energy_kwh
    A = ETA_D * (E - EMIN)
    terminal = t >= END - 86400
    phase = 'NORMAL_REFERENCE_GOVERNOR' if not terminal else 'H24_OPPORTUNITY_PREPARATION' if t < END - 3600 else 'TERMINAL_OPPORTUNITY_RECOVERY'
    trace = dict(policy='EXISTING_REFERENCE_AND_PRESENT_OPPORTUNITY_GOVERNOR', phase=phase, energy_pre_kwh=E, available_AC_kwh=A, original_R_kwh=reserve, reference_energy_kwh=REFERENCE, target_energy_kwh=TARGET, fixed_guard_hours=DT, auxiliary_buffer_enabled=False, buffer_AC_kwh=0.0, original_reserve_never_lowered=True, future_plant_information_used=False, new_optimizer_calls=0, terminal_target_conflicts_with_original_R=terminal and reserve > ETA_D * (TARGET - EMIN), original_R_goal_exceeds_physical_capacity=reserve > 456.0)
    errors = []
    try:
        oc, od, ordinary = common_ref(*nominal, load_kw, soc, reserve, True)
        original_valid = True
    except Exception as exc:
        oc = od = 0.0
        ordinary = dict(failure=repr(exc))
        errors.append('ORDINARY_MAP_REFUSED')
        original_valid = False
    trace['original_nominal_domain_valid'] = original_valid
    if not terminal and (not original_valid):
        trace.update(status='ORIGINAL_NOMINAL_REFUSAL_RETAINED_ZERO', requested_charge_kw=0.0, requested_discharge_kw=0.0, reason='Invalid nominal command falls back to zero without governor override', original_R_refill_kw=0.0)
        return dict(charge_kw=0.0, discharge_kw=0.0, ordinary_charge_kw=oc, ordinary_discharge_kw=od, ordinary=ordinary, phase=phase, recovery=trace, errors=errors)
    cmax = min(BATTERY.charge_limit_kw, max((BATTERY.soc_max - soc) * BATTERY.energy_kwh / (ETA_C * DT), 0.0))
    rfill = 0.0
    # EN: Ordinary hours: only withhold nominal charge above E_ref (330 kWh) or nominal discharge above current PCC excess; never enlarge nominal actions.
    # 中文：常规时段：仅削减超出参考电量（330 kWh）的名义充电或超出当前并网越限量的名义放电，绝不放大原名义指令。
    if not terminal:
        ref_cap = max((REFERENCE - E) / (ETA_C * DT), 0.0)
        excess = max(load_kw - 200.0, 0.0)
        rawc = min(nominal[0], ref_cap)
        rawd = min(nominal[1], excess)
        trace.update(charge_reference_cap_kw=ref_cap, discharge_actual_excess_cap_kw=excess, nominal_charge_withheld_kw=nominal[0] - rawc, nominal_discharge_withheld_kw=nominal[1] - rawd, reason='Only reduce valid original nominal; no enlargement of original REF permission')
    else:
        delta = TARGET - E
        headroom = max(200.0 - load_kw, 0.0)
        dmax = min(BATTERY.discharge_limit_kw, max(ETA_D * BATTERY.energy_kwh * (soc - BATTERY.soc_min) / DT, 0.0), max((A - reserve) / DT, 0.0))
        if abs(delta) <= ETOL:
            rawc = rawd = 0.0
            reason = 'TARGET_REACHED_ZERO_HOLD_NOT_MPC_NOMINAL'
        elif delta > 0:
            rawc = min(delta / (ETA_C * DT), cmax, headroom)
            rawd = 0.0
            reason = 'USE_CURRENT_CHARGING_OPPORTUNITY_NOW'
        else:
            rawc = 0.0
            rawd = min(-delta * ETA_D / DT, dmax, max(load_kw, 0.0))
            reason = 'USE_CURRENT_LOAD_ABSORPTION_NOW'
        # EN: Terminal 24-h window: opportunistically charge under PCC headroom or absorb local EV load, while prioritizing original reserve R restoration if A < R.
        # 中文：末 24 小时窗口：利用并网余量择机充电或利用本地 EV 负荷消纳放电向 300 kWh 恢复；若可用能量低于原备用 R 则优先补足备用。
        rfill = min(max(reserve - A, 0.0) / (ETA_C * ETA_D * DT), cmax, headroom)
        if rfill > 0:
            rawc = max(rawc, rfill)
            rawd = 0.0
            reason += ';ORIGINAL_R_RESTORATION_PRIORITY_TARGET_CONFLICT_RETAINED'
        trace.update(reason=reason, delta_target_kwh=delta, current_PCC_charge_headroom_kw=headroom, discharge_original_R_cap_kw=dmax, gap_check_hours=DT, deadline_delaying_rate_used=False, nominal_ordinary_is_diagnostic_only=True)
    trace.update(requested_charge_kw=rawc, requested_discharge_kw=rawd, original_R_refill_kw=rfill)
    try:
        c, d, mapped = common_ref(rawc, rawd, load_kw, soc, reserve, True)
        trace.update(status='GOVERNED_ACTION_ORIGINAL_COMMON_REF', mapping=mapped, REF_uses_original_R_only=True)
    except Exception as exc:
        c = d = 0.0
        errors.append('GOVERNED_MAP_REFUSED')
        trace.update(status='GOVERNED_ACTION_REFUSED_ZERO_RETAINED', failure=repr(exc))
    return dict(charge_kw=c, discharge_kw=d, ordinary_charge_kw=oc, ordinary_discharge_kw=od, ordinary=ordinary, phase=phase, recovery=trace, errors=errors)
