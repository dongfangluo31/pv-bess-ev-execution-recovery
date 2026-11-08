# EN: Minute-grid ACN workplace station execution rules (F0, REF, and 2-hour terminal finite-rate REF_RECOVERY) and exact segment integrator.
# 中文：ACN 办公充电场站分钟网格执行规则（F0、REF 及末 2 小时有限速率终端恢复 REF_RECOVERY）与精确事件分段积分器。

from fractions import Fraction as F
from dataclasses import asdict
import math
from ..common.config import BatteryConfig, FilterKind
from ..common.filter import apply_execution_filter, advance_soc_checked
from ..common.plant import realize_no_export
BATTERY = BatteryConfig(energy_kwh=600.0, charge_limit_kw=250.0, discharge_limit_kw=250.0, eta_charge=0.95, eta_discharge=0.95, soc_initial=0.5, soc_min=0.1, soc_max=0.9, soc_reference=0.55)
CAP = 200.0
TARGET = 300.0
END = 29 * 86400
ETOL = 1e-08
PTOL = 1e-09

def common_ref(c, d, L, soc, reserve, ref):
    common = apply_execution_filter(FilterKind.NONE, c, d, L, 0.0, soc, BATTERY, CAP, 1 / 60.0, reserve)
    final = apply_execution_filter(FilterKind.F1_RESERVE, common.charge_kw, common.discharge_kw, L, 0.0, soc, BATTERY, CAP, 1 / 60.0, reserve) if ref else common
    return (final.charge_kw, final.discharge_kw, dict(common=asdict(common), REF=asdict(final) if ref else None))

# EN: Apply F0 or REF during ordinary hours; in the final 2 hours of REF_RECOVERY, steer stored energy toward 300 kWh subject to PCC & reserve bounds.
# 中文：常规时段执行 F0 或 REF；REF_RECOVERY 臂在末 2 小时内按并网限额与备用约束以有限速率将电量拉回 300 kWh 目标。
def choose_minute(*, arm, t, soc, load_kw, nominal, reserve):
    if arm not in ('F0', 'REF', 'REF_RECOVERY') or t % 60 or (not 0 <= t < END):
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

# EN: Integrate exact rational sub-minute EV plug/unplug segments using Fraction timestamps and no-export realization.
# 中文：采用有理数精确时间戳（Fraction）对分钟内充电会话接入/离开子区间逐段执行防逆流功率结算与 SOC 积分。
def integrate_minute(soc, c, d, plant):
    if not all((math.isfinite(x) for x in (soc, c, d))) or min(c, d) < 0 or c > 250 + PTOL or (d > 250 + PTOL) or (c > PTOL and d > PTOL):
        raise ValueError('Invalid executed command')
    cursor = F(plant['start_s'])
    before = soc
    trace = []
    totals = dict(demand_kwh=0.0, grid_kwh=0.0, charge_kwh=0.0, discharge_kwh=0.0, throughput_kwh=0.0, PCC_excess_kwh=0.0, PCC_excess_seconds=0.0, noexport_clip_kwh=0.0, SOC_clip_kwh=0.0)
    maxgrid = 0.0
    for segment in plant['segments']:
        a = F(segment['start_s_exact'])
        b = F(segment['end_s_exact'])
        L = float(F(segment['demand_kw_exact']))
        if a != cursor or b <= a:
            raise ValueError('Noncontiguous exact service segments')
        duration = float((b - a) / 3600)
        e0 = soc * 600.0
        actual_c = min(c, max((0.9 - soc) * 600.0 / (0.95 * duration), 0.0))
        actual_d = min(d, max(0.95 * 600.0 * (soc - 0.1) / duration, 0.0))
        act = realize_no_export(actual_c, actual_d, L, 0.0)
        next_soc = advance_soc_checked(soc, act.charge_kw, act.discharge_kw, BATTERY, duration)
        delta = dict(demand_kwh=L * duration, grid_kwh=act.grid_import_kw * duration, charge_kwh=act.charge_kw * duration, discharge_kwh=act.discharge_kw * duration, throughput_kwh=(act.charge_kw + act.discharge_kw) * duration, PCC_excess_kwh=max(act.grid_import_kw - CAP, 0.0) * duration, PCC_excess_seconds=float(b - a) if act.grid_import_kw > CAP + PTOL else 0.0, noexport_clip_kwh=act.discharge_clip_kw * duration, SOC_clip_kwh=(c - actual_c + d - actual_d) * duration)
        for k, v in delta.items():
            totals[k] += v
        maxgrid = max(maxgrid, act.grid_import_kw)
        trace.append(dict(start_s_exact=str(a), end_s_exact=str(b), demand_kw=L, charge_kw=act.charge_kw, discharge_kw=act.discharge_kw, grid_kw=act.grid_import_kw, energy_pre_kwh=e0, energy_post_kwh=next_soc * 600.0, SOC_pre=soc, SOC_post=next_soc, pre_noexport_charge_kw=actual_c, pre_noexport_discharge_kw=actual_d, noexport_discharge_clip_kw=act.discharge_clip_kw, SOC_charge_clip_kw=c - actual_c, SOC_discharge_clip_kw=d - actual_d, EV_shed_kwh=0.0))
        soc = next_soc
        cursor = b
    if cursor != plant['end_s'] or plant['end_s'] - plant['start_s'] != 60:
        raise ValueError('Incomplete minute support')
    if abs(totals['demand_kwh'] - float(F(plant['energy_kwh_exact']))) > 1e-09:
        raise ValueError('Source energy mismatch')
    totals['common_cost_yuan'] = 0.6 * totals['grid_kwh'] + 0.08 * totals['throughput_kwh']
    return dict(SOC_pre=before, SOC_post=soc, energy_pre_kwh=before * 600.0, energy_post_kwh=soc * 600.0, segments=trace, totals=totals, maximum_grid_kw=maxgrid, EV_shed_kwh=0.0)
