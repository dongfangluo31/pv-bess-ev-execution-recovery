# EN: Station execution-layer operator (NONE, F0_PLAIN without reserve protection, F1_RESERVE with deliverable reserve floor).
# 中文：场站执行层修正算子（NONE 不修正、F0_PLAIN 不保底放电修正、F1_RESERVE 保护可交付备用下限的有序修正）。

from __future__ import annotations
from dataclasses import dataclass
import math
from .config import BatteryConfig, FilterKind
from .plant import PhysicalConstraintViolation, realize_no_export

@dataclass(frozen=True)
class FilterResult:
    mode: FilterKind
    charge_raw_kw: float
    discharge_raw_kw: float
    plant_charge_kw: float
    plant_discharge_kw: float
    common_discharge_clip_kw: float
    plant_net_exchange_kw: float
    plant_grid_kw: float
    plant_curtail_kw: float
    charge_kw: float
    discharge_kw: float
    raw_net_exchange_kw: float
    final_net_exchange_kw: float
    raw_grid_kw: float
    final_grid_kw: float
    curtail_kw: float
    charge_reduction_kw: float
    extra_discharge_kw: float
    triggered: bool
    correction_applied: bool
    correction_capacity_kw: float
    capacity_limited_zero_correction: bool
    capacity_limited_residual: bool
    residual_excess_kw: float
    certificate_applicable: bool

    @property
    def active(self) -> bool:
        return self.triggered

def advance_soc_checked(current_soc: float, charge_kw: float, discharge_kw: float, battery: BatteryConfig, step_hours: float, tolerance: float=1e-09) -> float:
    candidate = current_soc + battery.eta_charge * charge_kw * step_hours / battery.energy_kwh - discharge_kw * step_hours / (battery.eta_discharge * battery.energy_kwh)
    if candidate < battery.soc_min - tolerance or candidate > battery.soc_max + tolerance:
        raise PhysicalConstraintViolation('executed action produces a material SOC violation', soc_pre=current_soc, soc_candidate=candidate, charge_kw=charge_kw, discharge_kw=discharge_kw)
    return min(max(candidate, battery.soc_min), battery.soc_max)

def apply_execution_filter(mode: FilterKind | str, charge_kw: float, discharge_kw: float, ev_actual_kw: float, pv_actual_kw: float, current_soc: float, battery: BatteryConfig, feeder_safe_kw: float, step_hours: float, reserve_need_kwh: float=0.0) -> FilterResult:
    mode = FilterKind(mode)
    scalars = (charge_kw, discharge_kw, ev_actual_kw, pv_actual_kw, current_soc, feeder_safe_kw, step_hours, reserve_need_kwh)
    tolerance = 1e-09
    if not all((math.isfinite(x) for x in scalars)) or step_hours <= 0:
        raise ValueError('filter inputs must be finite and step duration positive')
    if min(charge_kw, discharge_kw, ev_actual_kw, pv_actual_kw, reserve_need_kwh, feeder_safe_kw) < -tolerance:
        raise ValueError('power and reserve inputs must be nonnegative')
    charge_kw = max(float(charge_kw), 0.0)
    discharge_kw = max(float(discharge_kw), 0.0)
    reserve_need_kwh = max(float(reserve_need_kwh), 0.0)
    if charge_kw > battery.charge_limit_kw + tolerance or discharge_kw > battery.discharge_limit_kw + tolerance:
        raise PhysicalConstraintViolation('nominal action exceeds a power limit', charge_kw=charge_kw, discharge_kw=discharge_kw)
    if charge_kw > tolerance and discharge_kw > tolerance:
        raise PhysicalConstraintViolation('nominal action simultaneously charges and discharges', charge_kw=charge_kw, discharge_kw=discharge_kw)
    max_charge_soc_kw = max((battery.soc_max - current_soc) * battery.energy_kwh / (battery.eta_charge * step_hours), 0.0)
    max_discharge_soc_kw = max(battery.eta_discharge * battery.energy_kwh * (current_soc - battery.soc_min) / step_hours, 0.0)
    if charge_kw > max_charge_soc_kw + tolerance or discharge_kw > max_discharge_soc_kw + tolerance:
        raise PhysicalConstraintViolation('nominal action exceeds SOC headroom', charge_kw=charge_kw, discharge_kw=discharge_kw, max_charge_soc_kw=max_charge_soc_kw, max_discharge_soc_kw=max_discharge_soc_kw)
    raw_net_exchange = ev_actual_kw + charge_kw - pv_actual_kw - discharge_kw
    charge = min(float(charge_kw), battery.charge_limit_kw, max_charge_soc_kw)
    discharge = min(float(discharge_kw), battery.discharge_limit_kw, max_discharge_soc_kw)
    plant = realize_no_export(charge, discharge, ev_actual_kw, pv_actual_kw, tolerance=tolerance)
    charge, discharge = (plant.charge_kw, plant.discharge_kw)
    working_net_exchange = plant.net_exchange_kw
    reduced = 0.0
    extra = 0.0
    correction_capacity = 0.0
    # EN: Trigger REF only when post-no-export net import exceeds feeder_safe_kw; cancel charging first, then add reserve-safe discharge.
    # 中文：仅当防逆流实现后的并网净受电超过 feeder_safe_kw 时触发修正；先撤回充电，再在扣除 protected_reserve 后的可用能量内补充放电。
    triggered = mode != FilterKind.NONE and working_net_exchange > feeder_safe_kw + tolerance
    if triggered:
        excess = working_net_exchange - feeder_safe_kw
        charge_correction_capacity = charge
        reduced = min(charge, excess)
        charge -= reduced
        excess -= reduced
        protected_reserve = reserve_need_kwh if mode == FilterKind.F1_RESERVE else 0.0
        usable_energy = max(battery.eta_discharge * battery.energy_kwh * (current_soc - battery.soc_min) - protected_reserve, 0.0)
        remaining_energy = max(usable_energy - discharge * step_hours, 0.0)
        correction_capacity = charge_correction_capacity + min(max(battery.discharge_limit_kw - discharge, 0.0), remaining_energy / step_hours)
        extra = min(max(battery.discharge_limit_kw - discharge, 0.0), remaining_energy / step_hours, max(excess, 0.0))
        discharge += extra
    final = realize_no_export(charge, discharge, ev_actual_kw, pv_actual_kw, tolerance=tolerance)
    if final.discharge_clip_kw > tolerance:
        raise PhysicalConstraintViolation('REF output unexpectedly requires a second no-export adaptation', extra_common_clip_kw=final.discharge_clip_kw)
    charge, discharge = (final.charge_kw, final.discharge_kw)
    final_net_exchange = final.net_exchange_kw
    advance_soc_checked(current_soc, charge, discharge, battery, step_hours, tolerance)
    final_grid = final.grid_import_kw
    curtail = final.pv_curtail_kw
    residual_excess = max(final_grid - feeder_safe_kw, 0.0)
    correction_applied = reduced > tolerance or extra > tolerance
    capacity_limited_residual = triggered and residual_excess > tolerance
    capacity_limited_zero_correction = capacity_limited_residual and (not correction_applied) and (correction_capacity <= tolerance)
    certificate_applicable = mode == FilterKind.F1_RESERVE and (working_net_exchange <= feeder_safe_kw + tolerance or correction_capacity + tolerance >= working_net_exchange - feeder_safe_kw)
    return FilterResult(mode=mode, charge_raw_kw=float(charge_kw), discharge_raw_kw=float(discharge_kw), plant_charge_kw=plant.charge_kw, plant_discharge_kw=plant.discharge_kw, common_discharge_clip_kw=plant.discharge_clip_kw, plant_net_exchange_kw=plant.net_exchange_kw, plant_grid_kw=plant.grid_import_kw, plant_curtail_kw=plant.pv_curtail_kw, charge_kw=charge, discharge_kw=discharge, raw_net_exchange_kw=raw_net_exchange, final_net_exchange_kw=final_net_exchange, raw_grid_kw=max(raw_net_exchange, 0.0), final_grid_kw=final_grid, curtail_kw=curtail, charge_reduction_kw=reduced, extra_discharge_kw=extra, triggered=triggered, correction_applied=correction_applied, correction_capacity_kw=correction_capacity, capacity_limited_zero_correction=capacity_limited_zero_correction, capacity_limited_residual=capacity_limited_residual, residual_excess_kw=residual_excess, certificate_applicable=certificate_applicable)
