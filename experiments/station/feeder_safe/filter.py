# EN: Reserve-bounded execution filter (REF): ordered charge cancellation (`delta_c`) followed by reserve-bounded extra discharge (`delta_d`).
# 中文：备用有界执行滤波器（REF）：先撤回名义充电（`delta_c`），再在保护可交付备用下限前提下补充放电（`delta_d`）。

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

# EN: Advance SOC over `step_hours` (`+eta_c * P_c * dt / E_cap - P_d * dt / (eta_d * E_cap)`) and assert `[soc_min, soc_max]`.
# 中文：按充放电效率推进步长 `step_hours` 后的 SOC，并严格校验结果落在 `[soc_min, soc_max]` 范围内。
def advance_soc_checked(current_soc: float, charge_kw: float, discharge_kw: float, battery: BatteryConfig, step_hours: float, tolerance: float=1e-09) -> float:
    candidate = current_soc + battery.eta_charge * charge_kw * step_hours / battery.energy_kwh - discharge_kw * step_hours / (battery.eta_discharge * battery.energy_kwh)
    if candidate < battery.soc_min - tolerance or candidate > battery.soc_max + tolerance:
        raise PhysicalConstraintViolation('executed action produces a material SOC violation', soc_pre=current_soc, soc_candidate=candidate, charge_kw=charge_kw, discharge_kw=discharge_kw)
    return min(max(candidate, battery.soc_min), battery.soc_max)

# EN: Apply ordered reserve-bounded execution filter (REF): cancel charging first, then add reserve-safe discharge.
# 中文：执行有序实时修正（REF）：优先撤回名义充电，再在可交付备用允许余量内补充放电。
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
    # EN: Common zero-export plant clipping applied equally to F0 and F1: cap discharge at max(EV - PV + charge, 0).
    # 中文：对 F0 与 F1 一视同仁的防逆流物理裁剪：将放电功率限制在本地净消纳缺口 max(EV - PV + 充电, 0) 以内。
    plant = realize_no_export(charge, discharge, ev_actual_kw, pv_actual_kw, tolerance=tolerance)
    charge, discharge = (plant.charge_kw, plant.discharge_kw)
    working_net_exchange = plant.net_exchange_kw
    reduced = 0.0
    extra = 0.0
    correction_capacity = 0.0
    triggered = mode != FilterKind.NONE and working_net_exchange > feeder_safe_kw + tolerance
    if triggered:
        # EN: Ordered two-channel REF: (1) cancel nominal charging first (1:1 feeder relief while preserving SOC),
        # EN: (2) cover remaining excess via extra discharge bounded by converter rating and deliverable reserve floor.
        # 中文：双通道有序修正：（1）优先撤回名义充电（1:1 压降受电且不消耗储能库存）；
        # 中文：（2）剩余越限量再由额外放电承担，放电上限同时受变流器额定功率与步前锁定的可交付备用余量约束。
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
