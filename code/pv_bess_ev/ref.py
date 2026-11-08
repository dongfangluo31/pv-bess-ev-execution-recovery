# EN: Standalone reserve-bounded execution filter (REF) reference implementation for a single PV-BESS-EV station step.
# 中文：单场站单步备用有界执行滤波器（REF）独立参考实现。
# EN: Sign convention: charge_kw >= 0, discharge_kw >= 0, e_actual_kw >= 0, pv_actual_kw >= 0 (all in kW; simultaneous charge & discharge forbidden).
# 中文：符号与单位约定：充电、放电、EV 负荷与光伏出力均为非负千瓦量（kW），禁止名义指令同时充放电。
# EN: Causal visibility: nominal (c0_kw, d0_kw) and reserve_req_kwh are fixed before step opening; e_actual_kw and pv_actual_kw are metered during execution.
# 中文：因果可见性：名义指令 (c0_kw, d0_kw) 与备用需求 reserve_req_kwh 在步前冻结；实际 EV 负荷与 PV 出力在步内量测可见。

from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class BatteryConfig:
    energy_kwh: float = 600.0
    p_charge_max_kw: float = 250.0
    p_discharge_max_kw: float = 250.0
    soc_min: float = 0.1
    soc_max: float = 0.9
    eta_charge: float = 0.95
    eta_discharge: float = 0.95

@dataclass(frozen=True)
class FilterOutput:
    p_charge_kw: float
    p_discharge_kw: float
    grid_import_kw: float
    pv_curtail_kw: float
    soc_next: float
    triggered: bool
    correction_applied: bool
    charge_reduction_kw: float
    extra_discharge_kw: float
    residual_excess_kw: float
    corrective_headroom_kw: float

def advance_soc_checked(current_soc: float, charge_kw: float, discharge_kw: float, battery: BatteryConfig, step_hours: float, tolerance: float=1e-06) -> float:
    candidate = current_soc + battery.eta_charge * charge_kw * step_hours / battery.energy_kwh - discharge_kw * step_hours / (battery.eta_discharge * battery.energy_kwh)
    if candidate < battery.soc_min - tolerance or candidate > battery.soc_max + tolerance:
        raise ValueError(f'Executed action produces out-of-bounds SOC: {candidate:.6f} outside [{battery.soc_min}, {battery.soc_max}] (pre={current_soc:.4f}, c={charge_kw:.2f}kW, d={discharge_kw:.2f}kW)')
    return min(max(candidate, battery.soc_min), battery.soc_max)

def realize_no_export(c0_kw: float, d0_kw: float, e_actual_kw: float) -> tuple[float, float]:
    p_c = max(0.0, c0_kw)
    p_d = min(max(0.0, d0_kw), e_actual_kw + p_c)
    return (p_c, p_d)

# EN: PV-aware zero-export plant clipping: cap BESS discharge at max(0, EV - PV + charge) so BESS never exports to the grid.
# 中文：光伏感知防逆流物理裁剪：将电池放电限制在 max(0, EV - PV + 充电) 以内，防止光伏大发时储能向电网倒送。
def realize_pv_aware(c0_kw: float, d0_kw: float, e_actual_kw: float, pv_actual_kw: float) -> tuple[float, float]:
    p_c = max(0.0, c0_kw)
    net_local_deficit = max(0.0, e_actual_kw - pv_actual_kw + p_c)
    p_d = min(max(0.0, d0_kw), net_local_deficit)
    return (p_c, p_d)

def apply_ref_correction(c0_kw: float, d0_kw: float, e_actual_kw: float, pv_actual_kw: float, current_soc: float, feeder_limit_kw: float=200.0, reserve_req_kwh: float=0.0, step_hours: float=1.0, battery: BatteryConfig=BatteryConfig(), pv_aware: bool=True, correction_enabled: bool=True, trigger_deadband_kw: float=1e-09, eval_tolerance_kw: float=1e-07, input_tolerance: float=1e-06) -> FilterOutput:
    scalars = [c0_kw, d0_kw, e_actual_kw, pv_actual_kw, current_soc, feeder_limit_kw, reserve_req_kwh, step_hours]
    if not all((math.isfinite(x) for x in scalars)) or step_hours <= 0:
        raise ValueError('Filter inputs must be finite and step duration positive.')
    if min(c0_kw, d0_kw, e_actual_kw, pv_actual_kw, feeder_limit_kw, reserve_req_kwh) < -input_tolerance:
        raise ValueError('Power, demand, and reserve inputs must be non-negative.')
    if current_soc < battery.soc_min - input_tolerance or current_soc > battery.soc_max + input_tolerance:
        raise ValueError(f'Current SOC {current_soc:.4f} outside operational range [{battery.soc_min}, {battery.soc_max}].')
    if c0_kw > input_tolerance and d0_kw > input_tolerance:
        raise ValueError(f'Nominal supervisory dispatch commands simultaneous charging ({c0_kw:.2f}kW) and discharging ({d0_kw:.2f}kW).')
    if c0_kw > battery.p_charge_max_kw + input_tolerance or d0_kw > battery.p_discharge_max_kw + input_tolerance:
        raise ValueError(f'Nominal dispatch exceeds power rating (c0={c0_kw:.2f}, d0={d0_kw:.2f}).')
    max_c_kw = max(0.0, (battery.soc_max - current_soc) * battery.energy_kwh / (battery.eta_charge * step_hours))
    max_d_kw = max(0.0, battery.eta_discharge * battery.energy_kwh * (current_soc - battery.soc_min) / step_hours)
    if c0_kw > max_c_kw + input_tolerance:
        raise ValueError(f'Nominal charging ({c0_kw:.2f}kW) exceeds available SOC charge headroom ({max_c_kw:.2f}kW).')
    if d0_kw > max_d_kw + input_tolerance:
        raise ValueError(f'Nominal discharging ({d0_kw:.2f}kW) exceeds available SOC discharge headroom ({max_d_kw:.2f}kW).')
    c0 = max(0.0, float(c0_kw))
    d0 = max(0.0, float(d0_kw))
    if pv_aware:
        p_c, p_d = realize_pv_aware(c0, d0, e_actual_kw, pv_actual_kw)
    else:
        p_c, p_d = realize_no_export(c0, d0, e_actual_kw)
    r = e_actual_kw - pv_actual_kw + p_c - p_d
    # EN: Reserve-bounded extra discharge headroom: protect deliverable reserve_req_kwh above SOC_min and converter rating p_discharge_max_kw.
    # 中文：受备用约束的额外放电余量：在扣除 SOC 下限与步前锁定的可交付备用需求 reserve_req_kwh 后，同时受变流器额定功率限制。
    e_deliverable_kwh = battery.eta_discharge * battery.energy_kwh * (current_soc - battery.soc_min)
    h_d = max(0.0, e_deliverable_kwh - reserve_req_kwh - p_d * step_hours) / step_hours
    max_add_discharge = min(battery.p_discharge_max_kw - p_d, h_d)
    headroom = p_c + max_add_discharge
    excursion = r - feeder_limit_kw
    triggered = excursion > trigger_deadband_kw
    # EN: Ordered two-channel correction: (1) cancel nominal charging delta_c first (saves stored energy 1:1), (2) add discharge delta_d only for remaining excursion.
    # 中文：双通道有序修正：（1）优先撤回名义充电 delta_c（1:1 降低馈线受电且不消耗储能库存）；（2）剩余越限量再由额外放电 delta_d 承担。
    if correction_enabled and triggered:
        xi_0 = excursion
        delta_c = min(p_c, xi_0)
        delta_d = min(max_add_discharge, xi_0 - delta_c)
        correction_applied = delta_c > 0.0 or delta_d > 0.0
    else:
        delta_c = 0.0
        delta_d = 0.0
        correction_applied = False
    p_c_final = p_c - delta_c
    p_d_final = p_d + delta_d
    r_final = r - delta_c - delta_d
    grid_import = max(0.0, r_final)
    pv_curtail = max(0.0, -r_final)
    residual = max(0.0, r_final - feeder_limit_kw)
    next_soc = advance_soc_checked(current_soc, p_c_final, p_d_final, battery, step_hours, tolerance=input_tolerance)
    return FilterOutput(p_charge_kw=p_c_final, p_discharge_kw=p_d_final, grid_import_kw=grid_import, pv_curtail_kw=pv_curtail, soc_next=next_soc, triggered=triggered, correction_applied=correction_applied, charge_reduction_kw=delta_c, extra_discharge_kw=delta_d, residual_excess_kw=residual, corrective_headroom_kw=headroom)
