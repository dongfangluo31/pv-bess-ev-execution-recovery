# EN: Physical PCC power-balance realization under PV-aware zero-export clipping (`discharge <= max(EV - PV + charge, 0)`).
# 中文：光伏感知防逆流裁剪（`放电 <= max(EV - PV + 充电, 0)`）下的并网点物理功率平衡结算。

from __future__ import annotations
from dataclasses import dataclass
import math
PLANT_REALIZATION = 'common_pv_aware_no_export_v2'

class PhysicalConstraintViolation(RuntimeError):

    def __init__(self, message: str, **details: float) -> None:
        self.details = details
        rendered = ', '.join((f'{key}={value:.12g}' for key, value in details.items()))
        super().__init__(f'{message}: {rendered}' if rendered else message)

@dataclass(frozen=True)
class NoExportAction:
    charge_kw: float
    discharge_kw: float
    discharge_clip_kw: float
    net_exchange_kw: float
    grid_import_kw: float
    pv_curtail_kw: float

# EN: Enforce PV-aware zero-export balance: clip discharge to `max(EV - PV + charge, 0)` and curtail surplus PV.
# 中文：执行光伏感知防逆流功率平衡：将放电裁剪至 `max(EV - PV + 充电, 0)`，多余光伏作弃光处理。
def realize_no_export(charge_kw: float, discharge_kw: float, ev_actual_kw: float, pv_actual_kw: float, *, tolerance: float=1e-09) -> NoExportAction:
    values = (charge_kw, discharge_kw, ev_actual_kw, pv_actual_kw, tolerance)
    if not all((math.isfinite(x) for x in values)) or tolerance < 0:
        raise ValueError('no-export inputs and tolerance must be finite')
    if min(charge_kw, discharge_kw, ev_actual_kw, pv_actual_kw) < -tolerance:
        raise ValueError('no-export powers and actual EV/PV must be nonnegative')
    charge = max(float(charge_kw), 0.0)
    requested_discharge = max(float(discharge_kw), 0.0)
    demand = max(float(ev_actual_kw), 0.0)
    pv = max(float(pv_actual_kw), 0.0)
    discharge = min(requested_discharge, max(demand + charge - pv, 0.0))
    signed = demand + charge - pv - discharge
    grid = max(signed, 0.0)
    curtail = min(max(-signed, 0.0), pv)
    residual = abs(grid + discharge - charge - curtail - (demand - pv))
    violation = max(-grid, -curtail, curtail - pv, discharge - demand - charge, 0.0)
    if not math.isfinite(residual) or max(residual, violation) > tolerance:
        raise PhysicalConstraintViolation('no-export realization violates physical energy balance', power_balance_residual_kw=residual, physical_bound_violation_kw=violation, pv_actual_kw=pv, pv_curtail_kw=curtail)
    return NoExportAction(charge_kw=charge, discharge_kw=discharge, discharge_clip_kw=requested_discharge - discharge, net_exchange_kw=signed, grid_import_kw=grid, pv_curtail_kw=curtail)
