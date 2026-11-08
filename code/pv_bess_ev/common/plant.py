# EN: Physical station PCC power balance under PV-aware zero-export realization.
# 中文：光伏感知防逆流（Zero-Export）约束下的场站并网点（PCC）物理功率平衡结算。

from __future__ import annotations
from dataclasses import dataclass
import math

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
    # EN: Clip discharge to local net load max(EV - PV + charge, 0); surplus PV above EV + charge is curtailed rather than exported.
    # 中文：将实际放电裁剪至本地净负荷赤字 max(EV - PV + 充电, 0)；超出本地消纳的光伏出力作弃光处理而不向电网反送。
    discharge = min(requested_discharge, max(demand - pv + charge, 0.0))
    signed = demand + charge - pv - discharge
    grid = max(signed, 0.0)
    curtail = min(max(-signed, 0.0), pv)
    residual = abs(grid + discharge - charge - curtail - (demand - pv))
    violation = max(-grid, -curtail, curtail - pv, discharge - max(demand - pv + charge, 0.0), 0.0)
    if not math.isfinite(residual) or max(residual, violation) > tolerance:
        raise PhysicalConstraintViolation('no-export realization violates physical energy balance', power_balance_residual_kw=residual, physical_bound_violation_kw=violation, pv_actual_kw=pv, pv_curtail_kw=curtail)
    return NoExportAction(charge_kw=charge, discharge_kw=discharge, discharge_clip_kw=requested_discharge - discharge, net_exchange_kw=signed, grid_import_kw=grid, pv_curtail_kw=curtail)
