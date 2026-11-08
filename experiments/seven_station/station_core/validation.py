# EN: Independent LP/QP reference solvers verifying closed-form single-station, shared-feeder, and network REF projections.
# 中文：用于核验闭式单站、共享馈线及配电网投影算子的独立 LP/QP 数值参考求解器。

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Callable, Iterable, Sequence
import numpy as np
from scipy.optimize import linprog
from .config import BatteryConfig, FilterKind
from .filter import apply_execution_filter

@dataclass(frozen=True)
class CoupledProjectionInput:
    aggregate_excess_kw: float
    reduce_charge_caps_kw: tuple[float, ...]
    extra_discharge_caps_kw: tuple[float, ...]

@dataclass(frozen=True)
class CoupledProjectionSolution:
    reduce_charge_kw: tuple[float, ...]
    extra_discharge_kw: tuple[float, ...]
    residual_excess_kw: float
    objective: float

def _require_finite_nonnegative(values: Iterable[float], label: str) -> tuple[float, ...]:
    checked = tuple((float(value) for value in values))
    if not all((math.isfinite(value) and value >= 0.0 for value in checked)):
        raise ValueError(f'{label} must contain finite nonnegative values')
    return checked

def _coupled_costs(station_count: int) -> tuple[np.ndarray, np.ndarray]:
    station_priority = np.arange(station_count, dtype=float) * 0.001
    return (1.0 + station_priority, 2.0 + station_priority)

def coupled_greedy_projection(problem: CoupledProjectionInput) -> CoupledProjectionSolution:
    reduce_caps = _require_finite_nonnegative(problem.reduce_charge_caps_kw, 'charge caps')
    discharge_caps = _require_finite_nonnegative(problem.extra_discharge_caps_kw, 'discharge caps')
    if len(reduce_caps) == 0 or len(reduce_caps) != len(discharge_caps):
        raise ValueError('coupled caps must be nonempty and have equal length')
    if not math.isfinite(problem.aggregate_excess_kw) or problem.aggregate_excess_kw < 0.0:
        raise ValueError('aggregate excess must be finite and nonnegative')
    charge_cost, discharge_cost = _coupled_costs(len(reduce_caps))
    action = np.zeros(2 * len(reduce_caps), dtype=float)
    caps = np.asarray(reduce_caps + discharge_caps, dtype=float)
    costs = np.concatenate((charge_cost, discharge_cost))
    remaining = float(problem.aggregate_excess_kw)
    for index in np.argsort(costs, kind='stable'):
        allocated = min(float(caps[index]), remaining)
        action[index] = allocated
        remaining -= allocated
        if remaining <= 0.0:
            remaining = 0.0
            break
    objective = float(costs @ action + 10.0 * remaining)
    count = len(reduce_caps)
    return CoupledProjectionSolution(tuple(action[:count]), tuple(action[count:]), remaining, objective)
