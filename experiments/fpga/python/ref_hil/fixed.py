# EN: Signed Q16.16 fixed-point arithmetic (1 LSB = 2^-16 kW) and reference model for the ordered reserve-bounded execution filter (REF).
# 中文：有符号 Q16.16 定点算术（1 LSB = 2^-16 kW）与有序备用有界执行滤波器（REF）定点参考模型。

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS
S32_MIN = -(1 << 31)
S32_MAX = (1 << 31) - 1
STATUS_EXCESS = 1 << 0
STATUS_REDUCED_CHARGE = 1 << 1
STATUS_ADDED_DISCHARGE = 1 << 2
STATUS_REMAINING_OVERLIMIT = 1 << 3
STATUS_INVALID_INPUT = 1 << 4
STATUS_SATURATED = 1 << 5
STATUS_SAFE_IDENTITY = 1 << 6
STATUS_FULLY_CLOSED = 1 << 7

def sat_s32(value: int) -> tuple[int, bool]:
    if value > S32_MAX:
        return (S32_MAX, True)
    if value < S32_MIN:
        return (S32_MIN, True)
    return (int(value), False)

def q_from_decimal(value: str | int | float | Decimal) -> int:
    scaled = (Decimal(str(value)) * Decimal(SCALE)).to_integral_value(rounding=ROUND_HALF_EVEN)
    return sat_s32(int(scaled))[0]

def q_to_float(value: int) -> float:
    return int(value) / SCALE

@dataclass(frozen=True)
class FilterInput:
    nominal_import: int
    nominal_charge: int
    nominal_discharge: int
    max_reduce_charge: int
    max_add_discharge: int
    feeder_limit: int

    @classmethod
    def from_kw(cls, values: Iterable[str | int | float | Decimal]) -> 'FilterInput':
        converted = tuple((q_from_decimal(value) for value in values))
        if len(converted) != 6:
            raise ValueError('FilterInput.from_kw requires exactly six values')
        return cls(*converted)

@dataclass(frozen=True)
class FilterResult:
    reduce_charge: int
    add_discharge: int
    executed_charge: int
    executed_discharge: int
    executed_import: int
    remaining_overlimit: int
    status: int
    latency_cycles: int = 2

def _clamp_nonnegative(value: int) -> tuple[int, bool]:
    value, saturated = sat_s32(value)
    if value < 0:
        return (0, True)
    return (value, saturated)

def apply_ref_filter(request: FilterInput) -> FilterResult:
    nominal_import, sat_import = sat_s32(request.nominal_import)
    nominal_charge, bad_charge = _clamp_nonnegative(request.nominal_charge)
    nominal_discharge, bad_discharge = _clamp_nonnegative(request.nominal_discharge)
    max_reduce, bad_reduce = _clamp_nonnegative(request.max_reduce_charge)
    max_add, bad_add = _clamp_nonnegative(request.max_add_discharge)
    feeder_limit, bad_limit = _clamp_nonnegative(request.feeder_limit)
    invalid = bad_charge or bad_discharge or bad_reduce or bad_add or bad_limit
    saturated = sat_import
    excess, sat_excess = sat_s32(nominal_import - feeder_limit)
    saturated |= sat_excess
    excess = max(excess, 0)
    reduce_charge = min(excess, nominal_charge, max_reduce)
    residual_after_charge = excess - reduce_charge
    executed_charge = nominal_charge - reduce_charge
    add_discharge = min(residual_after_charge, max_add) if executed_charge == 0 else 0
    executed_discharge, sat_discharge = sat_s32(nominal_discharge + add_discharge)
    executed_import, sat_exec_import = sat_s32(nominal_import - reduce_charge - add_discharge)
    saturated |= sat_discharge or sat_exec_import
    remaining, sat_remaining = sat_s32(executed_import - feeder_limit)
    saturated |= sat_remaining
    remaining = max(remaining, 0)
    status = 0
    if excess > 0:
        status |= STATUS_EXCESS
    if reduce_charge > 0:
        status |= STATUS_REDUCED_CHARGE
    if add_discharge > 0:
        status |= STATUS_ADDED_DISCHARGE
    if remaining > 0:
        status |= STATUS_REMAINING_OVERLIMIT
    if invalid:
        status |= STATUS_INVALID_INPUT
    if saturated:
        status |= STATUS_SATURATED
    if excess == 0:
        status |= STATUS_SAFE_IDENTITY
    if excess > 0 and remaining == 0:
        status |= STATUS_FULLY_CLOSED
    return FilterResult(reduce_charge=reduce_charge, add_discharge=add_discharge, executed_charge=executed_charge, executed_discharge=executed_discharge, executed_import=executed_import, remaining_overlimit=remaining, status=status)
