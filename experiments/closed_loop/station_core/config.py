# EN: Physical battery ratings (kWh/kW/SOC/efficiency), objective weights, and single-thread HiGHS solver settings.
# 中文：电池物理额定参数（容量/功率/SOC/效率）、目标函数权重与 HiGHS 单线程确定性求解配置。

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from .solver_determinism import HIGHS_PARALLEL, HIGHS_RANDOM_SEED, HIGHS_THREADS, SOC_CANONICAL_DECIMALS

class ControllerKind(str, Enum):
    M6 = 'M6'
    RB = 'RB'
    GS = 'GS'

class FilterKind(str, Enum):
    NONE = 'none'
    F0_PLAIN = 'F0_plain'
    F1_RESERVE = 'F1_reserve'

class Method(str, Enum):
    M6 = 'M6'
    RB_F0 = 'RB_F0'
    RB_F1 = 'RB_F1'
    GS_F0 = 'GS_F0'
    GS_F1 = 'GS_F1'
    GS_F1_ACI = 'GS_F1_ACI'
    RB = 'RB_F0'
    GS = 'GS_F0'
    ACI = 'GS_F1_ACI'

@dataclass(frozen=True)
class MethodSpec:
    controller: ControllerKind
    execution_filter: FilterKind
    adaptive_conformal: bool = False
_METHOD_SPECS = {Method.M6: MethodSpec(ControllerKind.M6, FilterKind.NONE), Method.RB_F0: MethodSpec(ControllerKind.RB, FilterKind.NONE), Method.RB_F1: MethodSpec(ControllerKind.RB, FilterKind.F1_RESERVE), Method.GS_F0: MethodSpec(ControllerKind.GS, FilterKind.NONE), Method.GS_F1: MethodSpec(ControllerKind.GS, FilterKind.F1_RESERVE), Method.GS_F1_ACI: MethodSpec(ControllerKind.GS, FilterKind.F1_RESERVE, True)}

def method_spec(method: Method | str) -> MethodSpec:
    return _METHOD_SPECS[Method(method)]

@dataclass(frozen=True)
class BatteryConfig:
    energy_kwh: float = 600.0
    charge_limit_kw: float = 250.0
    discharge_limit_kw: float = 250.0
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_initial: float = 0.5
    soc_reference: float = 0.55
    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    def __post_init__(self) -> None:
        positive = (self.energy_kwh, self.charge_limit_kw, self.discharge_limit_kw, self.eta_charge, self.eta_discharge)
        if not all((math.isfinite(x) and x > 0 for x in positive)):
            raise ValueError('battery energy, power, and efficiencies must be positive')
        if not 0 <= self.soc_min <= self.soc_initial <= self.soc_max <= 1:
            raise ValueError('SOC bounds/initial state are inconsistent')
        if not self.soc_min <= self.soc_reference <= self.soc_max:
            raise ValueError('SOC reference must lie within the SOC bounds')
        if self.eta_charge > 1 or self.eta_discharge > 1:
            raise ValueError('efficiencies cannot exceed one')

@dataclass(frozen=True)
class ObjectiveWeights:
    peak_yuan_per_kw: float
    curtail_yuan_per_kwh: float
    throughput_yuan_per_kwh: float
    reserve_slack_yuan_per_kwh: float = 0.0
    soc_swing_yuan_per_unit: float = 0.0
    soc_deviation_yuan_per_unit: float = 0.0
    grid_slack_yuan_per_kw: float = 0.0
    pressure_grid_yuan_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        if any((not math.isfinite(x) or x < 0 for x in self.__dict__.values())):
            raise ValueError('objective weights must be finite and nonnegative')
M6_WEIGHTS = ObjectiveWeights(5.0, 0.02, 0.08)
RB_WEIGHTS = ObjectiveWeights(5.0, 0.02, 0.08, reserve_slack_yuan_per_kwh=10.0)
GS_WEIGHTS = ObjectiveWeights(120.0, 0.02, 0.18, reserve_slack_yuan_per_kwh=80.0, soc_swing_yuan_per_unit=80.0, soc_deviation_yuan_per_unit=30.0, grid_slack_yuan_per_kw=300.0, pressure_grid_yuan_per_kwh=0.1)

def weights_for(controller: ControllerKind) -> ObjectiveWeights:
    return {ControllerKind.M6: M6_WEIGHTS, ControllerKind.RB: RB_WEIGHTS, ControllerKind.GS: GS_WEIGHTS}[controller]

@dataclass(frozen=True)
class SolverConfig:
    time_limit_seconds: float = 60.0
    mip_relative_gap: float = 0.001
    mip_feasibility_tolerance: float = 1e-09
    presolve: bool = True
    display: bool = False
    random_seed: int = HIGHS_RANDOM_SEED
    threads: int = HIGHS_THREADS
    parallel: bool = HIGHS_PARALLEL
    soc_canonical_decimals: int = SOC_CANONICAL_DECIMALS

    def __post_init__(self) -> None:
        if self.time_limit_seconds <= 0:
            raise ValueError('time limit must be positive')
        if not 0 <= self.mip_relative_gap < 1:
            raise ValueError('MIP relative gap must be in [0, 1)')
        if not 0 < self.mip_feasibility_tolerance <= 0.001:
            raise ValueError('MIP feasibility tolerance must lie in (0, 1e-3]')
        if isinstance(self.random_seed, bool) or self.random_seed != HIGHS_RANDOM_SEED:
            raise ValueError('formal HiGHS random seed must equal 20260706')
        if isinstance(self.threads, bool) or self.threads != HIGHS_THREADS:
            raise ValueError('formal HiGHS thread count must equal one')
        if type(self.parallel) is not bool or self.parallel is not HIGHS_PARALLEL:
            raise ValueError('formal HiGHS parallel mode must be disabled')
        if isinstance(self.soc_canonical_decimals, bool) or self.soc_canonical_decimals != SOC_CANONICAL_DECIMALS:
            raise ValueError('formal solver SOC canonicalization must use 15 decimals')

@dataclass(frozen=True)
class ExperimentConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    horizon_steps: int = 24
    step_hours: float = 1.0
    feeder_safe_kw: float = 200.0
    pv_rated_kw: float = 300.0
    reserve_hours: float = 1.0
    barrier_alpha: float = 0.3
    scenario_probabilities: tuple[float, float, float] = (0.2, 0.6, 0.2)
    conformal_miscoverage: float = 0.1
    calibration_window: int = 720
    pressure_quantile: float = 1.0
    seed: int = 20260706

    def __post_init__(self) -> None:
        if self.horizon_steps < 1 or self.step_hours <= 0:
            raise ValueError('horizon and step duration must be positive')
        if self.feeder_safe_kw <= 0 or self.pv_rated_kw <= 0:
            raise ValueError('feeder and PV ratings must be positive')
        if self.reserve_hours < 0 or not 0 < self.barrier_alpha <= 1:
            raise ValueError('reserve/barrier parameters are invalid')
        if len(self.scenario_probabilities) != 3:
            raise ValueError('exactly favorable/nominal/adverse probabilities are required')
        if any((p < 0 for p in self.scenario_probabilities)) or not math.isclose(sum(self.scenario_probabilities), 1.0, abs_tol=1e-12):
            raise ValueError('scenario probabilities must be nonnegative and sum to one')
        if not 0 < self.conformal_miscoverage < 1:
            raise ValueError('conformal miscoverage must lie in (0, 1)')
        if self.calibration_window < 1:
            raise ValueError('calibration window must be positive')
        if self.pressure_quantile != 1.0:
            raise ValueError('pressure scale is fixed to the calibration maximum')
