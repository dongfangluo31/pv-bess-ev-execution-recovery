# EN: Three-scenario (favorable / nominal / adverse) rolling-horizon MILP scheduler solved deterministically with SciPy/HiGHS.
# 中文：基于 SciPy/HiGHS 确定性求解的三场景（乐观/名义/最不利）滚动时域 MILP 名义调度模型。

from __future__ import annotations
from dataclasses import dataclass
import math
import time
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .config import ControllerKind, ExperimentConfig, Method, method_spec, weights_for
from .solver_determinism import canonicalize_solver_soc, highs_determinism_options
from .solver_attempts import call_external_milp

class SolveFailure(RuntimeError):
    pass
PRESOLVE_POLICY_VERSION = 2
PRESOLVE_GUARD_REASON = 'SOC_UPPER_BOUND_HIGHS_PRESOLVE_GUARD'
PRESOLVE_GUARD_SOC_ATOL = 1e-12

@dataclass(frozen=True)
class PresolvePolicy:
    version: int
    requested: bool
    effective: bool
    guard_applied: bool
    guard_reason: str | None
    soc_atol: float

def resolve_presolve_policy(*, requested: bool, step_hours: float, current_soc: float, soc_max: float, soc_atol: float=PRESOLVE_GUARD_SOC_ATOL) -> PresolvePolicy:
    if not all((math.isfinite(value) for value in (step_hours, current_soc, soc_max, soc_atol))):
        raise ValueError('presolve policy inputs must be finite')
    if step_hours <= 0 or soc_atol < 0:
        raise ValueError('step_hours must be positive and soc_atol nonnegative')
    at_upper = math.isclose(current_soc, soc_max, rel_tol=0.0, abs_tol=soc_atol)
    guard_applied = bool(requested) and at_upper
    return PresolvePolicy(version=PRESOLVE_POLICY_VERSION, requested=bool(requested), effective=False if guard_applied else bool(requested), guard_applied=guard_applied, guard_reason=PRESOLVE_GUARD_REASON if guard_applied else None, soc_atol=float(soc_atol))

@dataclass(frozen=True)
class HorizonData:
    ev_lower_kw: np.ndarray
    ev_point_kw: np.ndarray
    ev_upper_kw: np.ndarray
    pv_lower_kw: np.ndarray
    pv_point_kw: np.ndarray
    pv_upper_kw: np.ndarray
    price_yuan_per_kwh: np.ndarray
    pressure: np.ndarray

    def validated(self) -> 'HorizonData':
        arrays = {name: np.asarray(value, dtype=float).reshape(-1) for name, value in self.__dict__.items()}
        sizes = {array.size for array in arrays.values()}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) < 1:
            raise ValueError('all horizon arrays must be nonempty and aligned')
        if any((not np.all(np.isfinite(array)) for array in arrays.values())):
            raise ValueError('horizon data contain non-finite values')
        if np.any(arrays['ev_lower_kw'] > arrays['ev_point_kw']) or np.any(arrays['ev_point_kw'] > arrays['ev_upper_kw']):
            raise ValueError('EV intervals are not ordered')
        if np.any(arrays['pv_lower_kw'] > arrays['pv_point_kw']) or np.any(arrays['pv_point_kw'] > arrays['pv_upper_kw']):
            raise ValueError('PV intervals are not ordered')
        if np.any(arrays['pressure'] < 0):
            raise ValueError('pressure must be nonnegative')
        return HorizonData(**arrays)

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.ev_point_kw).size)

@dataclass(frozen=True)
class FirstStepDecision:
    method: Method
    controller: ControllerKind
    charge_kw: float
    discharge_kw: float
    planned_grid_kw: float
    planned_curtail_kw: float
    planned_soc: float
    reserve_slack_kwh: float
    grid_slack_kw: float
    objective: float
    solver_status: int
    solver_message: str
    solver_seconds: float
    mip_gap: float | None
    mip_node_count: int | None
    max_constraint_violation: float
    max_bound_violation: float
    max_integrality_violation: float
    first_stage_coupling_residual: float
    first_step_power_balance_residual: float
    first_step_soc_dynamics_residual: float
    presolve_policy_version: int = PRESOLVE_POLICY_VERSION
    presolve_requested: bool = True
    presolve_effective: bool = True
    presolve_guard_applied: bool = False
    presolve_guard_reason: str | None = None
    presolve_guard_soc_atol: float = PRESOLVE_GUARD_SOC_ATOL

class _Indices:

    def __init__(self, scenarios: int, horizon: int) -> None:
        self.scenarios = scenarios
        self.horizon = horizon
        self.offset = 0
        self.fields: dict[str, np.ndarray] = {}

    def matrix(self, name: str) -> np.ndarray:
        size = self.scenarios * self.horizon
        result = np.arange(self.offset, self.offset + size).reshape(self.scenarios, self.horizon)
        self.offset += size
        self.fields[name] = result
        return result

    def vector(self, name: str) -> np.ndarray:
        result = np.arange(self.offset, self.offset + self.scenarios)
        self.offset += self.scenarios
        self.fields[name] = result
        return result

# EN: Build and solve the 3-scenario rolling-horizon MILP via single-thread HiGHS and return the non-anticipative step-0 action.
# 中文：构建并由单线程 HiGHS 求解三场景滚动时域 MILP，返回满足首步非预期性耦合的名义充放电指令。
def solve_first_step(method: Method | str, horizon: HorizonData, current_soc: float, config: ExperimentConfig | None=None) -> FirstStepDecision:
    cfg = config or ExperimentConfig()
    method = Method(method)
    spec = method_spec(method)
    controller = spec.controller
    data = horizon.validated()
    battery = cfg.battery
    if not battery.soc_min <= current_soc <= battery.soc_max:
        raise ValueError('current SOC lies outside configured bounds')
    current_soc = canonicalize_solver_soc(current_soc, soc_min=battery.soc_min, soc_max=battery.soc_max, decimals=cfg.solver.soc_canonical_decimals)
    h = data.horizon
    s_num = 3
    probs = np.asarray(cfg.scenario_probabilities, dtype=float)
    # EN: Pair opposite uncertainty tails across 3 scenarios: s=0 favorable (EV low, PV high), s=1 point, s=2 adverse (EV high, PV low).
    # 中文：跨三场景反向配对不确定性分位：s=0 乐观（EV 下界 + PV 上界），s=1 点预测，s=2 最不利（EV 上界 + PV 下界）。
    ev_s = np.vstack((data.ev_lower_kw, data.ev_point_kw, data.ev_upper_kw))
    pv_s = np.vstack((data.pv_upper_kw, data.pv_point_kw, data.pv_lower_kw))
    ind = _Indices(s_num, h)
    grid = ind.matrix('grid')
    curt = ind.matrix('curt')
    charge = ind.matrix('charge')
    discharge = ind.matrix('discharge')
    soc = ind.matrix('soc')
    binary = ind.matrix('binary')
    reserve_slack = ind.matrix('reserve_slack') if controller != ControllerKind.M6 else None
    swing = ind.matrix('swing') if controller == ControllerKind.GS else None
    deviation = ind.matrix('deviation') if controller == ControllerKind.GS else None
    grid_slack = ind.matrix('grid_slack') if controller == ControllerKind.GS else None
    peak = ind.vector('peak')
    n_var = ind.offset
    weights = weights_for(controller)
    objective = np.zeros(n_var)
    dt = cfg.step_hours
    for s in range(s_num):
        objective[grid[s]] = probs[s] * (data.price_yuan_per_kwh + weights.pressure_grid_yuan_per_kwh * data.pressure) * dt
        objective[curt[s]] = probs[s] * weights.curtail_yuan_per_kwh * dt
        objective[charge[s]] = probs[s] * weights.throughput_yuan_per_kwh * dt
        objective[discharge[s]] = probs[s] * weights.throughput_yuan_per_kwh * dt
        objective[peak[s]] = probs[s] * weights.peak_yuan_per_kw
        if reserve_slack is not None:
            objective[reserve_slack[s]] = probs[s] * weights.reserve_slack_yuan_per_kwh
        if swing is not None and deviation is not None and (grid_slack is not None):
            objective[swing[s]] = probs[s] * weights.soc_swing_yuan_per_unit
            objective[deviation[s]] = probs[s] * weights.soc_deviation_yuan_per_unit
            objective[grid_slack[s]] = probs[s] * weights.grid_slack_yuan_per_kw
    lower_bounds = np.zeros(n_var)
    upper_bounds = np.full(n_var, np.inf)
    upper_bounds[charge] = battery.charge_limit_kw
    upper_bounds[discharge] = battery.discharge_limit_kw
    for s in range(s_num):
        upper_bounds[curt[s]] = pv_s[s]
    lower_bounds[soc] = battery.soc_min
    upper_bounds[soc] = battery.soc_max
    upper_bounds[binary] = 1.0
    integrality = np.zeros(n_var, dtype=np.uint8)
    integrality[binary] = 1
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_row(coefficients: dict[int, float], lo: float, hi: float) -> None:
        row = len(constraint_lower)
        for column, value in coefficients.items():
            rows.append(row)
            cols.append(int(column))
            values.append(float(value))
        constraint_lower.append(float(lo))
        constraint_upper.append(float(hi))
    for s in range(s_num):
        for t in range(h):
            net_load = ev_s[s, t] - pv_s[s, t]
            add_row({grid[s, t]: 1, discharge[s, t]: 1, charge[s, t]: -1, curt[s, t]: -1}, net_load, net_load)
            soc_coefficients = {soc[s, t]: 1, charge[s, t]: -battery.eta_charge * dt / battery.energy_kwh, discharge[s, t]: dt / (battery.eta_discharge * battery.energy_kwh)}
            if t == 0:
                add_row(soc_coefficients, current_soc, current_soc)
            else:
                soc_coefficients[soc[s, t - 1]] = -1
                add_row(soc_coefficients, 0, 0)
            add_row({grid[s, t]: 1, peak[s]: -1}, -np.inf, 0)
            add_row({charge[s, t]: 1, binary[s, t]: -battery.charge_limit_kw}, -np.inf, 0)
            add_row({discharge[s, t]: 1, binary[s, t]: battery.discharge_limit_kw}, -np.inf, battery.discharge_limit_kw)
    # EN: Non-anticipativity at t=0: force identical step-0 charge, discharge, and binary mode across all scenarios before actuals arrive.
    # 中文：首步非预期性耦合（t=0）：在观测当步实际负荷与光伏前，强制三场景首步充电、放电与互斥状态完全一致。
    for s in range(1, s_num):
        add_row({charge[s, 0]: 1, charge[0, 0]: -1}, 0, 0)
        add_row({discharge[s, 0]: 1, discharge[0, 0]: -1}, 0, 0)
        add_row({binary[s, 0]: 1, binary[0, 0]: -1}, 0, 0)
    # EN: Deliverable reserve floor & discrete barrier: protect h_R * max(EV_up - PV_lo - P_safe, 0) above SOC_min.
    # 中文：可交付备用下限与离散控制屏障：在 SOC_min 之上锁定最不利馈线超额缺口备用 h_R * max(EV_up - PV_lo - P_safe, 0)。
    if controller != ControllerKind.M6 and reserve_slack is not None:
        adverse_net = np.maximum(data.ev_upper_kw - data.pv_lower_kw, 0.0)
        if controller == ControllerKind.GS:
            d_wc = np.maximum(adverse_net - cfg.feeder_safe_kw, 0.0)
        else:
            d_wc = adverse_net
        eta_energy = battery.eta_discharge * battery.energy_kwh
        for s in range(s_num):
            for t in range(h):
                reserve_rhs = eta_energy * battery.soc_min + cfg.reserve_hours * d_wc[t]
                add_row({soc[s, t]: eta_energy, reserve_slack[s, t]: 1}, reserve_rhs, np.inf)
                if t == 0:
                    previous_margin = eta_energy * (current_soc - battery.soc_min) - cfg.reserve_hours * d_wc[t]
                    barrier_rhs = reserve_rhs + (1 - cfg.barrier_alpha) * previous_margin
                    add_row({soc[s, t]: eta_energy, reserve_slack[s, t]: 1}, barrier_rhs, np.inf)
                else:
                    add_row({soc[s, t]: eta_energy, soc[s, t - 1]: -(1 - cfg.barrier_alpha) * eta_energy, reserve_slack[s, t]: 1}, reserve_rhs - (1 - cfg.barrier_alpha) * (eta_energy * battery.soc_min + cfg.reserve_hours * d_wc[t - 1]), np.inf)
    if controller == ControllerKind.GS:
        assert swing is not None and deviation is not None and (grid_slack is not None)
        for s in range(s_num):
            for t in range(h):
                add_row({grid[s, t]: 1, grid_slack[s, t]: -1}, -np.inf, cfg.feeder_safe_kw)
                previous_soc_index = soc[s, t - 1] if t else None
                if previous_soc_index is None:
                    add_row({swing[s, t]: 1, soc[s, t]: -1}, -current_soc, np.inf)
                    add_row({swing[s, t]: 1, soc[s, t]: 1}, current_soc, np.inf)
                else:
                    add_row({swing[s, t]: 1, soc[s, t]: -1, previous_soc_index: 1}, 0, np.inf)
                    add_row({swing[s, t]: 1, soc[s, t]: 1, previous_soc_index: -1}, 0, np.inf)
                add_row({deviation[s, t]: 1, soc[s, t]: -1}, -battery.soc_reference, np.inf)
                add_row({deviation[s, t]: 1, soc[s, t]: 1}, battery.soc_reference, np.inf)
    terminal_lower = min(battery.soc_initial, current_soc)
    for s in range(s_num):
        add_row({soc[s, h - 1]: 1}, terminal_lower, battery.soc_max)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(constraint_lower), n_var)).tocsr()
    constraints = LinearConstraint(matrix, constraint_lower, constraint_upper)
    presolve_policy = resolve_presolve_policy(requested=cfg.solver.presolve, step_hours=cfg.step_hours, current_soc=current_soc, soc_max=battery.soc_max)
    options = {'time_limit': cfg.solver.time_limit_seconds, 'mip_rel_gap': cfg.solver.mip_relative_gap, 'mip_feasibility_tolerance': cfg.solver.mip_feasibility_tolerance, 'presolve': presolve_policy.effective, 'disp': cfg.solver.display, **highs_determinism_options(random_seed=cfg.solver.random_seed, threads=cfg.solver.threads, parallel=cfg.solver.parallel)}
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected: .*', category=RuntimeWarning)
        result = call_external_milp(lambda: milp(c=objective, integrality=integrality, bounds=Bounds(lower_bounds, upper_bounds), constraints=constraints, options=options), method=method, controller=controller)
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None:
        raise SolveFailure(f'MILP failed (status={result.status}): {result.message}')
    nominal = 1
    x = result.x
    tolerance = 1e-09
    acceptance_tolerance = 1e-07
    activity = np.asarray(matrix @ x, dtype=float).reshape(-1)
    row_lower = np.asarray(constraint_lower, dtype=float)
    row_upper = np.asarray(constraint_upper, dtype=float)
    lower_row_violation = np.maximum(row_lower - activity, 0.0)
    upper_row_violation = np.maximum(activity - row_upper, 0.0)
    max_constraint_violation = float(max(np.max(lower_row_violation, initial=0.0), np.max(upper_row_violation, initial=0.0)))
    max_bound_violation = float(max(np.max(np.maximum(lower_bounds - x, 0.0), initial=0.0), np.max(np.maximum(x - upper_bounds, 0.0), initial=0.0)))
    max_integrality_violation = float(np.max(np.abs(x[binary] - np.rint(x[binary])), initial=0.0))
    first_stage_coupling_residual = float(max(max((abs(x[charge[s, 0]] - x[charge[0, 0]]) for s in range(1, s_num))), max((abs(x[discharge[s, 0]] - x[discharge[0, 0]]) for s in range(1, s_num))), max((abs(x[binary[s, 0]] - x[binary[0, 0]]) for s in range(1, s_num)))))
    first_step_power_balance_residual = float(max((abs(x[grid[s, 0]] + x[discharge[s, 0]] - x[charge[s, 0]] - x[curt[s, 0]] - (ev_s[s, 0] - pv_s[s, 0])) for s in range(s_num))))
    first_step_soc_dynamics_residual = float(max((abs(x[soc[s, 0]] - battery.eta_charge * dt * x[charge[s, 0]] / battery.energy_kwh + dt * x[discharge[s, 0]] / (battery.eta_discharge * battery.energy_kwh) - current_soc) for s in range(s_num))))
    residuals = {'constraint': max_constraint_violation, 'bound': max_bound_violation, 'integrality': max_integrality_violation, 'first-stage coupling': first_stage_coupling_residual, 'first-step power balance': first_step_power_balance_residual, 'first-step SOC dynamics': first_step_soc_dynamics_residual}
    failed_residuals = {name: value for name, value in residuals.items() if value > acceptance_tolerance}
    if failed_residuals:
        rendered = ', '.join((f'{name}={value:.12g}' for name, value in failed_residuals.items()))
        raise SolveFailure(f'independent MILP residual check exceeds {acceptance_tolerance:.1e}: {rendered}')
    reported_gap = getattr(result, 'mip_gap', None)
    if reported_gap is not None and float(reported_gap) > cfg.solver.mip_relative_gap + 1e-12:
        raise SolveFailure(f'reported MIP gap exceeds configured acceptance threshold: gap={float(reported_gap):.12g}, limit={cfg.solver.mip_relative_gap:.12g}')

    def bounded(value: float, lo: float, hi: float, name: str) -> float:
        value = float(value)
        if value < lo - tolerance or value > hi + tolerance:
            raise SolveFailure(f'solver returned out-of-bound {name}={value}')
        return min(max(value, lo), hi)
    return FirstStepDecision(method=method, controller=controller, charge_kw=bounded(x[charge[nominal, 0]], 0.0, battery.charge_limit_kw, 'charge'), discharge_kw=bounded(x[discharge[nominal, 0]], 0.0, battery.discharge_limit_kw, 'discharge'), planned_grid_kw=bounded(x[grid[nominal, 0]], 0.0, np.inf, 'grid'), planned_curtail_kw=bounded(x[curt[nominal, 0]], 0.0, pv_s[nominal, 0], 'curtailment'), planned_soc=bounded(x[soc[nominal, 0]], battery.soc_min, battery.soc_max, 'SOC'), reserve_slack_kwh=max((bounded(x[reserve_slack[s, 0]], 0.0, np.inf, 'reserve slack') for s in range(s_num))) if reserve_slack is not None else 0.0, grid_slack_kw=max((bounded(x[grid_slack[s, 0]], 0.0, np.inf, 'grid slack') for s in range(s_num))) if grid_slack is not None else 0.0, objective=float(result.fun), solver_status=int(result.status), solver_message=str(result.message), solver_seconds=float(elapsed), mip_gap=float(result.mip_gap) if getattr(result, 'mip_gap', None) is not None else None, mip_node_count=int(result.mip_node_count) if getattr(result, 'mip_node_count', None) is not None else None, max_constraint_violation=max_constraint_violation, max_bound_violation=max_bound_violation, max_integrality_violation=max_integrality_violation, first_stage_coupling_residual=first_stage_coupling_residual, first_step_power_balance_residual=first_step_power_balance_residual, first_step_soc_dynamics_residual=first_step_soc_dynamics_residual, presolve_policy_version=presolve_policy.version, presolve_requested=presolve_policy.requested, presolve_effective=presolve_policy.effective, presolve_guard_applied=presolve_policy.guard_applied, presolve_guard_reason=presolve_policy.guard_reason, presolve_guard_soc_atol=presolve_policy.soc_atol)
