from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from station_core.config import ControllerKind, ExperimentConfig, Method, method_spec, weights_for
from station_core.solver_determinism import canonicalize_solver_soc
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

def emit_station_gs_block(method: Method | str, horizon: HorizonData, current_soc: float, config: ExperimentConfig | None=None, *, first_step_hours: float=1.0, reserve_horizon: HorizonData | None=None) -> dict:
    cfg = config or ExperimentConfig()
    method = Method(method)
    spec = method_spec(method)
    controller = spec.controller
    if controller != ControllerKind.GS:
        raise ValueError('This joint baseline retains the GS block only')
    data = horizon.validated()
    battery = cfg.battery
    if not battery.soc_min <= current_soc <= battery.soc_max:
        raise ValueError('current SOC lies outside configured bounds')
    current_soc = canonicalize_solver_soc(current_soc, soc_min=battery.soc_min, soc_max=battery.soc_max, decimals=cfg.solver.soc_canonical_decimals)
    h = data.horizon
    s_num = 3
    probs = np.asarray(cfg.scenario_probabilities, dtype=float)
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
    if not 0 < first_step_hours <= cfg.step_hours:
        raise ValueError('Positive remaining first-hour duration')
    dts = np.full(h, cfg.step_hours)
    dts[0] = first_step_hours
    dt = dts
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
            dt = dts[t]
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
    for s in range(1, s_num):
        add_row({charge[s, 0]: 1, charge[0, 0]: -1}, 0, 0)
        add_row({discharge[s, 0]: 1, discharge[0, 0]: -1}, 0, 0)
        add_row({binary[s, 0]: 1, binary[0, 0]: -1}, 0, 0)
    if controller != ControllerKind.M6 and reserve_slack is not None:
        reserve_data = data if reserve_horizon is None else reserve_horizon.validated()
        adverse_net = np.maximum(reserve_data.ev_upper_kw - reserve_data.pv_lower_kw, 0.0)
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
    return dict(c=objective, lower=lower_bounds, upper=upper_bounds, integrality=integrality, coo=dict(row=rows, col=cols, data=values, shape=[len(constraint_lower), n_var]), row_lower=constraint_lower, row_upper=constraint_upper, indices=ind.fields, issue_soc_canonical=current_soc, terminal_floor=terminal_lower, ev_s=ev_s, pv_s=pv_s, config=cfg, horizon=data, method=method, original_constructor_only=True)
