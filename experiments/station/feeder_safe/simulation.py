# EN: In-memory closed-loop rolling simulation loop coupling causal forecasts, MILP nominal dispatch, and REF execution.
# 中文：耦合因果预测、MILP 名义调度与 REF 实时修正的内存态滚动闭环仿真主循环。

from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from .config import ControllerKind, ExperimentConfig, Method, method_spec
from .conformal import AdaptiveConformal, AsymmetricConformal, CalibrationPressureScaler
from .filter import PhysicalConstraintViolation, advance_soc_checked, apply_execution_filter
from .forecast import CausalPersistenceForecaster
from .model import HorizonData, solve_first_step

@dataclass(frozen=True)
class SimulationInputs:
    ev_actual_kw: np.ndarray
    pv_actual_kw: np.ndarray
    price_yuan_per_kwh: np.ndarray
    timestamps: np.ndarray | None = None

    def validated(self) -> 'SimulationInputs':
        ev = np.asarray(self.ev_actual_kw, dtype=float).reshape(-1)
        pv = np.asarray(self.pv_actual_kw, dtype=float).reshape(-1)
        price = np.asarray(self.price_yuan_per_kwh, dtype=float).reshape(-1)
        if not (ev.size == pv.size == price.size and ev.size > 0):
            raise ValueError('simulation inputs must be nonempty and aligned')
        if not all((np.all(np.isfinite(x)) for x in (ev, pv, price))):
            raise ValueError('simulation inputs contain non-finite values')
        timestamps = np.arange(ev.size).astype(str) if self.timestamps is None else np.asarray(self.timestamps).reshape(-1).astype(str)
        if timestamps.size != ev.size:
            raise ValueError('timestamps must align with simulation inputs')
        return SimulationInputs(ev, pv, price, timestamps)

@dataclass(frozen=True)
class StepTrace:
    timestamp: str
    step: int
    run_id: str
    condition_id: str
    arm_id: str
    scheduler_id: str
    filter_id: str
    calibration_id: str
    soc_pre: float
    soc_post: float
    p_charge_nom_kw: float
    p_discharge_nom_kw: float
    net_exchange_nom_kw: float
    grid_import_nom_kw: float
    delta_charge_kw: float
    delta_discharge_kw: float
    p_charge_exec_kw: float
    p_discharge_exec_kw: float
    net_exchange_exec_kw: float
    grid_import_exec_kw: float
    reserve_requirement_kwh: float
    reserve_headroom_kwh: float
    reserve_shortage_kwh: float
    barrier_slack_kwh: float
    feeder_slack_kw: float
    solver_status: int
    mip_gap: float | None
    solve_time_s: float
    objective: float
    certificate_applicable: bool
STEP_TRACE_COLUMNS = tuple(StepTrace.__dataclass_fields__)

def traces_to_records(traces: list[StepTrace]) -> list[dict[str, object]]:
    records = [asdict(trace) for trace in traces]
    if any((tuple(record) != STEP_TRACE_COLUMNS for record in records)):
        raise RuntimeError('step trace schema/order drift detected')
    return records

def run_simulation(method: Method | str, inputs: SimulationInputs, ev_forecaster: CausalPersistenceForecaster, pv_forecaster: CausalPersistenceForecaster, ev_conformal: AsymmetricConformal, pv_conformal: AsymmetricConformal, pressure_scaler: CalibrationPressureScaler, start: int, stop: int | None=None, config: ExperimentConfig | None=None, *, run_id: str='run', condition_id: str='default', arm_id: str | None=None, calibration_id: str | None=None) -> list[StepTrace]:
    cfg = config or ExperimentConfig()
    method = Method(method)
    spec = method_spec(method)
    data = inputs.validated()
    stop = len(data.ev_actual_kw) if stop is None else min(stop, len(data.ev_actual_kw))
    if start < max(ev_forecaster.lag, pv_forecaster.lag) or not start < stop:
        raise ValueError('start must follow forecast warm-up and precede stop')
    if spec.adaptive_conformal and (not isinstance(ev_conformal, AdaptiveConformal)):
        raise TypeError('GS_F1_ACI requires AdaptiveConformal for EV')
    if spec.adaptive_conformal and (not isinstance(pv_conformal, AdaptiveConformal)):
        raise TypeError('GS_F1_ACI requires AdaptiveConformal for PV')
    arm_id = arm_id or method.value
    calibration_id = calibration_id or ('ACI' if spec.adaptive_conformal else 'static')
    soc = cfg.battery.soc_initial
    traces: list[StepTrace] = []
    for step in range(start, stop):
        horizon_steps = min(cfg.horizon_steps, stop - step)
        ev_point = np.maximum(ev_forecaster.forecast(data.ev_actual_kw[:step], horizon_steps), 0.0)
        pv_point = np.clip(pv_forecaster.forecast(data.pv_actual_kw[:step], horizon_steps), 0.0, cfg.pv_rated_kw)
        ev_frozen = ev_conformal.freeze_horizon(ev_point, step)
        pv_frozen = pv_conformal.freeze_horizon(pv_point, step)
        ev_lower = np.maximum(ev_frozen.lower, 0.0)
        ev_upper = np.maximum(ev_frozen.upper, ev_lower)
        pv_lower = np.clip(pv_frozen.lower, 0.0, cfg.pv_rated_kw)
        pv_upper = np.clip(np.maximum(pv_frozen.upper, pv_lower), 0.0, cfg.pv_rated_kw)
        adverse_net = np.maximum(ev_upper - pv_lower, 0.0)
        pressure = pressure_scaler.transform(adverse_net)
        decision = solve_first_step(method, HorizonData(ev_lower, ev_point, ev_upper, pv_lower, pv_point, pv_upper, data.price_yuan_per_kwh[step:step + horizon_steps], pressure), soc, cfg)
        if spec.controller == ControllerKind.RB:
            reserve_requirement = cfg.reserve_hours * adverse_net[0]
        elif spec.controller == ControllerKind.GS:
            reserve_requirement = cfg.reserve_hours * max(adverse_net[0] - cfg.feeder_safe_kw, 0.0)
        else:
            reserve_requirement = 0.0
        try:
            filtered = apply_execution_filter(spec.execution_filter, decision.charge_kw, decision.discharge_kw, data.ev_actual_kw[step], data.pv_actual_kw[step], soc, cfg.battery, cfg.feeder_safe_kw, cfg.step_hours, float(reserve_requirement))
            soc_after = advance_soc_checked(soc, filtered.charge_kw, filtered.discharge_kw, cfg.battery, cfg.step_hours)
        except PhysicalConstraintViolation as exc:
            details = dict(exc.details)
            details.update(step=float(step), soc_pre=float(soc))
            raise PhysicalConstraintViolation('simulation action rejected before trace emission', **details) from exc
        reserve_margin = cfg.battery.eta_discharge * cfg.battery.energy_kwh * (soc_after - cfg.battery.soc_min) - reserve_requirement
        net_nominal = data.ev_actual_kw[step] + decision.charge_kw - data.pv_actual_kw[step] - decision.discharge_kw
        traces.append(StepTrace(timestamp=str(data.timestamps[step]), step=step, run_id=run_id, condition_id=condition_id, arm_id=arm_id, scheduler_id=spec.controller.value, filter_id=spec.execution_filter.value, calibration_id=calibration_id, soc_pre=float(soc), soc_post=float(soc_after), p_charge_nom_kw=decision.charge_kw, p_discharge_nom_kw=decision.discharge_kw, net_exchange_nom_kw=float(net_nominal), grid_import_nom_kw=float(max(net_nominal, 0.0)), delta_charge_kw=float(filtered.charge_reduction_kw), delta_discharge_kw=float(filtered.extra_discharge_kw), p_charge_exec_kw=filtered.charge_kw, p_discharge_exec_kw=filtered.discharge_kw, net_exchange_exec_kw=filtered.final_net_exchange_kw, grid_import_exec_kw=filtered.final_grid_kw, reserve_requirement_kwh=float(reserve_requirement), reserve_headroom_kwh=float(max(reserve_margin, 0.0)), reserve_shortage_kwh=float(max(-reserve_margin, 0.0)), barrier_slack_kwh=decision.reserve_slack_kwh, feeder_slack_kw=decision.grid_slack_kw, solver_status=decision.solver_status, mip_gap=decision.mip_gap, solve_time_s=decision.solver_seconds, objective=decision.objective, certificate_applicable=spec.controller == ControllerKind.GS and decision.grid_slack_kw <= 1e-07 and filtered.certificate_applicable))
        ev_conformal.observe(data.ev_actual_kw[step], ev_point[0], ev_frozen)
        pv_conformal.observe(data.pv_actual_kw[step], pv_point[0], pv_frozen)
        soc = soc_after
    return traces
