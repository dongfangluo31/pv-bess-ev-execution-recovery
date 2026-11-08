# EN: End-to-end closed-loop station simulation runner emitting step traces, shadow-filter counterfactuals, and solver diagnostics.
# 中文：端到端场站闭环仿真运行器，输出步级轨迹、同状态影子修正对照与求解器残差诊断。

from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Mapping
from uuid import uuid4
import numpy as np
import pandas as pd
from .config import BatteryConfig, ControllerKind, ExperimentConfig as CoreExperimentConfig, FilterKind, Method, method_spec
from .conformal import AdaptiveConformal, AsymmetricConformal, CalibrationPressureScaler, FrozenIntervals, finite_sample_rank
from .data_pipeline import DEFAULT_EV_PATH, DEFAULT_PV_METADATA_PATH, DEFAULT_PV_PATH, PV_CALENDAR_MAPPING, PreparedDataset, SOURCE_TIMEZONE, TARGET_TIMEZONE, map_pvod_hourly, prices_for_timestamps, verify_prepared_dataset
from .filter import advance_soc_checked, apply_execution_filter
from .forecast import CausalPersistenceForecaster
from .plant import PLANT_REALIZATION
from .metrics import PairingAuditResult, audit_pairing
from .matrix import RunSpec
from .model import PRESOLVE_POLICY_VERSION, FirstStepDecision, HorizonData, solve_first_step
from .provenance import DEFAULT_RUN_ROOT, RunDirectory, atomic_write_json, canonical_json_sha256, ensure_within_output_root, file_manifest_rows, make_run_id, sha256_file, write_manifest_csv
from .simulation import STEP_TRACE_COLUMNS, StepTrace, traces_to_records
from .solver_attempts import SolverAttemptLogger, solver_attempt_scope
RUNNER_SCHEMA_VERSION = 2
FILTER_SEMANTICS_SCHEMA_VERSION = 2
RESIDUAL_TOLERANCE = 1e-07
PAIRING_TOLERANCE = 1e-09
SUPPORTED_RUN_KINDS = frozenset({'single_station_scheduler', 'single_station_15min_scheduler', 'single_station_aci_sensitivity', 'capacity_scheduler', 'clean_process_replay', 'profile_scheduler', 'seven_station_coupled_scheduler', 'closed_loop_delayed_filter', 'same_scheduler_filter_ablation'})
UNSUPPORTED_RUN_KINDS = frozenset({'network_replay_postprocess'})

class RunContractError(ValueError):
    pass

class RunExecutionError(RuntimeError):

    def __init__(self, message: str, *, run_id: str, run_root: Path) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.run_root = run_root

@dataclass(frozen=True)
class ExperimentConfig:
    prepared: PreparedDataset
    arm_id: str
    split_id: str
    condition_id: str = 'BASE'
    stage: str = 'AD_HOC'
    run_kind: str = 'single_station_scheduler'
    spec_id: str | None = None
    matrix_config_sha256: str | None = None
    parameters_json: str = '{}'
    formal_evidence: bool = False
    core: CoreExperimentConfig = field(default_factory=CoreExperimentConfig)
    step_minutes: int = 60
    forecast_lag: int = 24
    aci_gamma: float = 0.02
    aci_alpha_min: float = 0.005
    aci_alpha_max: float = 0.3
    aci_rolling_window: int | None = None
    max_steps: int | None = None
    dry_run: bool = False
    run_id: str | None = None
    run_root: Path = DEFAULT_RUN_ROOT
    residual_tolerance: float = RESIDUAL_TOLERANCE
    pairing_tolerance: float = PAIRING_TOLERANCE
    paired_f0_trace: Path | None = None
    paired_f0_input_sha256: str | None = None
    require_pair_activation: bool = False
    precalibration_history: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            method = Method(self.arm_id)
        except ValueError as exc:
            raise RunContractError(f'unsupported arm_id {self.arm_id!r}') from exc
        if self.split_id != str(self.prepared.manifest.get('split_id')):
            raise RunContractError('config split_id does not match prepared dataset')
        if self.forecast_lag < 1:
            raise RunContractError('forecast_lag must be positive')
        if self.max_steps is not None and self.max_steps < 1:
            raise RunContractError('max_steps must be positive when supplied')
        if self.residual_tolerance <= 0 or self.pairing_tolerance < 0:
            raise RunContractError('residual/pairing tolerances are invalid')
        if not 0 < self.aci_alpha_min < self.aci_alpha_max < 1:
            raise RunContractError('ACI clip bounds are invalid')
        if self.aci_gamma <= 0:
            raise RunContractError('ACI gamma must be positive')
        if method_spec(method).adaptive_conformal and self.core.calibration_window < 1:
            raise RunContractError('ACI requires a nonempty calibration window')
        if self.run_kind not in SUPPORTED_RUN_KINDS:
            known = sorted(SUPPORTED_RUN_KINDS | UNSUPPORTED_RUN_KINDS)
            raise RunContractError(f'unsupported run_kind {self.run_kind!r}; known run kinds: {known}')
        if self.step_minutes not in {15, 60}:
            raise RunContractError('runner currently supports only 15- or 60-minute steps')
        if not math.isclose(self.core.step_hours, self.step_minutes / 60.0, abs_tol=1e-12):
            raise RunContractError('core.step_hours does not match step_minutes')
        try:
            parameters = json.loads(self.parameters_json)
        except json.JSONDecodeError as exc:
            raise RunContractError('parameters_json is not valid JSON') from exc
        if not isinstance(parameters, dict):
            raise RunContractError('parameters_json must encode one object')
        if self.formal_evidence and (self.spec_id is None or self.matrix_config_sha256 is None):
            raise RunContractError('formal_evidence can only originate from a hash-bound RunSpec')

    @property
    def effective_formal_evidence(self) -> bool:
        return bool(self.formal_evidence and self.max_steps is None and (not self.dry_run))

@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    state: str
    run_root: Path | None
    step_count: int
    config_sha256: str
    input_sha256: str
    protocol_report: Mapping[str, object]
    formal_evidence: bool = False
    step_trace: Path | None = None
    step_table: Path | None = None
    forecast_trace: Path | None = None
    shadow_filter_trace: Path | None = None
    solver_diagnostics: Path | None = None
    pairing_audit: Path | None = None
    completion_marker: Path | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return value

@dataclass(frozen=True)
class PairedRunArtifacts:
    f0: RunArtifacts
    f1: RunArtifacts
    audit: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {'f0': self.f0.to_dict(), 'f1': self.f1.to_dict(), 'audit': dict(self.audit)}

@dataclass(frozen=True)
class BoundsSnapshot:
    decision_step: int
    ev: FrozenIntervals
    pv: FrozenIntervals

class CalibrationProvider:

    def __init__(self, ev: AsymmetricConformal, pv: AsymmetricConformal) -> None:
        self.ev = ev
        self.pv = pv

    def bounds(self, decision_time: int, ev_point: np.ndarray, pv_point: np.ndarray) -> BoundsSnapshot:
        return BoundsSnapshot(decision_step=int(decision_time), ev=self.ev.freeze_horizon(ev_point, decision_time), pv=self.pv.freeze_horizon(pv_point, decision_time))

    def observe(self, ev_actual: float, pv_actual: float, snapshot: BoundsSnapshot) -> None:
        self.ev.observe(ev_actual, float(snapshot.ev.point[0]), snapshot.ev)
        self.pv.observe(pv_actual, float(snapshot.pv.point[0]), snapshot.pv)

def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=' ')
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

def _number(parameters: Mapping[str, object], key: str) -> float:
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunContractError(f'RunSpec parameter {key!r} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise RunContractError(f'RunSpec parameter {key!r} must be finite')
    return result

def config_from_run_spec(spec: RunSpec, prepared: PreparedDataset, *, run_id: str | None=None, run_root: Path=DEFAULT_RUN_ROOT, dry_run: bool=False, max_steps: int | None=None) -> Any:
    if spec.schema_version != 2:
        raise RunContractError(f'runner requires RunSpec schema v2, got {spec.schema_version}')
    payload = spec.to_dict()
    claimed_hash = str(payload.pop('config_sha256'))
    payload.pop('spec_id')
    if canonical_json_sha256(payload) != claimed_hash:
        raise RunContractError(f'RunSpec {spec.spec_id} config SHA-256 does not verify')
    if spec.run_kind not in SUPPORTED_RUN_KINDS:
        raise RunContractError(f'unsupported run_kind {spec.run_kind!r}; supported={sorted(SUPPORTED_RUN_KINDS)}; registered-but-unsupported={sorted(UNSUPPORTED_RUN_KINDS)}')
    try:
        parameters = json.loads(spec.parameters_json)
    except json.JSONDecodeError as exc:
        raise RunContractError('RunSpec parameters_json is invalid') from exc
    if not isinstance(parameters, dict):
        raise RunContractError('RunSpec parameters_json must encode an object')
    allowed_keys = {'single_station_scheduler': {'horizon_hours', 'reserve_hours'}, 'single_station_15min_scheduler': {'horizon_steps', 'ev_resampling', 'pv_resampling', 'price_resampling'}, 'single_station_aci_sensitivity': {'gamma', 'history', 'cal_size_hours', 'tau_per_side', 'predictor'}, 'capacity_scheduler': {'battery_capacity_kwh'}, 'clean_process_replay': {'replicate_index'}}[spec.run_kind]
    unknown = set(parameters).difference(allowed_keys)
    if unknown:
        raise RunContractError(f'RunSpec {spec.spec_id} has unmapped parameters: {sorted(unknown)}')
    required_keys = {'single_station_scheduler': set(), 'single_station_15min_scheduler': allowed_keys, 'single_station_aci_sensitivity': allowed_keys, 'capacity_scheduler': allowed_keys, 'clean_process_replay': allowed_keys}[spec.run_kind]
    missing = required_keys.difference(parameters)
    if missing:
        raise RunContractError(f'RunSpec {spec.spec_id} lacks required parameters: {sorted(missing)}')
    try:
        method = Method(spec.arm_id)
    except ValueError as exc:
        raise RunContractError(f'run_kind {spec.run_kind!r} uses unsupported arm {spec.arm_id!r}') from exc
    resolved_method = method_spec(method)
    expected_scheduler = {ControllerKind.M6: 'M6_CONFORMAL_SCENARIO', ControllerKind.RB: 'RESERVE_BARRIER', ControllerKind.GS: 'GRID_SAFE'}[resolved_method.controller]
    expected_filter = 'REF' if resolved_method.execution_filter == FilterKind.F1_RESERVE else 'NONE'
    expected_calibration = 'PAST_ONLY_ACI' if resolved_method.adaptive_conformal else 'STATIC'
    if spec.scheduler_id != expected_scheduler or spec.filter_id != expected_filter or spec.calibration_id != expected_calibration:
        raise RunContractError('RunSpec scheduler/filter/calibration identity does not match the executable arm')
    try:
        required_outputs = json.loads(spec.required_outputs_json)
    except json.JSONDecodeError as exc:
        raise RunContractError('RunSpec required_outputs_json is invalid') from exc
    if required_outputs:
        raise RunContractError(f'supported runner cannot silently ignore required outputs: {required_outputs}')
    if spec.reuse_dependency:
        raise RunContractError('fresh single-station runs cannot declare a reuse dependency')
    step_minutes = int(spec.step_minutes)
    if step_minutes not in {15, 60}:
        raise RunContractError(f'unsupported RunSpec step_minutes={step_minutes}')
    if spec.run_kind == 'single_station_15min_scheduler' and step_minutes != 15:
        raise RunContractError('15-minute run kind must declare step_minutes=15')
    if spec.run_kind != 'single_station_15min_scheduler' and step_minutes != 60:
        raise RunContractError(f'{spec.run_kind} must currently use 60-minute steps')
    factor = 60 // step_minutes
    calibration_hours = int(prepared.manifest.get('row_counts', {}).get('calibration', 0))
    test_hours = int(prepared.manifest.get('row_counts', {}).get('test', 0))
    if calibration_hours < 1 or test_hours < 1:
        raise RunContractError('prepared manifest lacks calibration/test row counts')
    expected_steps = test_hours * factor
    if spec.expected_test_steps != expected_steps:
        raise RunContractError(f'RunSpec expected_test_steps={spec.expected_test_steps}, prepared data imply {expected_steps}')
    if spec.expected_trace_rows != expected_steps or spec.expected_milp_solves != expected_steps or spec.reused_milp_solves != 0 or (spec.expected_power_flow_calls != 0):
        raise RunContractError('supported scheduler run kinds require one trace row and one fresh MILP solve per step')
    horizon_steps = 24 * factor
    reserve_hours = 1.0
    forecast_hours = 24
    calibration_size_hours = calibration_hours
    miscoverage = 0.1
    gamma = 0.02
    rolling_window: int | None = None
    battery = BatteryConfig()
    if 'horizon_hours' in parameters:
        horizon_hours = _number(parameters, 'horizon_hours')
        raw_steps = horizon_hours * factor
        if horizon_hours <= 0 or not raw_steps.is_integer():
            raise RunContractError('horizon_hours does not map to an integer control horizon')
        horizon_steps = int(raw_steps)
    if 'reserve_hours' in parameters:
        reserve_hours = _number(parameters, 'reserve_hours')
        if reserve_hours < 0:
            raise RunContractError('reserve_hours must be nonnegative')
    if spec.run_kind == 'single_station_15min_scheduler':
        if parameters['ev_resampling'] != 'hourly_power_held_for_four_steps':
            raise RunContractError('unsupported 15-minute EV resampling declaration')
        if parameters['pv_resampling'] != 'native_15min_calendar_proxy':
            raise RunContractError('15-minute PV must use the native source mapping')
        if parameters['price_resampling'] != 'hourly_price_held_for_four_steps':
            raise RunContractError('unsupported 15-minute price resampling declaration')
        horizon_value = _number(parameters, 'horizon_steps')
        if horizon_value <= 0 or not horizon_value.is_integer():
            raise RunContractError('horizon_steps must be a positive integer')
        horizon_steps = int(horizon_value)
    if spec.run_kind == 'single_station_aci_sensitivity':
        if not method_spec(method).adaptive_conformal:
            raise RunContractError('ACI sensitivity run must use the adaptive-conformal arm')
        gamma = _number(parameters, 'gamma')
        if gamma <= 0:
            raise RunContractError('ACI gamma must be positive')
        calibration_size_hours = int(_number(parameters, 'cal_size_hours'))
        if calibration_size_hours < 1 or calibration_size_hours > calibration_hours:
            raise RunContractError('cal_size_hours exceeds the past-only calibration data available')
        tau = _number(parameters, 'tau_per_side')
        if not 0 < tau < 0.5:
            raise RunContractError('tau_per_side must lie in (0, 0.5)')
        miscoverage = 2.0 * tau
        predictor = str(parameters['predictor'])
        if predictor not in {'P24', 'P168'}:
            raise RunContractError(f'unsupported predictor {predictor!r}')
        forecast_hours = int(predictor[1:])
        history = str(parameters['history'])
        if history == 'expanding':
            rolling_window = None
        elif history in {'rolling_168', 'rolling_720', 'rolling_1440'}:
            rolling_window = int(history.split('_', maxsplit=1)[1]) * factor
        else:
            raise RunContractError(f'unsupported ACI history mode {history!r}')
    if spec.run_kind == 'capacity_scheduler':
        capacity = _number(parameters, 'battery_capacity_kwh')
        if capacity <= 0:
            raise RunContractError('battery_capacity_kwh must be positive')
        battery = replace(battery, energy_kwh=capacity)
    if spec.run_kind == 'clean_process_replay':
        replicate = _number(parameters, 'replicate_index')
        if not replicate.is_integer() or int(replicate) not in {1, 2, 3}:
            raise RunContractError('replicate_index must be one of 1, 2, 3')
    core = replace(CoreExperimentConfig(), battery=battery, horizon_steps=horizon_steps, step_hours=step_minutes / 60.0, reserve_hours=reserve_hours, conformal_miscoverage=miscoverage, calibration_window=calibration_size_hours * factor, seed=spec.verification_seed)
    forecast_lag = forecast_hours * factor
    return ExperimentConfig(prepared=prepared, arm_id=spec.arm_id, split_id=spec.split_id, condition_id=spec.condition_id, stage=spec.stage, run_kind=spec.run_kind, spec_id=spec.spec_id, matrix_config_sha256=spec.config_sha256, parameters_json=spec.parameters_json, formal_evidence=spec.formal_evidence, core=core, step_minutes=step_minutes, forecast_lag=forecast_lag, aci_gamma=gamma, aci_rolling_window=rolling_window, max_steps=max_steps, dry_run=dry_run, run_id=run_id, run_root=run_root)

def _configuration_payload(config: ExperimentConfig) -> dict[str, object]:
    expected_test_steps = int(config.prepared.inputs['split_role'].eq('test').sum())
    execution_expected_milp_solves = expected_test_steps if config.max_steps is None else min(expected_test_steps, config.max_steps)
    prepared_identity = {'manifest': config.prepared.manifest, 'input_rows': int(len(config.prepared.inputs)), 'pv_provenance_rows': int(len(config.prepared.pv_provenance))}
    return _jsonable({'runner_schema_version': RUNNER_SCHEMA_VERSION, 'arm_id': config.arm_id, 'split_id': config.split_id, 'condition_id': config.condition_id, 'stage': config.stage, 'run_kind': config.run_kind, 'spec_id': config.spec_id, 'matrix_config_sha256': config.matrix_config_sha256, 'parameters_json': config.parameters_json, 'declared_formal_evidence': config.formal_evidence, 'effective_formal_evidence': config.effective_formal_evidence, 'core': asdict(config.core), 'step_minutes': config.step_minutes, 'forecast_lag': config.forecast_lag, 'aci_gamma': config.aci_gamma, 'aci_alpha_min': config.aci_alpha_min, 'aci_alpha_max': config.aci_alpha_max, 'aci_rolling_window': config.aci_rolling_window, 'max_steps': config.max_steps, 'dry_run': config.dry_run, 'expected_test_steps': expected_test_steps, 'expected_trace_rows': expected_test_steps, 'expected_milp_solves': expected_test_steps, 'execution_expected_milp_solves': execution_expected_milp_solves, 'residual_tolerance': config.residual_tolerance, 'pairing_tolerance': config.pairing_tolerance, 'paired_f0_trace': config.paired_f0_trace, 'require_pair_activation': config.require_pair_activation, 'prepared_identity': prepared_identity})

def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path = ensure_within_output_root(path)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        frame.to_csv(temporary, index=False, encoding='utf-8', lineterminator='\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

def _atomic_text(text: str, path: Path) -> Path:
    path = ensure_within_output_root(path)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

def _source_paths(prepared: PreparedDataset) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in prepared.manifest.get('source_files', []):
        if isinstance(row, Mapping) and 'role' in row and ('path' in row):
            result[str(row['role'])] = Path(str(row['path']))
    return result

def _native_15minute_pv(target: pd.DataFrame, prepared: PreparedDataset) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    sources = _source_paths(prepared)
    pv_path = sources.get('pvod_station_power', DEFAULT_PV_PATH)
    raw = pd.read_csv(pv_path, usecols=['date_time', 'power'])
    raw['source_timestamp'] = pd.to_datetime(raw.pop('date_time'), errors='raise', utc=True).dt.tz_localize(None)
    raw['source_local_timestamp'] = raw['source_timestamp'].dt.tz_localize(SOURCE_TIMEZONE).dt.tz_convert(TARGET_TIMEZONE).dt.tz_localize(None)
    raw['raw_power_mw'] = pd.to_numeric(raw.pop('power'), errors='raise')
    if raw['source_timestamp'].duplicated().any() or (raw['raw_power_mw'] < 0).any():
        raise RunContractError('native 15-minute PV source is duplicated or negative')
    if not raw['source_timestamp'].dt.minute.isin([0, 15, 30, 45]).all():
        raise RunContractError('native PV source is not aligned to 15-minute quarters')
    raw['calendar_key'] = list(zip(raw['source_local_timestamp'].dt.month, raw['source_local_timestamp'].dt.day, raw['source_local_timestamp'].dt.hour, raw['source_local_timestamp'].dt.minute))
    if raw['calendar_key'].duplicated().any():
        raise RunContractError('native PV calendar keys are ambiguous')
    lookup = raw.set_index('calendar_key')
    raw['climatology_key'] = list(zip(raw['source_local_timestamp'].dt.month, raw['source_local_timestamp'].dt.hour, raw['source_local_timestamp'].dt.minute))
    climatology: dict[tuple[int, int, int], dict[str, object]] = {}
    for key, group in raw.groupby('climatology_key', sort=False):
        climatology[key] = {'raw_power_mw': float(group['raw_power_mw'].mean()), 'source_timestamp': pd.NaT, 'source_record_count': int(len(group)), 'source_earliest_timestamp': group['source_timestamp'].min(), 'source_latest_timestamp': group['source_timestamp'].max(), 'source_local_earliest_timestamp': group['source_local_timestamp'].min(), 'source_local_latest_timestamp': group['source_local_timestamp'].max()}
    pv_policy = prepared.manifest.get('pv_policy', {})
    capacity_mw = float(pv_policy.get('source_capacity_mw', float(pv_policy.get('source_capacity_kw', 0.0)) / 1000.0))
    if not math.isfinite(capacity_mw) or capacity_mw <= 0:
        raise RunContractError('prepared PV policy lacks a positive source capacity')
    rated_kw = float(pv_policy.get('pv_rated_scaled_kw', 0.0))
    multiplier = float(prepared.manifest.get('pv_multiplier', 1.0))
    if rated_kw <= 0 or multiplier < 0:
        raise RunContractError('prepared PV rating/multiplier is invalid')
    source_hash = sha256_file(pv_path)
    mapped_values: list[float] = []
    rows: list[dict[str, object]] = []
    for record in target.itertuples(index=False):
        timestamp = pd.Timestamp(record.timestamp)
        if timestamp.tzinfo is not None:
            raise RunContractError('target 15-minute timestamps must be local-naive')
        target_timestamp_utc = timestamp.tz_localize(TARGET_TIMEZONE).tz_convert(SOURCE_TIMEZONE).tz_localize(None)
        key = (timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)
        mapping_category = 'exact'
        mapping_day_offset: int | None = 0
        if key in lookup.index:
            source = lookup.loc[[key]].iloc[0]
            raw_power = float(source['raw_power_mw'])
            source_timestamp = pd.Timestamp(source['source_timestamp'])
            source_local_timestamp = pd.Timestamp(source['source_local_timestamp'])
            source_count = 1
            earliest = source_timestamp
            latest = source_timestamp
            local_earliest = source_local_timestamp
            local_latest = source_local_timestamp
        else:
            source = None
            for offset in (-1, 1, -2, 2, -3, 3):
                adjacent = timestamp + pd.Timedelta(days=offset)
                adjacent_key = (adjacent.month, adjacent.day, adjacent.hour, adjacent.minute)
                if adjacent_key in lookup.index:
                    source = lookup.loc[[adjacent_key]].iloc[0]
                    mapping_category = 'plus_minus_day'
                    mapping_day_offset = offset
                    break
            if source is not None:
                raw_power = float(source['raw_power_mw'])
                source_timestamp = pd.Timestamp(source['source_timestamp'])
                source_local_timestamp = pd.Timestamp(source['source_local_timestamp'])
                source_count = 1
                earliest = source_timestamp
                latest = source_timestamp
                local_earliest = source_local_timestamp
                local_latest = source_local_timestamp
            else:
                climate_key = (timestamp.month, timestamp.hour, timestamp.minute)
                climate = climatology.get(climate_key)
                if climate is None:
                    mapping_category = 'zero'
                    mapping_day_offset = None
                    raw_power = 0.0
                    source_timestamp = pd.NaT
                    source_local_timestamp = pd.NaT
                    source_count = 0
                    earliest = pd.NaT
                    latest = pd.NaT
                    local_earliest = pd.NaT
                    local_latest = pd.NaT
                else:
                    mapping_category = 'climatology'
                    mapping_day_offset = None
                    raw_power = float(climate['raw_power_mw'])
                    source_timestamp = pd.NaT
                    source_local_timestamp = pd.NaT
                    source_count = int(climate['source_record_count'])
                    earliest = climate['source_earliest_timestamp']
                    latest = climate['source_latest_timestamp']
                    local_earliest = climate['source_local_earliest_timestamp']
                    local_latest = climate['source_local_latest_timestamp']
        if source_count and (not pd.Timestamp(latest) < target_timestamp_utc):
            raise RunContractError('native PV calendar proxy contains a source time at/after the target time in UTC')
        normalized_unclipped = raw_power / capacity_mw
        normalized = float(np.clip(normalized_unclipped, 0.0, 1.0))
        before_multiplier = normalized * rated_kw
        after_multiplier = before_multiplier * multiplier
        mapped_values.append(after_multiplier)
        rows.append({'split_id': record.split_id, 'split_role': record.split_role, 'timestamp': timestamp, 'target_resolution_minutes': 15, 'source_resolution_minutes': 15, 'source_timestamp': source_timestamp, 'source_earliest_timestamp': earliest, 'source_latest_timestamp': latest, 'source_local_timestamp': source_local_timestamp, 'source_local_earliest_timestamp': local_earliest, 'source_local_latest_timestamp': local_latest, 'target_timestamp_utc': target_timestamp_utc, 'mapping_category': mapping_category, 'mapping_day_offset': mapping_day_offset, 'source_record_count': source_count, 'source_power_mean_mw': raw_power, 'source_capacity_mw': capacity_mw, 'normalized_power_unclipped_pu': normalized_unclipped, 'normalized_power_pu': normalized, 'pv_rated_scaled_kw': rated_kw, 'pv_multiplier': multiplier, 'pv_power_kw_before_multiplier': before_multiplier, 'pv_power_kw': after_multiplier, 'source_timezone': SOURCE_TIMEZONE, 'target_timezone': TARGET_TIMEZONE, 'timezone_conversion_applied': True, 'calendar_mapping': PV_CALENDAR_MAPPING, 'non_colocated_source': True, 'uses_target_future_data': False, 'source_file': str(pv_path.resolve()), 'source_file_sha256': source_hash})
    provenance = pd.DataFrame(rows)
    policy = {**dict(pv_policy), 'target_resolution_minutes': 15, 'source_resolution_minutes': 15, 'resampling': 'native_15min_calendar_proxy', 'hourly_average_repetition_used': False, 'mapping_category_counts': {str(key): int(value) for key, value in provenance['mapping_category'].value_counts().items()}, 'uses_target_future_data': False, 'source_file_sha256': source_hash}
    return (np.asarray(mapped_values, dtype=float), provenance, policy)

def _prepare_resolution(config: ExperimentConfig) -> PreparedDataset:
    if config.step_minutes == 60:
        verify_prepared_dataset(config.prepared)
        return config.prepared
    if config.prepared.manifest.get('control_resolution_minutes') == 15:
        _verify_runtime_prepared(config)
        return config.prepared
    verify_prepared_dataset(config.prepared)
    if config.run_kind != 'single_station_15min_scheduler':
        raise RunContractError('15-minute execution is restricted to its registered run kind')
    hourly = config.prepared.inputs.sort_values('timestamp').reset_index(drop=True)
    quarter_frames: list[pd.DataFrame] = []
    for quarter in range(4):
        frame = hourly.copy()
        frame['timestamp'] = pd.to_datetime(frame['timestamp']) + pd.Timedelta(minutes=15 * quarter)
        frame['within_hour_step'] = quarter
        quarter_frames.append(frame)
    inputs = pd.concat(quarter_frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    drop_sources = [name for name in inputs.columns if name.startswith('p24_source_') or name.startswith('p168_source_')]
    if drop_sources:
        inputs = inputs.drop(columns=drop_sources)
    pv_values, provenance, pv_policy = _native_15minute_pv(inputs[['split_id', 'split_role', 'timestamp']], config.prepared)
    inputs['pv_power_kw'] = pv_values
    inputs['pv_power_kw_source'] = pv_values / max(float(config.prepared.manifest.get('pv_multiplier', 1.0)), 1e-300)
    inputs['ev_energy_kwh'] = inputs['ev_load_kw'] * 0.25
    inputs['pv_energy_kwh'] = inputs['pv_power_kw'] * 0.25
    inputs['net_load_without_bess_kw'] = inputs['ev_load_kw'] - inputs['pv_power_kw']
    inputs['grid_load_without_bess_kw'] = inputs['net_load_without_bess_kw'].clip(lower=0.0)
    inputs['pv_curtail_without_bess_kw'] = (-inputs['net_load_without_bess_kw']).clip(lower=0.0)
    inputs['grid_cost_without_bess_yuan'] = inputs['grid_load_without_bess_kw'] * inputs['price_yuan_per_kwh'] * 0.25
    role_by_time = inputs.set_index('timestamp')['split_role']
    for predictor, hours in (('p24', 24), ('p168', 168)):
        source = inputs['timestamp'] - pd.Timedelta(hours=hours)
        inputs[f'{predictor}_source_timestamp'] = source
        inputs[f'{predictor}_source_role'] = source.map(role_by_time)
    manifest = dict(config.prepared.manifest)
    manifest.update({'control_resolution_minutes': 15, 'step_hours': 0.25, 'row_counts': {str(role): int(count) for role, count in inputs['split_role'].value_counts().items()}, 'row_count_unit': 'control_steps', 'hourly_source_row_counts': config.prepared.manifest.get('row_counts', {}), 'resampling': {'ev': 'hourly_power_held_for_four_steps', 'pv': 'native_15min_calendar_proxy', 'price': 'hourly_price_held_for_four_steps'}, 'pv_policy': pv_policy})
    return PreparedDataset(inputs=inputs, pv_provenance=provenance, manifest=manifest)

def _verify_runtime_prepared(config: ExperimentConfig) -> None:
    inputs = config.prepared.inputs
    provenance = config.prepared.pv_provenance
    required = {'split_id', 'split_role', 'timestamp', 'ev_load_kw', 'pv_power_kw', 'price_yuan_per_kwh'}
    if (missing := required.difference(inputs.columns)):
        raise RunContractError(f'runtime inputs miss columns: {sorted(missing)}')
    if inputs['timestamp'].duplicated().any() or not inputs['timestamp'].is_monotonic_increasing:
        raise RunContractError('runtime timestamps are duplicated or unsorted')
    values = inputs[['ev_load_kw', 'pv_power_kw', 'price_yuan_per_kwh']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RunContractError('runtime core inputs are invalid')
    if len(inputs) != len(provenance) or not inputs['timestamp'].reset_index(drop=True).equals(provenance['timestamp'].reset_index(drop=True)):
        raise RunContractError('runtime PV provenance is not one-to-one with inputs')
    if provenance['uses_target_future_data'].astype(bool).any():
        raise RunContractError('runtime PV mapping reads target-future data')
    if config.step_minutes == 15:
        required_clock_fields = {'source_record_count', 'source_latest_timestamp', 'source_local_latest_timestamp', 'target_timestamp_utc', 'source_timezone', 'target_timezone', 'timezone_conversion_applied', 'calendar_mapping'}
        if (missing := required_clock_fields.difference(provenance.columns)):
            raise RunContractError(f'runtime PV clock provenance is incomplete: {sorted(missing)}')
        if not (provenance['source_timezone'].eq(SOURCE_TIMEZONE).all() and provenance['target_timezone'].eq(TARGET_TIMEZONE).all() and provenance['timezone_conversion_applied'].eq(True).all() and provenance['calendar_mapping'].eq(PV_CALENDAR_MAPPING).all()):
            raise RunContractError('runtime PV clock mapping is not UTC-to-Asia/Shanghai')
        target_utc = pd.to_datetime(provenance['timestamp'], errors='raise').dt.tz_localize(TARGET_TIMEZONE).dt.tz_convert(SOURCE_TIMEZONE).dt.tz_localize(None)
        if not target_utc.equals(pd.to_datetime(provenance['target_timestamp_utc'], errors='raise')):
            raise RunContractError('runtime target UTC provenance differs from its local timestamp')
        has_source = pd.to_numeric(provenance['source_record_count'], errors='raise').gt(0)
        source_utc = pd.to_datetime(provenance.loc[has_source, 'source_latest_timestamp'], errors='raise')
        if source_utc.isna().any() or not source_utc.lt(target_utc.loc[has_source]).all():
            raise RunContractError('runtime PV source is not strictly past when compared in UTC')
        source_local = source_utc.dt.tz_localize(SOURCE_TIMEZONE).dt.tz_convert(TARGET_TIMEZONE).dt.tz_localize(None)
        if not source_local.equals(pd.to_datetime(provenance.loc[has_source, 'source_local_latest_timestamp'], errors='raise')):
            raise RunContractError('runtime source UTC/local provenance differs by an invalid offset')
    frequency = f'{config.step_minutes}min'
    windows = config.prepared.manifest.get('windows', {})
    for role, window in windows.items():
        observed = pd.DatetimeIndex(inputs.loc[inputs['split_role'].eq(role), 'timestamp'])
        expected = pd.date_range(pd.Timestamp(window['start']), pd.Timestamp(window['end']), inclusive='left', freq=frequency)
        if not observed.equals(expected):
            raise RunContractError(f'runtime {role} grid has {len(observed)} rows, expected {len(expected)}')

def prepare_control_resolution(config: ExperimentConfig) -> PreparedDataset:
    return _prepare_resolution(config)

def _load_precalibration_history(config: ExperimentConfig) -> pd.DataFrame:
    calibration = config.prepared.inputs.loc[config.prepared.inputs['split_role'].eq('calibration')].sort_values('timestamp')
    if calibration.empty:
        raise RunContractError('prepared dataset has no calibration role')
    calibration_start = pd.Timestamp(calibration['timestamp'].iloc[0])
    step = pd.Timedelta(minutes=config.step_minutes)
    expected = pd.date_range(calibration_start - step * config.forecast_lag, calibration_start, inclusive='left', freq=f'{config.step_minutes}min')
    if config.precalibration_history is not None:
        history = config.precalibration_history.copy()
        required = {'timestamp', 'ev_load_kw', 'pv_power_kw'}
        missing = required.difference(history.columns)
        if missing:
            raise RunContractError(f'precalibration_history missing {sorted(missing)}')
        history['timestamp'] = pd.to_datetime(history['timestamp'], errors='raise')
        history = history.sort_values('timestamp').reset_index(drop=True)
    elif 'forecast_history' in set(config.prepared.inputs['split_role']):
        history = config.prepared.inputs.loc[config.prepared.inputs['split_role'].eq('forecast_history')].sort_values('timestamp').tail(config.forecast_lag).copy()
        history = history[['timestamp', 'ev_load_kw', 'pv_power_kw']].reset_index(drop=True)
    else:
        if config.step_minutes != 60:
            raise RunContractError('15-minute pre-calibration history must be present in the prepared runtime data')
        sources = _source_paths(config.prepared)
        ev_path = sources.get('hourly_ev_load', DEFAULT_EV_PATH)
        pv_path = sources.get('pvod_station_power', DEFAULT_PV_PATH)
        metadata_path = sources.get('pvod_station_metadata', DEFAULT_PV_METADATA_PATH)
        ev = pd.read_csv(ev_path, usecols=['time', 'ev_load_kw'])
        ev['timestamp'] = pd.to_datetime(ev.pop('time'), errors='raise')
        ev = ev.set_index('timestamp').reindex(expected)
        ev.index.name = 'timestamp'
        if ev['ev_load_kw'].isna().any():
            raise RunContractError('raw EV source lacks complete pre-calibration history')
        ev_multiplier = float(config.prepared.manifest.get('ev_multiplier', 1.0))
        ev['ev_load_kw'] = pd.to_numeric(ev['ev_load_kw'], errors='raise') * ev_multiplier
        history = ev.reset_index()
        history['split_role'] = 'precalibration_history'
        pv_rated = float(config.prepared.manifest.get('pv_policy', {}).get('pv_rated_scaled_kw', config.core.pv_rated_kw))
        mapped, _, _ = map_pvod_hourly(history[['timestamp', 'split_role']], pv_path=pv_path, metadata_path=metadata_path, pv_rated_kw=pv_rated)
        pv_multiplier = float(config.prepared.manifest.get('pv_multiplier', 1.0))
        history['pv_power_kw'] = mapped['pv_power_kw'].to_numpy(float) * pv_multiplier
    observed = pd.DatetimeIndex(history['timestamp'])
    if not observed.equals(expected):
        raise RunContractError(f'pre-calibration history must contain exactly {len(expected)} hourly rows')
    values = history[['ev_load_kw', 'pv_power_kw']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RunContractError('pre-calibration history contains invalid values')
    history['source_role'] = 'past_only_precalibration'
    return history[['timestamp', 'ev_load_kw', 'pv_power_kw', 'source_role']]

def _calibration_residuals(config: ExperimentConfig, prehistory: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    calibration = config.prepared.inputs.loc[config.prepared.inputs['split_role'].eq('calibration')].sort_values('timestamp').reset_index(drop=True)
    forecaster = CausalPersistenceForecaster(config.forecast_lag)
    history_times = list(pd.to_datetime(prehistory['timestamp']))
    ev_history = list(prehistory['ev_load_kw'].to_numpy(float))
    pv_history = list(prehistory['pv_power_kw'].to_numpy(float))
    rows: list[dict[str, object]] = []
    ev_points: list[float] = []
    pv_points: list[float] = []
    for index, row in calibration.iterrows():
        target = pd.Timestamp(row['timestamp'])
        ev_point = float(forecaster.forecast(np.asarray(ev_history), 1)[0])
        pv_point = float(forecaster.forecast(np.asarray(pv_history), 1)[0])
        source_time = history_times[len(history_times) - config.forecast_lag]
        if not source_time < target:
            raise RunContractError('calibration persistence source is not strictly in the past')
        ev_actual = float(row['ev_load_kw'])
        pv_actual = float(row['pv_power_kw'])
        for variable, actual, point in (('EV', ev_actual, ev_point), ('PV', pv_actual, pv_point)):
            rows.append({'calibration_index': int(index), 'variable': variable, 'target_time': target, 'source_time': source_time, 'source_kind': 'observed_past', 'forecast_model': forecaster.name, 'actual_kw': actual, 'point_kw': point, 'residual_kw': actual - point, 'strictly_past': True})
        ev_points.append(ev_point)
        pv_points.append(pv_point)
        history_times.append(target)
        ev_history.append(ev_actual)
        pv_history.append(pv_actual)
    return (pd.DataFrame(rows), np.asarray(ev_points), np.asarray(pv_points))

def _build_calibration(config: ExperimentConfig, residual_frame: pd.DataFrame, ev_points: np.ndarray, pv_points: np.ndarray) -> tuple[CalibrationProvider, CalibrationPressureScaler, dict[str, object]]:
    calibration_rows = len(ev_points)
    n = config.core.calibration_window
    if calibration_rows < n:
        raise RunContractError(f'calibration requires {n} residuals but only {calibration_rows} are available')
    ev_residual_all = residual_frame.loc[residual_frame['variable'].eq('EV'), 'residual_kw'].to_numpy(float)
    pv_residual_all = residual_frame.loc[residual_frame['variable'].eq('PV'), 'residual_kw'].to_numpy(float)
    ev_residual = ev_residual_all[-n:]
    pv_residual = pv_residual_all[-n:]
    alpha = config.core.conformal_miscoverage
    method = Method(config.arm_id)
    if method_spec(method).adaptive_conformal:
        ev_cal: AsymmetricConformal = AdaptiveConformal(ev_residual, target_miscoverage=alpha, gamma=config.aci_gamma, rolling_window=config.aci_rolling_window, alpha_min=config.aci_alpha_min, alpha_max=config.aci_alpha_max)
        pv_cal: AsymmetricConformal = AdaptiveConformal(pv_residual, target_miscoverage=alpha, gamma=config.aci_gamma, rolling_window=config.aci_rolling_window, alpha_min=config.aci_alpha_min, alpha_max=config.aci_alpha_max)
        calibration_mode = 'PAST_ONLY_ACI'
    else:
        ev_cal = AsymmetricConformal(ev_residual, miscoverage=alpha)
        pv_cal = AsymmetricConformal(pv_residual, miscoverage=alpha)
        calibration_mode = 'STATIC'
    ev_static = AsymmetricConformal(ev_residual, miscoverage=alpha)
    pv_static = AsymmetricConformal(pv_residual, miscoverage=alpha)
    ev_snapshot = ev_static.freeze_horizon(ev_points[-n:], -1)
    pv_snapshot = pv_static.freeze_horizon(pv_points[-n:], -1)
    ev_upper = np.maximum(ev_snapshot.upper, 0.0)
    pv_lower = np.clip(pv_snapshot.lower, 0.0, config.core.pv_rated_kw)
    scaler = CalibrationPressureScaler.fit(ev_upper, pv_lower, quantile=config.core.pressure_quantile, clip=False)
    tail_alpha = alpha / 2.0
    protocol = {'calibration_mode': calibration_mode, 'calibration_residual_count_ev': int(ev_residual.size), 'calibration_residual_count_pv': int(pv_residual.size), 'tail_miscoverage': tail_alpha, 'finite_sample_rank': finite_sample_rank(n, tail_alpha), 'expected_default_rank_720_tau_0p05': 685, 'pressure_scale_kw': scaler.scale_kw, 'pressure_calibration_size': scaler.calibration_size, 'pressure_clip': scaler.clip}
    return (CalibrationProvider(ev_cal, pv_cal), scaler, protocol)

def _protocol_preflight(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, object]]:
    if config.step_minutes == 60:
        verify_prepared_dataset(config.prepared)
    else:
        _verify_runtime_prepared(config)
    prehistory = _load_precalibration_history(config)
    residual_frame, ev_points, pv_points = _calibration_residuals(config, prehistory)
    calibration = config.prepared.inputs.loc[config.prepared.inputs['split_role'].eq('calibration')]
    test = config.prepared.inputs.loc[config.prepared.inputs['split_role'].eq('test')]
    if len(calibration) < config.core.calibration_window:
        raise RunContractError(f'calibration has {len(calibration)} rows but this RunSpec requires {config.core.calibration_window}')
    if test.empty:
        raise RunContractError('prepared dataset has no test role')
    pretest = config.prepared.inputs.loc[config.prepared.inputs['timestamp'] < pd.Timestamp(test['timestamp'].min())].sort_values('timestamp')
    last_history = pd.DatetimeIndex(pretest['timestamp'].tail(config.forecast_lag))
    expected_history = pd.date_range(pd.Timestamp(test['timestamp'].min()) - pd.Timedelta(minutes=config.step_minutes * config.forecast_lag), pd.Timestamp(test['timestamp'].min()), inclusive='left', freq=f'{config.step_minutes}min')
    if not last_history.equals(expected_history):
        raise RunContractError('test first step lacks direct contiguous pre-test history')
    if len(residual_frame) != 2 * len(calibration):
        raise RunContractError('calibration residual trace is incomplete')
    if not residual_frame['strictly_past'].all():
        raise RunContractError('calibration source-time gate failed')
    report: dict[str, object] = {'schema_version': RUNNER_SCHEMA_VERSION, 'verdict': 'PASS', 'split_id': config.split_id, 'arm_id': config.arm_id, 'run_kind': config.run_kind, 'step_minutes': config.step_minutes, 'parameters_json': config.parameters_json, 'horizon_steps': config.core.horizon_steps, 'horizon_hours': config.core.horizon_steps * config.core.step_hours, 'reserve_hours': config.core.reserve_hours, 'battery_capacity_kwh': config.core.battery.energy_kwh, 'conformal_miscoverage': config.core.conformal_miscoverage, 'aci_gamma': config.aci_gamma, 'aci_rolling_window_steps': config.aci_rolling_window, 'prepared_dataset_contract': 'PASS', 'calibration_rows': int(len(calibration)), 'test_rows': int(len(test)), 'calibration_valid_residuals_per_variable': int(len(calibration)), 'precalibration_history_rows': int(len(prehistory)), 'precalibration_first': str(prehistory['timestamp'].iloc[0]), 'precalibration_last': str(prehistory['timestamp'].iloc[-1]), 'test_first_step_direct_predecessor': str(last_history[-1]), 'forecast_lag': config.forecast_lag, 'forecast_lag_hours': config.forecast_lag * config.step_minutes / 60.0, 'future_actual_reads': 0, 'pv_provenance_future_use_rows': int(config.prepared.pv_provenance['uses_target_future_data'].astype(bool).sum()), 'pv_resampling': config.prepared.manifest.get('resampling', {}).get('pv', 'hourly_native_target'), 'pv_hourly_average_repetition_used': config.prepared.manifest.get('pv_policy', {}).get('hourly_average_repetition_used'), 'dry_run': config.dry_run, 'max_steps': config.max_steps, 'declared_formal_evidence': config.formal_evidence, 'formal_evidence': config.effective_formal_evidence}
    return (prehistory, residual_frame, ev_points, pv_points, report)

def _forecast_sources(history_times: list[pd.Timestamp], target_times: list[pd.Timestamp], lag: int) -> list[tuple[pd.Timestamp, str]]:
    origin = len(history_times)
    sources: list[tuple[pd.Timestamp, str]] = []
    for offset in range(len(target_times)):
        source = origin + offset - lag
        if source < origin:
            sources.append((history_times[source], 'observed_past'))
        else:
            sources.append((target_times[source - origin], 'recursive_prediction'))
    return sources

def _reserve_requirement(controller: ControllerKind, adverse_net_kw: float, config: CoreExperimentConfig) -> float:
    if controller == ControllerKind.RB:
        return config.reserve_hours * adverse_net_kw
    if controller == ControllerKind.GS:
        return config.reserve_hours * max(adverse_net_kw - config.feeder_safe_kw, 0.0)
    return 0.0

def _gate_decision(decision: FirstStepDecision, config: ExperimentConfig) -> str:
    if decision.solver_status != 0:
        raise RunContractError(f'non-optimal solver status {decision.solver_status}')
    if decision.mip_gap is None:
        if 'optimal' not in decision.solver_message.casefold():
            raise RunContractError('solver omitted MIP gap without an optimal-status explanation')
        gap_policy = 'ACCEPTED_STATUS_0_OPTIMAL_MESSAGE_GAP_NOT_REPORTED'
    else:
        if not math.isfinite(decision.mip_gap) or decision.mip_gap > config.core.solver.mip_relative_gap:
            raise RunContractError(f'reported MIP gap {decision.mip_gap} exceeds {config.core.solver.mip_relative_gap}')
        gap_policy = 'REPORTED_AND_WITHIN_LIMIT'
    diagnostic_fields = ('max_constraint_violation', 'max_bound_violation', 'max_integrality_violation', 'first_stage_coupling_residual', 'first_step_power_balance_residual', 'first_step_soc_dynamics_residual')
    for name in diagnostic_fields:
        value = float(getattr(decision, name))
        if not math.isfinite(value) or value > config.residual_tolerance:
            raise RunContractError(f'MILP diagnostic {name}={value} exceeds {config.residual_tolerance}')
    return gap_policy

def _run_spec(config: ExperimentConfig, config_hash: str) -> dict[str, object]:
    payload = _configuration_payload(config)
    payload.update({'config_sha256': config_hash, 'declared_formal_evidence': config.formal_evidence, 'formal_evidence': config.effective_formal_evidence, 'diagnostic_truncation': config.max_steps is not None})
    return payload

def _write_input_artifacts(run: RunDirectory, config: ExperimentConfig, prehistory: pd.DataFrame, residuals: pd.DataFrame) -> tuple[str, dict[str, Path]]:
    inputs = config.prepared.inputs.sort_values('timestamp').reset_index(drop=True)
    paths = {'prepared_inputs': _atomic_csv(inputs, run.resolve('inputs/case_inputs.csv')), 'pv_provenance': _atomic_csv(config.prepared.pv_provenance.reset_index(drop=True), run.resolve('inputs/pv_hourly_provenance.csv')), 'precalibration_history': _atomic_csv(prehistory.reset_index(drop=True), run.resolve('inputs/precalibration_history.csv')), 'calibration_residuals': _atomic_csv(residuals.reset_index(drop=True), run.resolve('inputs/calibration_residuals.csv'))}
    paths['prepared_manifest'] = run.write_json('inputs/prepared_dataset_manifest.json', config.prepared.manifest)
    manifest_inputs = dict(paths)
    for role, path in _source_paths(config.prepared).items():
        manifest_inputs[f'raw_{role}'] = path
    run.record_input_manifest(manifest_inputs)
    return (sha256_file(paths['prepared_inputs']), paths)

def _partial_failure_outputs(run: RunDirectory, traces: list[StepTrace], step_rows: list[dict[str, object]], forecast_rows: list[dict[str, object]], shadow_rows: list[dict[str, object]], solver_rows: list[dict[str, object]]) -> None:
    candidates = ((traces, 'traces/partial_step_trace.csv', lambda: pd.DataFrame(traces_to_records(traces))), (step_rows, 'traces/partial_step_table.csv', lambda: pd.DataFrame(step_rows)), (forecast_rows, 'traces/partial_forecast_trace.csv', lambda: pd.DataFrame(forecast_rows)), (shadow_rows, 'traces/partial_shadow_filter_trace.csv', lambda: pd.DataFrame(shadow_rows)), (solver_rows, 'solver/partial_solver_diagnostics.csv', lambda: pd.DataFrame(solver_rows)))
    for rows, relative, make_frame in candidates:
        if rows:
            _atomic_csv(make_frame(), run.resolve(relative))

def _failed_manifest(run: RunDirectory) -> None:
    target = run.resolve('manifests/failed_output_files.csv')
    exclusions = {run.state_path.resolve(), target.resolve()}
    files = [path for path in run.root.rglob('*') if path.is_file() and path.resolve() not in exclusions]
    if files:
        write_manifest_csv(target, file_manifest_rows(files, base=run.root, role='failed_run_artifact'))

def simulate_arm(config: Any) -> RunArtifacts:
    if not isinstance(config, ExperimentConfig):
        raise RunContractError(f'unsupported experiment config type: {type(config).__name__}')
    runtime_prepared = _prepare_resolution(config)
    if runtime_prepared is not config.prepared:
        config = replace(config, prepared=runtime_prepared)
    config_payload = _configuration_payload(config)
    config_hash = canonical_json_sha256(config_payload)
    prehistory, residual_frame, ev_cal_points, pv_cal_points, protocol = _protocol_preflight(config)
    provider, pressure_scaler, calibration_protocol = _build_calibration(config, residual_frame, ev_cal_points, pv_cal_points)
    protocol.update(calibration_protocol)
    prepared_input_hash = canonical_json_sha256({'manifest': config.prepared.manifest, 'inputs': config.prepared.inputs.to_json(orient='split', date_format='iso')})
    if config.dry_run:
        return RunArtifacts(run_id=config.run_id or 'DRY_RUN', state='DRY_RUN', run_root=None, step_count=0, config_sha256=config_hash, input_sha256=prepared_input_hash, protocol_report=protocol, formal_evidence=False)
    spec_token = config.spec_id or f'{config.split_id}_{config.condition_id}_{config.arm_id}'
    run_id = config.run_id or make_run_id(spec_token)
    run = RunDirectory.create(run_id, _run_spec(config, config_hash), run_root=config.run_root, seed=config.core.seed)
    traces: list[StepTrace] = []
    step_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    shadow_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    attempt_logger: SolverAttemptLogger | None = None
    current_step: int | None = None
    run.transition('RUNNING', detail='causal closed-loop simulation started')
    try:
        attempt_logger = SolverAttemptLogger(run.resolve('solver/solver_attempts.csv'), run_id=run_id, run_kind=config.run_kind)
        input_sha256, _ = _write_input_artifacts(run, config, prehistory, residual_frame)
        run.write_json('audits/protocol_preflight.json', protocol)
        run.write_json('config/resolved_runner_config.json', config_payload)
        data = config.prepared.inputs.sort_values('timestamp').reset_index(drop=True)
        test = data.loc[data['split_role'].eq('test')].reset_index(drop=True)
        pretest = data.loc[data['timestamp'] < pd.Timestamp(test['timestamp'].iloc[0])].sort_values('timestamp')
        limit = len(test) if config.max_steps is None else min(len(test), config.max_steps)
        test = test.iloc[:limit].reset_index(drop=True)
        forecaster = CausalPersistenceForecaster(config.forecast_lag)
        history_times = list(pd.to_datetime(pretest['timestamp']))
        ev_history = list(pretest['ev_load_kw'].to_numpy(float))
        pv_history = list(pretest['pv_power_kw'].to_numpy(float))
        soc = config.core.battery.soc_initial
        method = Method(config.arm_id)
        spec = method_spec(method)
        scenario_hash = canonical_json_sha256({'core': asdict(config.core), 'controller': spec.controller.value})
        forecast_hash = canonical_json_sha256({'model': forecaster.name, 'lag': config.forecast_lag, 'calibration': calibration_protocol, 'aci_gamma': config.aci_gamma if spec.adaptive_conformal else None, 'aci_rolling_window': config.aci_rolling_window if spec.adaptive_conformal else None})
        scheduler_code_hash = sha256_file(Path(__file__).with_name('model.py'))
        for step, actual in test.iterrows():
            current_step = int(step)
            horizon_steps = config.core.horizon_steps
            decision_time = pd.Timestamp(actual['timestamp'])
            target_times = list(pd.date_range(decision_time, periods=horizon_steps, freq=f'{config.step_minutes}min'))
            ev_point = np.maximum(forecaster.forecast(np.asarray(ev_history), horizon_steps), 0.0)
            pv_point = np.clip(forecaster.forecast(np.asarray(pv_history), horizon_steps), 0.0, config.core.pv_rated_kw)
            frozen = provider.bounds(step, ev_point, pv_point)
            ev_lower = np.maximum(frozen.ev.lower, 0.0)
            ev_upper = np.maximum(frozen.ev.upper, ev_lower)
            pv_lower = np.clip(frozen.pv.lower, 0.0, config.core.pv_rated_kw)
            pv_upper = np.clip(np.maximum(frozen.pv.upper, pv_lower), 0.0, config.core.pv_rated_kw)
            adverse_net = np.maximum(ev_upper - pv_lower, 0.0)
            pressure = pressure_scaler.transform(adverse_net)
            ev_sources = _forecast_sources(history_times, target_times, config.forecast_lag)
            pv_sources = _forecast_sources(history_times, target_times, config.forecast_lag)
            for variable, point, lower, upper, snapshot, sources in (('EV', ev_point, ev_lower, ev_upper, frozen.ev, ev_sources), ('PV', pv_point, pv_lower, pv_upper, frozen.pv, pv_sources)):
                for offset, target_time in enumerate(target_times):
                    source_time, source_kind = sources[offset]
                    if source_kind == 'observed_past' and (not source_time < decision_time):
                        raise RunContractError('forecast observed source is not before decision time')
                    forecast_rows.append({'run_id': run_id, 'arm_id': config.arm_id, 'decision_step': int(step), 'decision_time': decision_time, 'target_time': target_time, 'horizon_offset': int(offset), 'variable': variable, 'forecast_model': forecaster.name, 'source_time': source_time, 'source_kind': source_kind, 'actual_future_read': False, 'point_kw': float(point[offset]), 'lower_kw': float(lower[offset]), 'upper_kw': float(upper[offset]), 'alpha_lower': snapshot.alpha_lower, 'alpha_upper': snapshot.alpha_upper, 'calibration_size': snapshot.calibration_size, 'lower_offset_kw': snapshot.lower_offset, 'upper_offset_kw': snapshot.upper_offset, 'forecast_config_sha256': forecast_hash})
            price_source = _source_paths(config.prepared).get('time_of_use_price')
            horizon_prices = prices_for_timestamps(target_times, price_source if price_source is not None else None) if price_source is not None else prices_for_timestamps(target_times)
            attempt_metadata: dict[str, object] = {'decision_step': int(step), 'decision_timestamp': decision_time}
            if 'profile_id' in actual.index:
                attempt_metadata.update({'profile_id': str(actual['profile_id']), 'station_id': str(actual['profile_station_id'])})
            # EN: Pre-step causal boundary: forecasts & conformal bounds use only strictly past history (`source_time < decision_time`);
            # EN: realized `(ev_load_kw, pv_power_kw)` remain hidden from `solve_first_step` and enter only during `apply_execution_filter`.
            # 中文：步前因果边界：点预测与保形区间仅读取决策时刻之前的历史观测（`source_time < decision_time`）；
            # 中文：当步真实负荷与光伏在 `solve_first_step` 求解时不可见，仅在执行层 `apply_execution_filter` 阶段量测接入。
            with solver_attempt_scope(attempt_logger, **attempt_metadata):
                decision = solve_first_step(method, HorizonData(ev_lower, ev_point, ev_upper, pv_lower, pv_point, pv_upper, horizon_prices, pressure), soc, config.core)
            gap_policy = _gate_decision(decision, config)
            reserve_requirement = _reserve_requirement(spec.controller, float(adverse_net[0]), config.core)
            filtered = apply_execution_filter(spec.execution_filter, decision.charge_kw, decision.discharge_kw, float(actual['ev_load_kw']), float(actual['pv_power_kw']), soc, config.core.battery, config.core.feeder_safe_kw, config.core.step_hours, reserve_requirement)
            soc_after = advance_soc_checked(soc, filtered.charge_kw, filtered.discharge_kw, config.core.battery, config.core.step_hours)
            # EN: Same-step counterfactual shadow F1 (`shadow_state_committed=False`) vs closed-loop state update (`soc = soc_after`).
            # 中文：同初值单步反事实影子修正（不改变闭环状态）与真实闭环状态推进（`soc = soc_after`）同步记录。
            shadow = apply_execution_filter(FilterKind.F1_RESERVE, decision.charge_kw, decision.discharge_kw, float(actual['ev_load_kw']), float(actual['pv_power_kw']), soc, config.core.battery, config.core.feeder_safe_kw, config.core.step_hours, reserve_requirement)
            shadow_soc = advance_soc_checked(soc, shadow.charge_kw, shadow.discharge_kw, config.core.battery, config.core.step_hours)
            ev_actual = float(actual['ev_load_kw'])
            pv_actual = float(actual['pv_power_kw'])
            net_nominal = ev_actual + decision.charge_kw - pv_actual - decision.discharge_kw
            grid_nominal = max(net_nominal, 0.0)
            curtail_nominal = filtered.plant_curtail_kw
            executed_formula = ev_actual + filtered.charge_kw - pv_actual - filtered.discharge_kw
            soc_formula = soc + config.core.battery.eta_charge * filtered.charge_kw * config.core.step_hours / config.core.battery.energy_kwh - filtered.discharge_kw * config.core.step_hours / (config.core.battery.eta_discharge * config.core.battery.energy_kwh)
            power_balance_residual = abs(executed_formula - filtered.final_net_exchange_kw)
            nominal_grid_power_balance_residual = abs(filtered.plant_grid_kw + filtered.plant_discharge_kw - filtered.plant_charge_kw - curtail_nominal - (ev_actual - pv_actual))
            execution_grid_power_balance_residual = abs(filtered.final_grid_kw + filtered.discharge_kw - filtered.charge_kw - filtered.curtail_kw - (ev_actual - pv_actual))
            soc_dynamics_residual = abs(soc_formula - soc_after)
            filter_algebra_residual = max(abs(decision.charge_kw - filtered.charge_reduction_kw - filtered.charge_kw), abs(decision.discharge_kw - filtered.common_discharge_clip_kw + filtered.extra_discharge_kw - filtered.discharge_kw), abs(max(filtered.final_net_exchange_kw, 0.0) - filtered.final_grid_kw), abs(max(filtered.final_net_exchange_kw - config.core.feeder_safe_kw, 0.0) - filtered.residual_excess_kw))
            physical_bound_violation = max(filtered.curtail_kw - pv_actual, -filtered.curtail_kw, -filtered.final_grid_kw, filtered.discharge_kw - ev_actual - filtered.charge_kw, 0.0)
            for name, value in (('execution_power_balance_residual', power_balance_residual), ('nominal_grid_power_balance_residual', nominal_grid_power_balance_residual), ('execution_grid_power_balance_residual', execution_grid_power_balance_residual), ('execution_soc_dynamics_residual', soc_dynamics_residual), ('filter_algebra_residual', filter_algebra_residual), ('execution_physical_bound_violation_kw', physical_bound_violation)):
                if not math.isfinite(value) or value > config.residual_tolerance:
                    raise RunContractError(f'{name}={value} exceeds {config.residual_tolerance} at step {step}')
            reserve_margin = config.core.battery.eta_discharge * config.core.battery.energy_kwh * (soc_after - config.core.battery.soc_min) - reserve_requirement
            trace = StepTrace(timestamp=decision_time.isoformat(sep=' '), step=int(step), run_id=run_id, condition_id=config.condition_id, arm_id=config.arm_id, scheduler_id=spec.controller.value, filter_id=spec.execution_filter.value, calibration_id='PAST_ONLY_ACI' if spec.adaptive_conformal else 'STATIC', soc_pre=float(soc), soc_post=float(soc_after), p_charge_nom_kw=decision.charge_kw, p_discharge_nom_kw=decision.discharge_kw, net_exchange_nom_kw=float(net_nominal), grid_import_nom_kw=float(grid_nominal), delta_charge_kw=filtered.charge_reduction_kw, delta_discharge_kw=filtered.extra_discharge_kw, p_charge_exec_kw=filtered.charge_kw, p_discharge_exec_kw=filtered.discharge_kw, net_exchange_exec_kw=filtered.final_net_exchange_kw, grid_import_exec_kw=filtered.final_grid_kw, reserve_requirement_kwh=float(reserve_requirement), reserve_headroom_kwh=float(max(reserve_margin, 0.0)), reserve_shortage_kwh=float(max(-reserve_margin, 0.0)), barrier_slack_kwh=decision.reserve_slack_kwh, feeder_slack_kw=decision.grid_slack_kw, solver_status=decision.solver_status, mip_gap=decision.mip_gap, solve_time_s=decision.solver_seconds, objective=decision.objective, certificate_applicable=spec.controller == ControllerKind.GS and decision.grid_slack_kw <= config.residual_tolerance and filtered.certificate_applicable)
            traces.append(trace)
            canonical = traces_to_records([trace])[0]
            evidence = {'config_sha256': config_hash, 'formal_evidence': config.effective_formal_evidence, 'input_sha256': input_sha256, 'forecast_config_sha256': forecast_hash, 'scenario_config_sha256': scenario_hash, 'scheduler_code_sha256': scheduler_code_hash, 'ev_actual_kw': ev_actual, 'pv_actual_kw': pv_actual, 'price_yuan_per_kwh': float(actual['price_yuan_per_kwh']), 'ev_point_kw': float(ev_point[0]), 'ev_lower_kw': float(ev_lower[0]), 'ev_upper_kw': float(ev_upper[0]), 'pv_point_kw': float(pv_point[0]), 'pv_lower_kw': float(pv_lower[0]), 'pv_upper_kw': float(pv_upper[0]), 'pressure': float(pressure[0]), 'feeder_safe_kw': config.core.feeder_safe_kw, 'filter_semantics_schema_version': FILTER_SEMANTICS_SCHEMA_VERSION, 'filter_active': filtered.triggered, 'filter_triggered': filtered.triggered, 'filter_correction_applied': filtered.correction_applied, 'filter_correction_capacity_kw': filtered.correction_capacity_kw, 'filter_capacity_limited_zero_correction': filtered.capacity_limited_zero_correction, 'filter_capacity_limited_residual': filtered.capacity_limited_residual, 'filter_residual_excess_kw': filtered.residual_excess_kw, 'plant_realization': PLANT_REALIZATION, 'plant_charge_kw': filtered.plant_charge_kw, 'plant_discharge_kw': filtered.plant_discharge_kw, 'plant_net_exchange_kw': filtered.plant_net_exchange_kw, 'plant_grid_kw': filtered.plant_grid_kw, 'plant_curtail_kw': filtered.plant_curtail_kw, 'common_discharge_clip_kw': filtered.common_discharge_clip_kw, 'raw_nominal_curtail_requirement_kw': max(-net_nominal, 0.0), 'raw_nominal_no_export_deficit_kw': max(decision.discharge_kw - ev_actual - decision.charge_kw, 0.0), 'canonical_delta_charge_kw': decision.charge_kw - filtered.charge_kw, 'canonical_delta_discharge_kw': filtered.discharge_kw - decision.discharge_kw, 'pv_curtail_nom_kw': float(curtail_nominal), 'pv_curtail_exec_kw': float(filtered.curtail_kw), 'execution_power_balance_residual': power_balance_residual, 'nominal_grid_power_balance_residual': nominal_grid_power_balance_residual, 'execution_grid_power_balance_residual': execution_grid_power_balance_residual, 'execution_soc_dynamics_residual': soc_dynamics_residual, 'filter_algebra_residual': filter_algebra_residual, 'execution_physical_bound_violation_kw': physical_bound_violation, 'presolve_policy_version': decision.presolve_policy_version, 'presolve_requested': decision.presolve_requested, 'presolve_effective': decision.presolve_effective, 'presolve_guard_applied': decision.presolve_guard_applied, 'presolve_guard_reason': decision.presolve_guard_reason, 'presolve_guard_soc_atol': decision.presolve_guard_soc_atol, 'solver_step_hours': config.core.step_hours, 'solver_soc_pre': float(soc), 'solver_soc_max': config.core.battery.soc_max, 'solver_soc_upper_distance': config.core.battery.soc_max - float(soc)}
            if 'profile_id' in actual.index:
                evidence.update({'profile_id': str(actual['profile_id']), 'profile_station_id': str(actual['profile_station_id']), 'profile_scale': float(actual['profile_scale'])})
            step_rows.append({**canonical, **evidence})
            shadow_reserve_margin = config.core.battery.eta_discharge * config.core.battery.energy_kwh * (shadow_soc - config.core.battery.soc_min) - reserve_requirement
            shadow_canonical = dict(canonical)
            shadow_canonical.update({'arm_id': f'{config.arm_id}_SHADOW_F1', 'filter_id': FilterKind.F1_RESERVE.value, 'soc_post': float(shadow_soc), 'reserve_headroom_kwh': float(max(shadow_reserve_margin, 0.0)), 'reserve_shortage_kwh': float(max(-shadow_reserve_margin, 0.0)), 'delta_charge_kw': shadow.charge_reduction_kw, 'delta_discharge_kw': shadow.extra_discharge_kw, 'p_charge_exec_kw': shadow.charge_kw, 'p_discharge_exec_kw': shadow.discharge_kw, 'net_exchange_exec_kw': shadow.final_net_exchange_kw, 'grid_import_exec_kw': shadow.final_grid_kw, 'certificate_applicable': spec.controller == ControllerKind.GS and decision.grid_slack_kw <= config.residual_tolerance and shadow.certificate_applicable})
            shadow_rows.append({**shadow_canonical, **evidence, 'filter_active': shadow.triggered, 'filter_triggered': shadow.triggered, 'filter_correction_applied': shadow.correction_applied, 'filter_correction_capacity_kw': shadow.correction_capacity_kw, 'filter_capacity_limited_zero_correction': shadow.capacity_limited_zero_correction, 'filter_capacity_limited_residual': shadow.capacity_limited_residual, 'filter_residual_excess_kw': shadow.residual_excess_kw, 'canonical_delta_charge_kw': decision.charge_kw - shadow.charge_kw, 'canonical_delta_discharge_kw': shadow.discharge_kw - decision.discharge_kw, 'pv_curtail_exec_kw': float(shadow.curtail_kw), 'execution_physical_bound_violation_kw': max(shadow.curtail_kw - pv_actual, -shadow.curtail_kw, -shadow.final_grid_kw, shadow.discharge_kw - ev_actual - shadow.charge_kw, 0.0), 'execution_grid_power_balance_residual': abs(shadow.final_grid_kw + shadow.discharge_kw - shadow.charge_kw - shadow.curtail_kw - (ev_actual - pv_actual)), 'shadow_filter_active': shadow.triggered, 'shadow_filter_triggered': shadow.triggered, 'shadow_filter_correction_applied': shadow.correction_applied, 'shadow_filter_correction_capacity_kw': shadow.correction_capacity_kw, 'shadow_filter_capacity_limited_zero_correction': shadow.capacity_limited_zero_correction, 'shadow_filter_capacity_limited_residual': shadow.capacity_limited_residual, 'shadow_residual_excess_kw': shadow.residual_excess_kw, 'shadow_state_committed': False})
            solver_rows.append({'timestamp': decision_time, 'step': int(step), 'run_id': run_id, 'arm_id': config.arm_id, 'solver_status': decision.solver_status, 'solver_message': decision.solver_message, 'mip_gap': decision.mip_gap, 'mip_gap_policy': gap_policy, 'mip_node_count': decision.mip_node_count, 'solve_time_s': decision.solver_seconds, 'objective': decision.objective, 'planned_grid_kw': decision.planned_grid_kw, 'planned_curtail_kw': decision.planned_curtail_kw, 'planned_soc': decision.planned_soc, 'max_constraint_violation': decision.max_constraint_violation, 'max_bound_violation': decision.max_bound_violation, 'max_integrality_violation': decision.max_integrality_violation, 'first_stage_coupling_residual': decision.first_stage_coupling_residual, 'first_step_power_balance_residual': decision.first_step_power_balance_residual, 'first_step_soc_dynamics_residual': decision.first_step_soc_dynamics_residual, 'presolve_policy_version': decision.presolve_policy_version, 'presolve_requested': decision.presolve_requested, 'presolve_effective': decision.presolve_effective, 'presolve_guard_applied': decision.presolve_guard_applied, 'presolve_guard_reason': decision.presolve_guard_reason, 'presolve_guard_soc_atol': decision.presolve_guard_soc_atol, 'solver_step_hours': config.core.step_hours, 'solver_soc_pre': float(soc), 'solver_soc_max': config.core.battery.soc_max, 'solver_soc_upper_distance': config.core.battery.soc_max - float(soc), 'execution_power_balance_residual': power_balance_residual, 'nominal_grid_power_balance_residual': nominal_grid_power_balance_residual, 'execution_grid_power_balance_residual': execution_grid_power_balance_residual, 'execution_soc_dynamics_residual': soc_dynamics_residual, 'filter_algebra_residual': filter_algebra_residual, 'execution_physical_bound_violation_kw': physical_bound_violation, 'residual_tolerance': config.residual_tolerance})
            soc = soc_after
            history_times.append(decision_time)
            ev_history.append(ev_actual)
            pv_history.append(pv_actual)
            provider.observe(ev_actual, pv_actual, frozen)
        step_trace_path = _atomic_csv(pd.DataFrame(traces_to_records(traces), columns=STEP_TRACE_COLUMNS), run.resolve('traces/step_trace.csv'))
        step_table_path = _atomic_csv(pd.DataFrame(step_rows), run.resolve('traces/step_table.csv'))
        forecast_path = _atomic_csv(pd.DataFrame(forecast_rows), run.resolve('traces/forecast_source_trace.csv'))
        shadow_path = _atomic_csv(pd.DataFrame(shadow_rows), run.resolve('traces/shadow_filter_trace.csv'))
        solver_path = _atomic_csv(pd.DataFrame(solver_rows), run.resolve('solver/solver_diagnostics.csv'))
        nominal_source = pd.DataFrame(step_rows).copy()
        nominal_source['delta_charge_kw'] = 0.0
        nominal_source['delta_discharge_kw'] = 0.0
        nominal_source['p_charge_exec_kw'] = nominal_source['plant_charge_kw']
        nominal_source['p_discharge_exec_kw'] = nominal_source['plant_discharge_kw']
        nominal_source['net_exchange_exec_kw'] = nominal_source['plant_net_exchange_kw']
        nominal_source['grid_import_exec_kw'] = nominal_source['plant_grid_kw']
        nominal_source['pv_curtail_exec_kw'] = nominal_source['plant_curtail_kw']
        nominal_source['soc_post'] = nominal_source['soc_pre'] + config.core.battery.eta_charge * nominal_source['plant_charge_kw'] * config.core.step_hours / config.core.battery.energy_kwh - nominal_source['plant_discharge_kw'] * config.core.step_hours / (config.core.battery.eta_discharge * config.core.battery.energy_kwh)
        shadow_audit = audit_pairing(nominal_source, pd.DataFrame(shadow_rows), mode='shadow', tolerance=config.pairing_tolerance, left_input_hash=input_sha256, right_input_hash=input_sha256, require_activation=False)
        run.write_json('audits/shadow_filter_pairing.json', shadow_audit.to_dict())
        if shadow_audit.verdict == 'FAIL':
            raise RunContractError('shadow-filter identity audit failed')
        pairing_path: Path | None = None
        if config.paired_f0_trace is not None:
            reference = pd.read_csv(config.paired_f0_trace, float_precision='round_trip')
            pair_result = audit_pairing(reference, pd.DataFrame(step_rows), mode='closed_loop', tolerance=config.pairing_tolerance, left_input_hash=config.paired_f0_input_sha256, right_input_hash=input_sha256, require_activation=config.require_pair_activation)
            pairing_path = run.write_json('audits/closed_loop_f0_f1_pairing.json', pair_result.to_dict())
            if pair_result.verdict == 'FAIL':
                raise RunContractError('closed-loop F0/F1 pairing audit failed')
        summary = {'schema_version': RUNNER_SCHEMA_VERSION, 'run_id': run_id, 'step_count': len(traces), 'expected_test_steps': len(config.prepared.inputs.loc[config.prepared.inputs['split_role'].eq('test')]), 'diagnostic_truncation': config.max_steps is not None, 'declared_formal_evidence': config.formal_evidence, 'formal_evidence': config.effective_formal_evidence, 'run_kind': config.run_kind, 'step_minutes': config.step_minutes, 'solver_success_count': sum((row['solver_status'] == 0 for row in solver_rows)), 'solver_failure_count': sum((row['solver_status'] != 0 for row in solver_rows)), 'max_reported_mip_gap': max((row['mip_gap'] for row in solver_rows if row['mip_gap'] is not None), default=None), 'max_constraint_violation': max((row['max_constraint_violation'] for row in solver_rows), default=None), 'max_execution_power_balance_residual': max((row['execution_power_balance_residual'] for row in solver_rows), default=None), 'max_nominal_grid_power_balance_residual': max((row['nominal_grid_power_balance_residual'] for row in solver_rows), default=None), 'max_execution_grid_power_balance_residual': max((row['execution_grid_power_balance_residual'] for row in solver_rows), default=None), 'max_execution_soc_dynamics_residual': max((row['execution_soc_dynamics_residual'] for row in solver_rows), default=None), 'max_filter_algebra_residual': max((row['filter_algebra_residual'] for row in solver_rows), default=None), 'max_execution_physical_bound_violation_kw': max((row['execution_physical_bound_violation_kw'] for row in solver_rows), default=None), 'plant_realization': PLANT_REALIZATION, 'common_no_export_adaptation_count': sum((row['common_discharge_clip_kw'] > config.residual_tolerance for row in step_rows)), 'common_no_export_discharge_clip_kwh': sum((row['common_discharge_clip_kw'] * config.core.step_hours for row in step_rows)), 'filter_semantics_schema_version': FILTER_SEMANTICS_SCHEMA_VERSION, 'filter_activation_count': sum((bool(row['filter_triggered']) for row in step_rows)), 'filter_trigger_count': sum((bool(row['filter_triggered']) for row in step_rows)), 'filter_correction_count': sum((bool(row['filter_correction_applied']) for row in step_rows)), 'filter_capacity_limited_zero_correction_count': sum((bool(row['filter_capacity_limited_zero_correction']) for row in step_rows)), 'filter_capacity_limited_residual_count': sum((bool(row['filter_capacity_limited_residual']) for row in step_rows)), 'shadow_filter_activation_count': sum((bool(row['shadow_filter_triggered']) for row in shadow_rows)), 'shadow_filter_trigger_count': sum((bool(row['shadow_filter_triggered']) for row in shadow_rows)), 'shadow_filter_correction_count': sum((bool(row['shadow_filter_correction_applied']) for row in shadow_rows)), 'shadow_filter_capacity_limited_zero_correction_count': sum((bool(row['shadow_filter_capacity_limited_zero_correction']) for row in shadow_rows)), 'shadow_filter_capacity_limited_residual_count': sum((bool(row['shadow_filter_capacity_limited_residual']) for row in shadow_rows)), 'presolve_policy_version': PRESOLVE_POLICY_VERSION, 'presolve_guard_count': sum((bool(row['presolve_guard_applied']) for row in solver_rows)), 'presolve_effective_true_count': sum((bool(row['presolve_effective']) for row in solver_rows)), 'presolve_requested_true_count': sum((bool(row['presolve_requested']) for row in solver_rows)), 'presolve_guard_reason_counts': {str(reason): sum((row['presolve_guard_reason'] == reason for row in solver_rows)) for reason in sorted({row['presolve_guard_reason'] for row in solver_rows if row['presolve_guard_reason'] is not None})}, 'future_actual_reads': 0}
        if attempt_logger is None:
            raise RunContractError('solver-attempt logger was not initialized')
        summary.update(attempt_logger.summary(expected_milp_solves=limit))
        run.write_json('solver/solver_summary.json', summary)
        complete = run.finalize(required_outputs=('traces/step_trace.csv', 'traces/step_table.csv', 'traces/forecast_source_trace.csv', 'traces/shadow_filter_trace.csv', 'solver/solver_diagnostics.csv', 'solver/solver_attempts.csv', 'solver/solver_summary.json', 'audits/protocol_preflight.json', 'audits/shadow_filter_pairing.json'))
        return RunArtifacts(run_id=run_id, state='COMPLETE', run_root=run.root, step_count=len(traces), config_sha256=config_hash, input_sha256=input_sha256, protocol_report=protocol, formal_evidence=config.effective_formal_evidence, step_trace=step_trace_path, step_table=step_table_path, forecast_trace=forecast_path, shadow_filter_trace=shadow_path, solver_diagnostics=solver_path, pairing_audit=pairing_path, completion_marker=complete)
    except Exception as exc:
        failure_evidence = f'step={current_step}; exception={type(exc).__name__}'
        try:
            _partial_failure_outputs(run, traces, step_rows, forecast_rows, shadow_rows, solver_rows)
            _atomic_text(traceback.format_exc(), run.resolve('logs/failure_traceback.txt'))
            run.write_json('RUN_FAILED.json', {'schema_version': RUNNER_SCHEMA_VERSION, 'run_id': run_id, 'failed_step': current_step, 'exception_type': type(exc).__name__, 'message': str(exc), 'completed_steps': len(traces), 'solver_attempt_summary': attempt_logger.summary() if attempt_logger is not None else None})
            run.record_anomaly(f"RUN-{run_id}-STEP-{(current_step if current_step is not None else 'PRE')}", 'ERROR', 'closed_loop_simulation', str(exc), evidence=failure_evidence)
            _failed_manifest(run)
            run.transition('FAILED', detail=failure_evidence)
        except Exception as preservation_exc:
            raise RunExecutionError(f'run failed and evidence preservation also failed: {exc}; {preservation_exc}', run_id=run_id, run_root=run.root) from exc
        raise RunExecutionError(f'run {run_id} failed at step {current_step}: {exc}', run_id=run_id, run_root=run.root) from exc

def _pair_identity(config: ExperimentConfig) -> str:
    return canonical_json_sha256({'prepared_manifest': config.prepared.manifest, 'prepared_inputs': config.prepared.inputs.to_json(orient='split', date_format='iso'), 'split_id': config.split_id, 'condition_id': config.condition_id, 'core': asdict(config.core), 'forecast_lag': config.forecast_lag, 'max_steps': config.max_steps})

# EN: Execute or verify paired `F0` (uncorrected baseline) and `F1` (REF-corrected) runs on identical inputs and forecasts.
# 中文：在完全一致的输入与预测序列上运行或核验 `F0`（无修正基准臂）与 `F1`（REF 修正臂）配对闭环对比。
def simulate_paired_arms(f0_config: ExperimentConfig, f1_config: ExperimentConfig) -> PairedRunArtifacts:
    f0_spec = method_spec(Method(f0_config.arm_id))
    f1_spec = method_spec(Method(f1_config.arm_id))
    if f0_spec.execution_filter != FilterKind.NONE:
        raise RunContractError('left pair arm must be filter-off')
    if f1_spec.execution_filter != FilterKind.F1_RESERVE:
        raise RunContractError('right pair arm must be REF filter-on')
    if f0_spec.controller != f1_spec.controller:
        raise RunContractError('paired arms must share the same scheduler')
    if f0_spec.adaptive_conformal != f1_spec.adaptive_conformal:
        raise RunContractError('paired arms must share the same calibration process')
    if _pair_identity(f0_config) != _pair_identity(f1_config):
        raise RunContractError('paired arms do not share identical inputs/configuration')
    if f0_config.dry_run or f1_config.dry_run:
        raise RunContractError('paired execution requires non-dry-run configurations')
    f0 = simulate_arm(f0_config)
    paired_f1 = replace(f1_config, paired_f0_trace=f0.step_table, paired_f0_input_sha256=f0.input_sha256)
    f1 = simulate_arm(paired_f1)
    if f1.pairing_audit is None:
        raise RunContractError('paired F1 run did not emit an audit')
    audit = json.loads(f1.pairing_audit.read_text(encoding='utf-8'))
    return PairedRunArtifacts(f0=f0, f1=f1, audit=audit)
__all__ = ['BoundsSnapshot', 'CalibrationProvider', 'ExperimentConfig', 'PairedRunArtifacts', 'RunArtifacts', 'RunContractError', 'RunExecutionError', 'SUPPORTED_RUN_KINDS', 'UNSUPPORTED_RUN_KINDS', 'config_from_run_spec', 'prepare_control_resolution', 'simulate_arm', 'simulate_paired_arms']
