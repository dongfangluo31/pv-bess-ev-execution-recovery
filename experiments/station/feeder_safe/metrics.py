# EN: Canonical 30-column step-trace normalization, 35 endpoint metrics, and paired `F0`/`F1` trajectory comparison.
# 中文：30 列标准步级轨迹规整、35 项终端评价指标计算及配对 `F0`/`F1` 闭环轨迹对比。
# EN: Inventory reference: paired terminal SOC/energy deltas always use the uncorrected `GS_F0` (`F0`) run on the same split/condition as baseline.
# 中文：库存对照基准：配对终端 SOC 与储能库存差均以同一切分与倍率条件下的无修正臂 `GS_F0`（`F0`）为基准对照。

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
TRACE_SCHEMA_VERSION = 2
TRACE_COLUMNS = ('timestamp', 'step', 'run_id', 'condition_id', 'arm_id', 'scheduler_id', 'filter_id', 'calibration_id', 'soc_pre', 'soc_post', 'p_charge_nom_kw', 'p_discharge_nom_kw', 'net_exchange_nom_kw', 'grid_import_nom_kw', 'delta_charge_kw', 'delta_discharge_kw', 'p_charge_exec_kw', 'p_discharge_exec_kw', 'net_exchange_exec_kw', 'grid_import_exec_kw', 'reserve_requirement_kwh', 'reserve_headroom_kwh', 'reserve_shortage_kwh', 'barrier_slack_kwh', 'feeder_slack_kw', 'solver_status', 'mip_gap', 'solve_time_s', 'objective', 'certificate_applicable')
TRACE_ALIASES: dict[str, tuple[str, ...]] = {'timestamp': ('time', 'datetime'), 'arm_id': ('method',), 'scheduler_id': ('controller',), 'filter_id': ('filter_mode',), 'soc_pre': ('soc_before', 'soc'), 'soc_post': ('soc_after',), 'p_charge_nom_kw': ('charge_raw_kw', 'battery_charge_raw_kw'), 'p_discharge_nom_kw': ('discharge_raw_kw', 'battery_discharge_raw_kw'), 'grid_import_nom_kw': ('raw_grid_kw', 'grid_load_before_filter_kw'), 'delta_charge_kw': ('filter_charge_reduction_kw', 'charge_reduction_kw'), 'delta_discharge_kw': ('filter_extra_discharge_kw', 'extra_discharge_kw'), 'p_charge_exec_kw': ('charge_kw', 'battery_charge_kw'), 'p_discharge_exec_kw': ('discharge_kw', 'battery_discharge_kw'), 'grid_import_exec_kw': ('grid_kw', 'grid_load_after_filter_kw', 'grid_load_kw'), 'reserve_requirement_kwh': ('reserve_need_kwh', 'reserve_required_kwh'), 'reserve_headroom_kwh': ('reserve_margin_kwh',), 'barrier_slack_kwh': ('reserve_slack_planned_kwh', 'reserve_slack_first_kwh'), 'feeder_slack_kw': ('grid_slack_planned_kw', 'grid_slack_first_kw'), 'solve_time_s': ('solver_seconds', 'solve_seconds')}
EXOGENOUS_ALIASES: dict[str, tuple[str, ...]] = {'ev_actual_kw': ('ev_load_kw',), 'pv_actual_kw': ('pv_power_kw',), 'price_yuan_per_kwh': ('price',), 'ev_point_kw': ('ev_pred_kw',), 'ev_lower_kw': ('ev_lo_kw',), 'ev_upper_kw': ('ev_up_kw',), 'pv_point_kw': ('pv_pred_kw',), 'pv_lower_kw': ('pv_lo_kw',), 'pv_upper_kw': ('pv_up_kw',), 'pressure': ('calibration_pressure',)}
HASH_COLUMNS = ('input_sha256', 'forecast_config_sha256', 'scenario_config_sha256', 'scheduler_code_sha256')
REQUIRED_EXOGENOUS_FIELDS = ('ev_actual_kw', 'pv_actual_kw', 'price_yuan_per_kwh')
REQUIRED_ENDPOINT_METRIC_IDS = frozenset({'peak_grid_import_nom_kw', 'peak_grid_import_exec_kw', 'prefilter_over_limit_step_count', 'prefilter_over_limit_hours', 'prefilter_longest_over_limit_duration_hours', 'postfilter_over_limit_step_count', 'postfilter_over_limit_hours', 'postfilter_longest_over_limit_duration_hours', 'median_solve_time_s', 'p95_solve_time_s', 'p99_solve_time_s', 'max_solve_time_s', 'barrier_slack_total_kwh', 'barrier_slack_peak_kwh', 'feeder_slack_total_kwh', 'feeder_slack_peak_kw', 'reserve_shortage_total_kwh', 'reserve_shortage_peak_kwh', 'ev_mpiw_kw', 'ev_pinaw', 'ev_interval_width_median_kw', 'ev_interval_width_p95_kw', 'ev_max_rolling_miss_rate', 'pv_mpiw_kw', 'pv_pinaw', 'pv_interval_width_median_kw', 'pv_interval_width_p95_kw', 'pv_max_rolling_miss_rate', 'aci_alpha_clip_rate', 'soc_lower_bound_touch_rate', 'soc_upper_bound_touch_rate', 'grid_energy_purchase_cost_yuan', 'lambda_eval_yuan_per_kwh', 'throughput_cost_at_lambda_eval_yuan', 'comparable_operating_cost_at_lambda_eval_yuan'})
COMMON_GS_RESERVE_METRIC_UNITS: dict[str, str] = {'common_gs_reserve_requirement_peak_kwh': 'kWh', 'common_gs_available_reserve_energy_peak_kwh': 'kWh', 'common_gs_reserve_headroom_peak_kwh': 'kWh', 'common_gs_reserve_shortage_peak_kwh': 'kWh', 'common_gs_reserve_shortage_active_step_count': 'steps', 'common_gs_reserve_shortage_active_hours': 'h', 'common_gs_reserve_shortage_exposure_kwh_h': 'kWh*h'}

class MetricContractError(ValueError):
    pass

@dataclass(frozen=True)
class MetricRecord:
    metric_id: str
    status: str
    value: float | None
    unit: str
    numerator: float | None
    denominator: float | None
    denominator_definition: str
    excluded_count: int
    notes: str

@dataclass(frozen=True)
class PairingAuditResult:
    schema_version: int
    verdict: str
    mode: str
    tolerance: float
    row_count_left: int
    row_count_right: int
    external_hash_evidence_present: bool
    external_hashes_match: bool
    timestamps_match: bool
    scheduler_match: bool
    exogenous_match: bool
    first_trigger_step: int | None
    first_trigger_timestamp: str | None
    first_correction_step: int | None
    first_correction_timestamp: str | None
    first_activation_step: int | None
    first_activation_timestamp: str | None
    compared_prefix_rows: int
    mismatch_counts: dict[str, int]
    first_mismatches: dict[str, dict[str, object]]
    post_correction_state_divergence_rows: int
    post_activation_state_divergence_rows: int
    trigger_event_snapshot: dict[str, object] | None
    correction_event_snapshot: dict[str, object] | None
    event_snapshot: dict[str, object] | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.schema_version == 1:
            for key in ('first_trigger_step', 'first_trigger_timestamp', 'first_correction_step', 'first_correction_timestamp', 'post_correction_state_divergence_rows', 'trigger_event_snapshot', 'correction_event_snapshot'):
                value.pop(key, None)
        return value

def _first_existing(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)

def normalize_trace_schema(frame: pd.DataFrame, *, defaults: Mapping[str, object] | None=None, require_complete: bool=False) -> pd.DataFrame:
    result = frame.copy()
    defaults = dict(defaults or {})
    for target, aliases in {**TRACE_ALIASES, **EXOGENOUS_ALIASES}.items():
        if target not in result:
            source = _first_existing(result, aliases)
            if source is not None:
                result[target] = result[source]
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
    if 'timestamp' in result:
        result['timestamp'] = pd.to_datetime(result['timestamp'], errors='raise')
        if result['timestamp'].dt.tz is not None:
            raise MetricContractError('trace timestamps must be naive')
    if 'step' not in result:
        result['step'] = np.arange(len(result), dtype=int)
    if 'calibration_id' not in result:
        method = result.get('arm_id', pd.Series('', index=result.index)).astype(str).str.upper()
        result['calibration_id'] = np.where(method.str.contains('ACI'), 'PAST_ONLY_ACI', 'STATIC')
    ev = _first_existing(result, ('ev_actual_kw', 'ev_load_kw'))
    pv = _first_existing(result, ('pv_actual_kw', 'pv_power_kw'))
    if 'net_exchange_nom_kw' not in result and ev and pv and {'p_charge_nom_kw', 'p_discharge_nom_kw'}.issubset(result.columns):
        result['net_exchange_nom_kw'] = pd.to_numeric(result[ev]) - pd.to_numeric(result[pv]) + pd.to_numeric(result['p_charge_nom_kw']) - pd.to_numeric(result['p_discharge_nom_kw'])
    if 'grid_import_nom_kw' not in result and 'net_exchange_nom_kw' in result:
        result['grid_import_nom_kw'] = pd.to_numeric(result['net_exchange_nom_kw']).clip(lower=0.0)
    if 'net_exchange_exec_kw' not in result and ev and pv and {'p_charge_exec_kw', 'p_discharge_exec_kw'}.issubset(result.columns):
        result['net_exchange_exec_kw'] = pd.to_numeric(result[ev]) - pd.to_numeric(result[pv]) + pd.to_numeric(result['p_charge_exec_kw']) - pd.to_numeric(result['p_discharge_exec_kw'])
    if 'grid_import_exec_kw' not in result and 'net_exchange_exec_kw' in result:
        result['grid_import_exec_kw'] = pd.to_numeric(result['net_exchange_exec_kw']).clip(lower=0.0)
    if 'reserve_shortage_kwh' not in result and 'reserve_headroom_kwh' in result:
        source = _first_existing(result, ('reserve_margin_kwh',))
        if source is not None:
            result['reserve_shortage_kwh'] = (-pd.to_numeric(result[source], errors='coerce')).clip(lower=0.0)
    if 'delta_charge_kw' not in result and {'p_charge_nom_kw', 'p_charge_exec_kw'}.issubset(result.columns):
        result['delta_charge_kw'] = pd.to_numeric(result['p_charge_nom_kw']) - pd.to_numeric(result['p_charge_exec_kw'])
    if 'delta_discharge_kw' not in result and {'p_discharge_nom_kw', 'p_discharge_exec_kw'}.issubset(result.columns):
        if 'common_discharge_clip_kw' in result and 'plant_discharge_kw' not in result:
            raise MetricContractError('physical-plant trace lacks explicit REF discharge correction and plant input')
        if 'downstream_actuation_delta_discharge_kw' in result:
            raise MetricContractError('delayed/ramped trace requires explicit operator-only correction')
        nominal_column = 'plant_discharge_kw' if 'plant_discharge_kw' in result else 'p_discharge_nom_kw'
        result['delta_discharge_kw'] = pd.to_numeric(result['p_discharge_exec_kw']) - pd.to_numeric(result[nominal_column])
    if require_complete:
        missing = [column for column in TRACE_COLUMNS if column not in result]
        if missing:
            raise MetricContractError(f'trace is missing frozen-schema columns: {missing}')
    return result

def _evaluation_mask(frame: pd.DataFrame) -> np.ndarray:
    if 'evaluation_eligible' not in frame:
        return np.ones(len(frame), dtype=bool)
    raw = frame['evaluation_eligible']
    if raw.isna().any():
        raise MetricContractError('evaluation_eligible cannot contain missing values')
    return raw.astype(bool).to_numpy()

def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise MetricContractError(f'required metric column is absent: {column}')
    return pd.to_numeric(frame[column], errors='coerce').to_numpy(float)

def _success_mask(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors='coerce')
    text = values.astype(str).str.strip().str.casefold()
    return (numeric.eq(0) | text.isin({'optimal', 'success', 'ok', '0'})).to_numpy()

def _strict_bool_mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise MetricContractError(f'required boolean metric column is absent: {column}')
    parsed: list[bool] = []
    invalid: list[object] = []
    for index, value in frame[column].items():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.strip().casefold() in {'true', 'false'}:
            parsed.append(value.strip().casefold() == 'true')
        else:
            invalid.append(index)
            parsed.append(False)
    if invalid:
        raise MetricContractError(f'boolean metric column {column} contains invalid values at rows {invalid[:10]}')
    return np.asarray(parsed, dtype=bool)

def _record(metric_id: str, value: float | None, unit: str, numerator: float | None, denominator: float | None, denominator_definition: str, excluded_count: int=0, notes: str='', status: str='OK') -> MetricRecord:
    allowed_statuses = {'OK', 'PARTIAL', 'PENDING', 'UNDEFINED', 'NOT_APPLICABLE'}
    if status not in allowed_statuses:
        raise MetricContractError(f'invalid metric status {status!r}')
    if denominator == 0:
        value = None
        if status == 'OK':
            status = 'UNDEFINED'
    return MetricRecord(metric_id=metric_id, status=status, value=None if value is None or not math.isfinite(value) else float(value), unit=unit, numerator=None if numerator is None or not math.isfinite(numerator) else float(numerator), denominator=None if denominator is None or not math.isfinite(denominator) else float(denominator), denominator_definition=denominator_definition, excluded_count=int(excluded_count), notes=notes)

def _pending_record(metric_id: str, unit: str, reason: str) -> MetricRecord:
    return _record(metric_id, None, unit, None, None, 'pending: required trace evidence is unavailable', notes=reason, status='PENDING')

def _validity_status(valid_count: int, eligible_count: int) -> str:
    if valid_count == 0:
        return 'PENDING'
    return 'OK' if valid_count == eligible_count else 'PARTIAL'

def _rate_record(metric_id: str, events: np.ndarray, eligible: np.ndarray, denominator_definition: str, *, notes: str='') -> MetricRecord:
    numerator = int(np.count_nonzero(events & eligible))
    denominator = int(np.count_nonzero(eligible))
    return _record(metric_id, numerator / denominator if denominator else None, 'fraction', numerator, denominator, denominator_definition, notes=notes)

def _finite_rate_record(metric_id: str, events: np.ndarray, valid: np.ndarray, eligible: np.ndarray, denominator_definition: str, *, notes: str='') -> MetricRecord:
    denominator_mask = eligible & valid
    denominator = int(np.count_nonzero(denominator_mask))
    numerator = int(np.count_nonzero(events & denominator_mask))
    eligible_count = int(np.count_nonzero(eligible))
    return _record(metric_id, numerator / denominator if denominator else None, 'fraction', numerator, denominator, denominator_definition, excluded_count=eligible_count - denominator, notes=notes, status=_validity_status(denominator, eligible_count))

def _coverage_record(metric_id: str, actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, eligible: np.ndarray) -> MetricRecord:
    finite = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    valid = eligible & finite
    covered = valid & (actual >= lower) & (actual <= upper)
    denominator = int(np.count_nonzero(valid))
    numerator = int(np.count_nonzero(covered))
    excluded = int(np.count_nonzero(eligible & ~finite))
    return _record(metric_id, numerator / denominator if denominator else None, 'fraction', numerator, denominator, 'eligible test steps with finite actual, lower, and upper values', excluded_count=excluded, notes='Excluded coverage rows are disclosed and are not removed from safety denominators.', status=_validity_status(denominator, int(np.count_nonzero(eligible))))

def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest

def _peak_record(metric_id: str, values: np.ndarray, eligible: np.ndarray, *, notes: str) -> MetricRecord:
    valid = eligible & np.isfinite(values)
    valid_count = int(np.count_nonzero(valid))
    eligible_count = int(np.count_nonzero(eligible))
    return _record(metric_id, float(np.max(values[valid])) if valid_count else None, 'kW', None, valid_count, 'finite eligible step values used by the maximum statistic', excluded_count=eligible_count - valid_count, notes=notes, status=_validity_status(valid_count, eligible_count))

def _over_limit_records(prefix: str, values: np.ndarray, eligible: np.ndarray, feeder_limit_kw: float, tolerance: float, step_hours: float) -> list[MetricRecord]:
    finite = np.isfinite(values)
    valid = eligible & finite
    events = valid & (values > feeder_limit_kw + tolerance)
    event_count = int(np.count_nonzero(events))
    eligible_count = int(np.count_nonzero(eligible))
    valid_count = int(np.count_nonzero(valid))
    status = _validity_status(valid_count, eligible_count)
    duration = event_count * step_hours
    longest_steps = _longest_true_run(events)
    complete = valid_count == eligible_count
    excess_energy = float(np.maximum(values[eligible] - feeder_limit_kw, 0.0).sum() * step_hours) if complete else None
    return [_record(f'{prefix}_over_limit_step_count', event_count, 'steps', event_count, eligible_count, 'all pre-registered evaluation steps', excluded_count=eligible_count - valid_count, notes='Only finite measured imports count as observed exceedances; missing imports remain separately disclosed.', status=status), _record(f'{prefix}_over_limit_hours', duration, 'h', duration, eligible_count * step_hours, 'total pre-registered evaluation duration', excluded_count=eligible_count - valid_count, status=status), _record(f'{prefix}_longest_over_limit_duration_hours', longest_steps * step_hours, 'h', longest_steps, eligible_count, 'longest consecutive finite over-limit run divided by all registered steps only for context', excluded_count=eligible_count - valid_count, status=status), _record(f'{prefix}_over_limit_energy_kwh', excess_energy, 'kWh', excess_energy, eligible_count * step_hours, 'integral over the complete pre-registered evaluation duration', excluded_count=eligible_count - valid_count, notes='Undefined when any eligible import is missing, because a partial integral would be biased low.', status='OK' if complete else 'PENDING')]

def _resolve_constant(frame: pd.DataFrame, column: str, override: float | None, *, tolerance: float) -> float | None:
    if override is not None:
        value = float(override)
        if not math.isfinite(value):
            raise MetricContractError(f'{column} override must be finite')
        return value
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors='coerce').dropna().to_numpy(float)
    if len(values) == 0:
        return None
    if not np.allclose(values, values[0], atol=tolerance, rtol=0.0):
        raise MetricContractError(f'{column} must be constant within one trace')
    return float(values[0])

def _slack_records(trace: pd.DataFrame, eligible: np.ndarray, *, column: str, prefix: str, source_unit: str, step_hours: float, tolerance: float, total_notes: str='') -> list[MetricRecord]:
    total_id = f'{prefix}_total_kwh'
    peak_id = f"{prefix}_peak_{('kw' if source_unit == 'kW' else 'kwh')}"
    rate_id = f'{prefix}_active_rate'
    if column not in trace:
        reason = f'missing required slack column {column}'
        return [_pending_record(total_id, 'kWh', reason), _pending_record(peak_id, source_unit, reason), _pending_record(rate_id, 'fraction', reason)]
    values = pd.to_numeric(trace[column], errors='coerce').to_numpy(float)
    if np.any(np.isfinite(values) & (values < -tolerance)):
        raise MetricContractError(f'{column} contains a negative slack')
    valid = eligible & np.isfinite(values)
    eligible_count = int(np.count_nonzero(eligible))
    valid_count = int(np.count_nonzero(valid))
    complete = valid_count == eligible_count
    multiplier = step_hours if source_unit == 'kW' else 1.0
    total_value = float(np.maximum(values[valid], 0.0).sum() * multiplier) if complete else None
    peak_value = float(np.max(np.maximum(values[valid], 0.0))) if valid_count else None
    active = np.isfinite(values) & (values > tolerance)
    return [_record(total_id, total_value, 'kWh', total_value, eligible_count, 'all registered steps; kW slack is integrated using the declared step duration', excluded_count=eligible_count - valid_count, notes=' '.join((value for value in ('Undefined if any eligible slack value is missing.', total_notes) if value)), status='OK' if complete else 'PENDING'), _record(peak_id, peak_value, source_unit, None, valid_count, 'finite eligible slack observations used by the maximum statistic', excluded_count=eligible_count - valid_count, status=_validity_status(valid_count, eligible_count)), _finite_rate_record(rate_id, active, np.isfinite(values), eligible, 'eligible steps with a finite slack value')]

def _energy_deficit_duration_records(trace: pd.DataFrame, eligible: np.ndarray, *, column: str, prefix: str, step_hours: float, tolerance: float, notes: str) -> list[MetricRecord]:
    metric_units = {f'{prefix}_active_step_count': 'steps', f'{prefix}_active_hours': 'h', f'{prefix}_exposure_kwh_h': 'kWh*h'}
    if column not in trace:
        reason = f'missing required energy-deficit column {column}'
        return [_pending_record(metric_id, unit, reason) for metric_id, unit in metric_units.items()]
    values = pd.to_numeric(trace[column], errors='coerce').to_numpy(float)
    if np.any(np.isfinite(values) & (values < -tolerance)):
        raise MetricContractError(f'{column} contains a negative energy deficit')
    valid = eligible & np.isfinite(values)
    eligible_count = int(np.count_nonzero(eligible))
    valid_count = int(np.count_nonzero(valid))
    if valid_count != eligible_count:
        reason = f'{column} is missing on {eligible_count - valid_count} of {eligible_count} eligible steps; safety exposure is not partially integrated'
        return [_record(metric_id, None, unit, None, eligible_count, 'all pre-registered evaluation steps', excluded_count=eligible_count - valid_count, notes=reason, status='PENDING') for metric_id, unit in metric_units.items()]
    nonnegative = np.maximum(values, 0.0)
    active = eligible & (nonnegative > tolerance)
    active_steps = int(np.count_nonzero(active))
    active_hours = float(active_steps * step_hours)
    exposure = float(nonnegative[eligible].sum() * step_hours)
    total_hours = float(eligible_count * step_hours)
    return [_record(f'{prefix}_active_step_count', active_steps, 'steps', active_steps, eligible_count, 'all pre-registered evaluation steps', notes=notes), _record(f'{prefix}_active_hours', active_hours, 'h', active_hours, total_hours, 'all pre-registered evaluation hours', notes=notes), _record(f'{prefix}_exposure_kwh_h', exposure, 'kWh*h', exposure, total_hours, 'time integral of the per-step energy deficit over all evaluation hours', notes=notes)]

def _one_text_value(frame: pd.DataFrame, column: str, *, role: str) -> str:
    if column not in frame:
        raise MetricContractError(f'{role} is missing required provenance column {column}')
    values = frame[column].dropna().astype(str).str.strip().unique()
    if len(values) != 1 or not values[0]:
        raise MetricContractError(f'{role} must contain exactly one nonempty {column}, got {values.tolist()}')
    return str(values[0])

def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.strip().casefold()
    if len(normalized) != 64:
        raise MetricContractError(f'{label} must be one 64-character SHA-256')
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise MetricContractError(f'{label} is not hexadecimal') from exc
    return normalized

def compute_common_gs_reserve_step_table(frame: pd.DataFrame, *, reserve_reference_frame: pd.DataFrame, feeder_limit_kw: float, reserve_hours_eval: float, battery_energy_kwh: float, eta_discharge: float, soc_min: float, step_hours: float, tolerance: float=1e-09, reserve_reference_sha256: str | None=None) -> pd.DataFrame:
    scalar_values = {'feeder_limit_kw': feeder_limit_kw, 'reserve_hours_eval': reserve_hours_eval, 'battery_energy_kwh': battery_energy_kwh, 'eta_discharge': eta_discharge, 'soc_min': soc_min, 'step_hours': step_hours, 'tolerance': tolerance}
    if any((not math.isfinite(float(value)) for value in scalar_values.values())):
        raise MetricContractError('common reserve constants must all be finite')
    if feeder_limit_kw <= 0 or reserve_hours_eval < 0 or battery_energy_kwh <= 0:
        raise MetricContractError('common reserve requires positive feeder/energy values and nonnegative reserve hours')
    if not 0 < eta_discharge <= 1 or not 0 <= soc_min < 1:
        raise MetricContractError('common reserve requires 0 < eta_discharge <= 1 and 0 <= soc_min < 1')
    if step_hours <= 0 or tolerance < 0:
        raise MetricContractError('step_hours must be positive and tolerance nonnegative')
    target_required = {'timestamp', 'step', 'run_id', 'input_sha256', 'soc_post'}
    reference_required = {'timestamp', 'step', 'run_id', 'arm_id', 'scheduler_id', 'calibration_id', 'input_sha256', 'ev_upper_kw', 'pv_lower_kw'}
    missing_target = sorted(target_required.difference(frame.columns))
    missing_reference = sorted(reference_required.difference(reserve_reference_frame.columns))
    if missing_target or missing_reference:
        raise MetricContractError(f'common reserve evidence is incomplete: target_missing={missing_target}, reference_missing={missing_reference}')
    target = normalize_trace_schema(frame)
    reference = normalize_trace_schema(reserve_reference_frame)
    if len(target) != len(reference):
        raise MetricContractError(f'common reserve target/reference row counts differ: {len(target)} != {len(reference)}')
    if len(target) == 0:
        raise MetricContractError('common reserve target/reference traces are empty')
    target_run_id = _one_text_value(target, 'run_id', role='target trace')
    reference_run_id = _one_text_value(reference, 'run_id', role='reserve reference')
    reference_arm = _one_text_value(reference, 'arm_id', role='reserve reference').upper()
    reference_scheduler = _one_text_value(reference, 'scheduler_id', role='reserve reference').upper()
    reference_calibration = _one_text_value(reference, 'calibration_id', role='reserve reference').upper()
    if reference_arm != 'GS_F0' or reference_scheduler != 'GS' or reference_calibration != 'STATIC':
        raise MetricContractError(f'common reserve reference must be GS_F0/GS/STATIC, got {reference_arm}/{reference_scheduler}/{reference_calibration}')
    target_input_sha256 = _validate_sha256(_one_text_value(target, 'input_sha256', role='target trace'), label='target input_sha256')
    reference_input_sha256 = _validate_sha256(_one_text_value(reference, 'input_sha256', role='reserve reference'), label='reference input_sha256')
    if target_input_sha256 != reference_input_sha256:
        raise MetricContractError('common reserve target/reference input_sha256 values do not match')
    if reserve_reference_sha256 is not None:
        reference_file_sha256 = _validate_sha256(reserve_reference_sha256, label='reserve reference trace SHA-256')
    else:
        reference_file_sha256 = 'UNRECORDED'
    target_time = pd.DatetimeIndex(target['timestamp'])
    reference_time = pd.DatetimeIndex(reference['timestamp'])
    target_step = pd.to_numeric(target['step'], errors='raise').to_numpy(float)
    reference_step = pd.to_numeric(reference['step'], errors='raise').to_numpy(float)
    if not (np.all(np.isfinite(target_step)) and np.all(np.isfinite(reference_step)) and np.array_equal(target_step, np.round(target_step)) and np.array_equal(reference_step, np.round(reference_step))):
        raise MetricContractError('common reserve target/reference step keys must be finite integers')
    if target_time.has_duplicates or reference_time.has_duplicates or (not target_time.is_monotonic_increasing) or (not reference_time.is_monotonic_increasing):
        raise MetricContractError('common reserve target/reference timestamps must be unique and increasing')
    if not target_time.equals(reference_time) or not np.array_equal(target_step, reference_step):
        raise MetricContractError('common reserve target/reference timestamp and step keys must match exactly')
    if 'condition_id' in target and 'condition_id' in reference:
        target_condition = target['condition_id'].fillna('').astype(str).to_numpy()
        reference_condition = reference['condition_id'].fillna('').astype(str).to_numpy()
        if not np.array_equal(target_condition, reference_condition):
            raise MetricContractError('common reserve target/reference condition_id values must match exactly')
    if 'evaluation_eligible' in target and 'evaluation_eligible' in reference:
        if not np.array_equal(_evaluation_mask(target), _evaluation_mask(reference)):
            raise MetricContractError('common reserve target/reference evaluation eligibility must match exactly')
    ev_upper = pd.to_numeric(reference['ev_upper_kw'], errors='coerce').to_numpy(float)
    pv_lower = pd.to_numeric(reference['pv_lower_kw'], errors='coerce').to_numpy(float)
    soc_post = pd.to_numeric(target['soc_post'], errors='coerce').to_numpy(float)
    if not (np.all(np.isfinite(ev_upper)) and np.all(np.isfinite(pv_lower)) and np.all(np.isfinite(soc_post))):
        raise MetricContractError('common reserve bounds and target soc_post must be finite on every aligned step')
    if np.any(soc_post < soc_min - tolerance):
        first = int(np.flatnonzero(soc_post < soc_min - tolerance)[0])
        raise MetricContractError(f'target soc_post is below soc_min beyond tolerance at row {first}')
    adverse_net = np.maximum(ev_upper - pv_lower, 0.0)
    reserve_requirement = reserve_hours_eval * np.maximum(adverse_net - feeder_limit_kw, 0.0)
    available_energy = eta_discharge * battery_energy_kwh * np.maximum(soc_post - soc_min, 0.0)
    reserve_margin = available_energy - reserve_requirement
    headroom = np.maximum(reserve_margin, 0.0)
    shortage = np.maximum(-reserve_margin, 0.0)
    evaluation_eligible = _evaluation_mask(target)
    return pd.DataFrame({'timestamp': target_time, 'step': target_step.astype(int), 'target_run_id': target_run_id, 'reference_run_id': reference_run_id, 'target_input_sha256': target_input_sha256, 'reference_input_sha256': reference_input_sha256, 'reference_trace_sha256': reference_file_sha256, 'evaluation_eligible': evaluation_eligible, 'step_hours': float(step_hours), 'feeder_limit_kw': float(feeder_limit_kw), 'reserve_hours_eval': float(reserve_hours_eval), 'battery_energy_kwh': float(battery_energy_kwh), 'eta_discharge': float(eta_discharge), 'soc_min': float(soc_min), 'soc_post': soc_post, 'reference_ev_upper_kw': ev_upper, 'reference_pv_lower_kw': pv_lower, 'common_gs_adverse_net_kw': adverse_net, 'common_gs_reserve_requirement_kwh': reserve_requirement, 'common_gs_available_reserve_energy_kwh': available_energy, 'common_gs_reserve_headroom_kwh': headroom, 'common_gs_reserve_shortage_kwh': shortage, 'common_gs_reserve_shortage_active': shortage > tolerance})

def _common_gs_reserve_records(trace: pd.DataFrame, eligible: np.ndarray, *, reserve_reference_frame: pd.DataFrame | None, feeder_limit_kw: float, reserve_hours_eval: float | None, battery_energy_kwh: float | None, eta_discharge: float | None, soc_min: float | None, step_hours: float, tolerance: float, reserve_reference_sha256: str | None) -> list[MetricRecord]:
    missing: list[str] = []
    if reserve_reference_frame is None:
        missing.append('registered GS_F0 reserve reference trace')
    for name, value in (('reserve_hours_eval', reserve_hours_eval), ('battery_energy_kwh', battery_energy_kwh), ('eta_discharge', eta_discharge), ('soc_min', soc_min)):
        if value is None:
            missing.append(name)
    if missing:
        reason = 'common GS-style reserve is pending; missing ' + ', '.join(missing)
        return [_pending_record(metric_id, unit, reason) for metric_id, unit in COMMON_GS_RESERVE_METRIC_UNITS.items()]
    evidence = compute_common_gs_reserve_step_table(trace, reserve_reference_frame=reserve_reference_frame, feeder_limit_kw=feeder_limit_kw, reserve_hours_eval=float(reserve_hours_eval), battery_energy_kwh=float(battery_energy_kwh), eta_discharge=float(eta_discharge), soc_min=float(soc_min), step_hours=step_hours, tolerance=tolerance, reserve_reference_sha256=reserve_reference_sha256)
    requirement = _numeric(evidence, 'common_gs_reserve_requirement_kwh')
    available = _numeric(evidence, 'common_gs_available_reserve_energy_kwh')
    headroom = _numeric(evidence, 'common_gs_reserve_headroom_kwh')
    shortage = _numeric(evidence, 'common_gs_reserve_shortage_kwh')
    total = int(np.count_nonzero(eligible))
    active = eligible & (shortage > tolerance)
    active_steps = int(np.count_nonzero(active))
    active_hours = float(active_steps * step_hours)
    exposure = float(shortage[eligible].sum() * step_hours)
    total_hours = float(total * step_hours)
    reference_run_id = str(evidence['reference_run_id'].iloc[0])
    reference_trace_sha = str(evidence['reference_trace_sha256'].iloc[0])
    common_notes = f'Method-independent postprocess using the identical registered GS_F0/STATIC adverse bounds; reference_run_id={reference_run_id}; reference_trace_sha256={reference_trace_sha}.'
    peak_records: list[MetricRecord] = []
    for metric_id, values in (('common_gs_reserve_requirement_peak_kwh', requirement), ('common_gs_available_reserve_energy_peak_kwh', available), ('common_gs_reserve_headroom_peak_kwh', headroom), ('common_gs_reserve_shortage_peak_kwh', shortage)):
        peak_records.append(_record(metric_id, float(np.max(values[eligible])), 'kWh', None, total, 'all aligned pre-registered evaluation steps used by the maximum statistic', notes=common_notes))
    return peak_records + [_record('common_gs_reserve_shortage_active_step_count', active_steps, 'steps', active_steps, total, 'all aligned pre-registered evaluation steps', notes=common_notes), _record('common_gs_reserve_shortage_active_hours', active_hours, 'h', active_hours, total_hours, 'all aligned pre-registered evaluation hours', notes=common_notes), _record('common_gs_reserve_shortage_exposure_kwh_h', exposure, 'kWh*h', exposure, total_hours, 'time integral of common GS-style shortage over all evaluation hours', notes=common_notes)]

def _interval_records(trace: pd.DataFrame, eligible: np.ndarray, *, prefix: str, actual_col: str, lower_col: str, upper_col: str, tolerance: float, step_hours: float, rolling_window_hours: float) -> list[MetricRecord]:
    metric_units = {f'{prefix}_interval_coverage': 'fraction', f'{prefix}_interval_miss_rate': 'fraction', f'{prefix}_lower_miss_rate': 'fraction', f'{prefix}_upper_miss_rate': 'fraction', f'{prefix}_mpiw_kw': 'kW', f'{prefix}_pinaw': 'fraction', f'{prefix}_interval_width_median_kw': 'kW', f'{prefix}_interval_width_p95_kw': 'kW', f'{prefix}_max_rolling_miss_rate': 'fraction'}
    required = {actual_col, lower_col, upper_col}
    missing = sorted(required.difference(trace.columns))
    if missing:
        reason = f'missing interval trace columns: {missing}'
        return [_pending_record(metric_id, unit, reason) for metric_id, unit in metric_units.items()]
    actual = _numeric(trace, actual_col)
    lower = _numeric(trace, lower_col)
    upper = _numeric(trace, upper_col)
    finite_bounds = np.isfinite(lower) & np.isfinite(upper)
    if np.any(finite_bounds & (upper < lower - tolerance)):
        raise MetricContractError(f'{prefix} prediction interval has upper < lower')
    finite = np.isfinite(actual) & finite_bounds
    valid = eligible & finite
    covered = valid & (actual >= lower) & (actual <= upper)
    miss = valid & ~covered
    lower_miss = valid & (actual < lower)
    upper_miss = valid & (actual > upper)
    eligible_count = int(np.count_nonzero(eligible))
    valid_count = int(np.count_nonzero(valid))
    status = _validity_status(valid_count, eligible_count)
    width_valid = eligible & finite_bounds
    width_count = int(np.count_nonzero(width_valid))
    widths = np.maximum(upper[width_valid] - lower[width_valid], 0.0)
    width_status = _validity_status(width_count, eligible_count)
    mpiw = float(np.mean(widths)) if width_count else None
    records = [_coverage_record(f'{prefix}_interval_coverage', actual, lower, upper, eligible), _finite_rate_record(f'{prefix}_interval_miss_rate', miss, finite, eligible, 'eligible steps with finite actual and interval bounds'), _finite_rate_record(f'{prefix}_lower_miss_rate', lower_miss, finite, eligible, 'eligible steps with finite actual and interval bounds'), _finite_rate_record(f'{prefix}_upper_miss_rate', upper_miss, finite, eligible, 'eligible steps with finite actual and interval bounds'), _record(f'{prefix}_mpiw_kw', mpiw, 'kW', float(widths.sum()) if width_count else None, width_count, 'eligible steps with finite lower and upper bounds', excluded_count=eligible_count - width_count, notes='Mean prediction interval width (MPIW).', status=width_status), _record(f'{prefix}_interval_width_median_kw', float(np.quantile(widths, 0.5)) if width_count else None, 'kW', None, width_count, 'finite eligible interval widths used by the quantile statistic', excluded_count=eligible_count - width_count, status=width_status), _record(f'{prefix}_interval_width_p95_kw', float(np.quantile(widths, 0.95)) if width_count else None, 'kW', None, width_count, 'finite eligible interval widths used by the quantile statistic', excluded_count=eligible_count - width_count, status=width_status)]
    actual_range = float(np.max(actual[valid]) - np.min(actual[valid])) if valid_count else None
    if mpiw is None or actual_range is None:
        pinaw_status = 'PENDING'
        pinaw = None
    elif actual_range <= tolerance:
        pinaw_status = 'UNDEFINED'
        pinaw = None
    else:
        pinaw_status = status
        pinaw = mpiw / actual_range
    records.append(_record(f'{prefix}_pinaw', pinaw, 'fraction', mpiw, actual_range, 'MPIW divided by the finite eligible observed-actual range', excluded_count=eligible_count - valid_count, notes='PINAW is undefined when the observed range is zero.', status=pinaw_status))
    ratio = rolling_window_hours / step_hours
    window_steps = int(round(ratio))
    if rolling_window_hours <= 0 or window_steps < 1 or (not math.isclose(ratio, window_steps, rel_tol=0.0, abs_tol=1e-09)):
        raise MetricContractError('rolling window must be a positive integer number of steps')
    candidate_windows = max(len(trace) - window_steps + 1, 0)
    miss_counts: list[int] = []
    for start in range(candidate_windows):
        stop = start + window_steps
        if np.all(valid[start:stop] & eligible[start:stop]):
            miss_counts.append(int(np.count_nonzero(miss[start:stop])))
    if miss_counts:
        worst_misses = max(miss_counts)
        rolling_status = 'OK' if len(miss_counts) == candidate_windows else 'PARTIAL'
        rolling_value = worst_misses / window_steps
    else:
        worst_misses = None
        rolling_status = 'PENDING'
        rolling_value = None
    records.append(_record(f'{prefix}_max_rolling_miss_rate', rolling_value, 'fraction', worst_misses, window_steps, f'worst complete trailing {rolling_window_hours:g} h window', excluded_count=candidate_windows - len(miss_counts), notes=f'valid_windows={len(miss_counts)}; candidate_windows={candidate_windows}', status=rolling_status))
    return records

def _aci_records(trace: pd.DataFrame, forecast_frame: pd.DataFrame | None, *, alpha_min: float | None, alpha_max: float | None, tolerance: float) -> list[MetricRecord]:
    metric_ids = ('aci_alpha_clip_rate', 'aci_alpha_lower_clip_rate', 'aci_alpha_upper_clip_rate', 'aci_decision_any_alpha_clip_rate')
    calibration = trace.get('calibration_id', pd.Series('', index=trace.index, dtype=str)).astype(str)
    applicable = calibration.str.upper().str.contains('ACI').any()
    if not applicable:
        return [_record(metric_id, None, 'fraction', 0, 0, 'not applicable to a non-ACI trace', notes='Static conformal methods do not update alpha.', status='NOT_APPLICABLE') for metric_id in metric_ids]
    source = forecast_frame if forecast_frame is not None else trace
    missing = {'alpha_lower', 'alpha_upper'}.difference(source.columns)
    if missing:
        reason = f'ACI trace requires forecast alpha columns: {sorted(missing)}'
        return [_pending_record(metric_id, 'fraction', reason) for metric_id in metric_ids]
    if alpha_min is None or alpha_max is None:
        reason = 'ACI alpha clip bounds were not supplied and are absent from the step schema'
        return [_pending_record(metric_id, 'fraction', reason) for metric_id in metric_ids]
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    if not 0 < alpha_min < alpha_max < 1:
        raise MetricContractError('ACI alpha clip bounds are invalid')
    source = source.copy()
    if 'horizon_offset' in source:
        offsets = pd.to_numeric(source['horizon_offset'], errors='coerce')
        source = source.loc[offsets.eq(0)].copy()
    lower = pd.to_numeric(source['alpha_lower'], errors='coerce').to_numpy(float)
    upper = pd.to_numeric(source['alpha_upper'], errors='coerce').to_numpy(float)
    if np.any(np.isfinite(lower) & ((lower < alpha_min - tolerance) | (lower > alpha_max + tolerance))):
        raise MetricContractError('alpha_lower lies outside registered clip bounds')
    if np.any(np.isfinite(upper) & ((upper < alpha_min - tolerance) | (upper > alpha_max + tolerance))):
        raise MetricContractError('alpha_upper lies outside registered clip bounds')
    lower_valid = np.isfinite(lower)
    upper_valid = np.isfinite(upper)
    lower_clip = lower_valid & (np.isclose(lower, alpha_min, atol=tolerance, rtol=0.0) | np.isclose(lower, alpha_max, atol=tolerance, rtol=0.0))
    upper_clip = upper_valid & (np.isclose(upper, alpha_min, atol=tolerance, rtol=0.0) | np.isclose(upper, alpha_max, atol=tolerance, rtol=0.0))

    def clip_record(metric_id: str, events: np.ndarray, valid: np.ndarray) -> MetricRecord:
        denominator = int(np.count_nonzero(valid))
        numerator = int(np.count_nonzero(events))
        return _record(metric_id, numerator / denominator if denominator else None, 'fraction', numerator, denominator, 'finite horizon-offset-zero ACI alpha snapshots', excluded_count=len(valid) - denominator, status=_validity_status(denominator, len(valid)))
    combined_valid = np.r_[lower_valid, upper_valid]
    combined_clip = np.r_[lower_clip, upper_clip]
    records = [clip_record('aci_alpha_clip_rate', combined_clip, combined_valid), clip_record('aci_alpha_lower_clip_rate', lower_clip, lower_valid), clip_record('aci_alpha_upper_clip_rate', upper_clip, upper_valid)]
    if 'decision_step' in source:
        decision_keys = source['decision_step'].astype(str).to_numpy()
    elif 'decision_time' in source:
        decision_keys = source['decision_time'].astype(str).to_numpy()
    else:
        decision_keys = np.arange(len(source)).astype(str)
    decision_frame = pd.DataFrame({'decision': decision_keys, 'valid': lower_valid & upper_valid, 'clipped': lower_clip | upper_clip})
    grouped = decision_frame.groupby('decision', sort=False).agg(valid=('valid', 'all'), clipped=('clipped', 'any'))
    valid_decisions = grouped['valid'].to_numpy(bool)
    clipped_decisions = grouped['clipped'].to_numpy(bool) & valid_decisions
    denominator = int(np.count_nonzero(valid_decisions))
    numerator = int(np.count_nonzero(clipped_decisions))
    records.append(_record('aci_decision_any_alpha_clip_rate', numerator / denominator if denominator else None, 'fraction', numerator, denominator, 'decision steps whose horizon-offset-zero EV/PV alpha snapshots are finite', excluded_count=len(grouped) - denominator, status=_validity_status(denominator, len(grouped))))
    return records

def infer_step_hours(frame: pd.DataFrame) -> float:
    if 'step_hours' in frame:
        unique = pd.to_numeric(frame['step_hours'], errors='raise').unique()
        if len(unique) != 1 or unique[0] <= 0:
            raise MetricContractError('step_hours must be one positive constant')
        return float(unique[0])
    if 'timestamp' not in frame or len(frame) < 2:
        return 1.0
    values = pd.DatetimeIndex(frame['timestamp']).sort_values().unique()
    deltas = np.diff(values).astype('timedelta64[ns]').astype(np.int64).astype(float) / 3600000000000.0
    if len(deltas) == 0:
        return 1.0
    if not np.allclose(deltas, deltas[0], atol=1e-12, rtol=0.0) or deltas[0] <= 0:
        raise MetricContractError('cannot infer one regular time step from timestamps')
    return float(deltas[0])

def compute_trace_metrics(frame: pd.DataFrame, *, forecast_frame: pd.DataFrame | None=None, reserve_reference_frame: pd.DataFrame | None=None, feeder_limit_kw: float=200.0, tolerance: float=1e-09, step_hours: float | None=None, soc_min: float | None=None, soc_max: float | None=None, reserve_hours_eval: float | None=None, battery_energy_kwh: float | None=None, eta_discharge: float | None=None, reserve_reference_sha256: str | None=None, aci_alpha_min: float | None=None, aci_alpha_max: float | None=None, rolling_miss_window_hours: float=24.0, lambda_eval_yuan_per_kwh: float | None=None) -> pd.DataFrame:
    if feeder_limit_kw <= 0 or tolerance < 0 or rolling_miss_window_hours <= 0:
        raise MetricContractError('feeder limit must be positive and tolerance nonnegative')
    trace = normalize_trace_schema(frame)
    eligible = _evaluation_mask(trace)
    total = int(np.count_nonzero(eligible))
    if total == 0:
        raise MetricContractError('trace contains no evaluation-eligible rows')
    inferred_dt = infer_step_hours(trace)
    dt = inferred_dt if step_hours is None else float(step_hours)
    if dt <= 0 or not math.isfinite(dt):
        raise MetricContractError('step_hours must be finite and positive')
    if step_hours is not None and len(trace) >= 2 and (not math.isclose(dt, inferred_dt, rel_tol=0.0, abs_tol=max(tolerance, 1e-12))):
        raise MetricContractError(f'declared step_hours={dt} conflicts with timestamp interval {inferred_dt}')
    if lambda_eval_yuan_per_kwh is not None and (not math.isfinite(lambda_eval_yuan_per_kwh) or lambda_eval_yuan_per_kwh < 0):
        raise MetricContractError('lambda_eval_yuan_per_kwh must be finite and nonnegative')
    resolved_soc_min = _resolve_constant(trace, 'soc_min', soc_min, tolerance=tolerance)
    resolved_soc_max = _resolve_constant(trace, 'soc_max', soc_max, tolerance=tolerance)
    resolved_reserve_hours = _resolve_constant(trace, 'reserve_hours_eval', reserve_hours_eval, tolerance=tolerance)
    resolved_battery_energy = _resolve_constant(trace, 'battery_energy_kwh', battery_energy_kwh, tolerance=tolerance)
    resolved_eta_discharge = _resolve_constant(trace, 'eta_discharge', eta_discharge, tolerance=tolerance)
    nominal = _numeric(trace, 'grid_import_nom_kw')
    executed = _numeric(trace, 'grid_import_exec_kw')
    delta_charge = _numeric(trace, 'delta_charge_kw')
    delta_discharge = _numeric(trace, 'delta_discharge_kw')
    invalid_exec = ~np.isfinite(executed)
    pressure = np.isfinite(nominal) & (nominal > feeder_limit_kw + tolerance)
    correction = np.isfinite(delta_charge) & np.isfinite(delta_discharge) & ((np.abs(delta_charge) > tolerance) | (np.abs(delta_discharge) > tolerance))
    if 'filter_triggered' in trace:
        triggered = _strict_bool_mask(trace, 'filter_triggered')
    elif 'filter_active' in trace:
        triggered = _strict_bool_mask(trace, 'filter_active')
    elif 'filter_id' in trace:
        enabled = trace['filter_id'].astype(str).str.casefold().ne('none').to_numpy()
        triggered = enabled & pressure
    else:
        triggered = correction.copy()
    if np.any(correction & ~triggered):
        raise MetricContractError('trace contains a correction without a filter trigger')
    if 'filter_correction_applied' in trace:
        explicit_correction = _strict_bool_mask(trace, 'filter_correction_applied')
        if not np.array_equal(explicit_correction, correction):
            raise MetricContractError('filter_correction_applied disagrees with correction deltas')
    if 'filter_capacity_limited_residual' in trace:
        capacity_limited_residual = _strict_bool_mask(trace, 'filter_capacity_limited_residual')
    else:
        capacity_limited_residual = triggered & np.isfinite(executed) & (executed > feeder_limit_kw + tolerance)
    if 'filter_capacity_limited_zero_correction' in trace:
        capacity_limited_zero = _strict_bool_mask(trace, 'filter_capacity_limited_zero_correction')
    else:
        capacity_limited_zero = capacity_limited_residual & ~correction
    unsafe = invalid_exec | (executed > feeder_limit_kw + tolerance)
    closed = np.isfinite(executed) & (executed <= feeder_limit_kw + tolerance)
    records: list[MetricRecord] = [_record('evaluation_step_count', total, 'steps', total, total, 'all pre-registered evaluation steps'), _rate_record('prefilter_pressure_rate', pressure, eligible, 'all pre-registered evaluation steps, including solver failures'), _rate_record('filter_activation_rate', triggered, eligible, 'all pre-registered evaluation steps, including safe and failed steps', notes='compatibility alias for filter_trigger_rate; activation means trigger'), _rate_record('filter_trigger_rate', triggered, eligible, 'all pre-registered evaluation steps, including safe and failed steps'), _rate_record('filter_correction_rate', correction, eligible, 'all pre-registered evaluation steps, including safe and failed steps'), _rate_record('filter_capacity_limited_zero_correction_rate', capacity_limited_zero, eligible, 'all pre-registered evaluation steps, including safe and failed steps'), _rate_record('filter_capacity_limited_residual_rate', capacity_limited_residual, eligible, 'all pre-registered evaluation steps, including safe and failed steps'), _rate_record('grid_safety_violation_rate', unsafe, eligible, 'all pre-registered evaluation steps; missing executed import counts as unsafe'), _rate_record('missing_executed_import_rate', invalid_exec, eligible, 'all pre-registered evaluation steps')]
    conditional_eligible = eligible & pressure
    conditional_denominator = int(np.count_nonzero(conditional_eligible))
    conditional_numerator = int(np.count_nonzero(conditional_eligible & closed))
    records.append(_record('conditional_filter_closure_rate', conditional_numerator / conditional_denominator if conditional_denominator else None, 'fraction', conditional_numerator, conditional_denominator, 'evaluation steps whose nominal import exceeds the feeder limit', notes='Undefined rather than zero when no nominal-pressure event occurs.'))
    records.extend([_peak_record('peak_grid_import_nom_kw', nominal, eligible, notes='Pre-filter/nominal grid-import peak.'), _peak_record('peak_grid_import_exec_kw', executed, eligible, notes='Post-filter/executed grid-import peak.')])
    prefilter_duration = _over_limit_records('prefilter', nominal, eligible, feeder_limit_kw, tolerance, dt)
    postfilter_duration = _over_limit_records('postfilter', executed, eligible, feeder_limit_kw, tolerance, dt)
    records.extend(prefilter_duration)
    records.extend(postfilter_duration)
    post_energy = next((record for record in postfilter_duration if record.metric_id == 'postfilter_over_limit_energy_kwh'))
    post_peak = next((record for record in records if record.metric_id == 'peak_grid_import_exec_kw'))
    records.extend([_record('grid_over_limit_energy_kwh', post_energy.value, 'kWh', post_energy.numerator, post_energy.denominator, post_energy.denominator_definition, excluded_count=post_energy.excluded_count, notes='Compatibility alias of postfilter_over_limit_energy_kwh.', status=post_energy.status), _record('peak_grid_import_kw', post_peak.value, 'kW', post_peak.numerator, post_peak.denominator, post_peak.denominator_definition, excluded_count=post_peak.excluded_count, notes='Compatibility alias of peak_grid_import_exec_kw.', status=post_peak.status)])
    if 'solver_status' in trace:
        success = _success_mask(trace['solver_status'])
        records.append(_rate_record('solver_success_rate', success, eligible, 'all pre-registered evaluation steps'))
    else:
        records.append(_pending_record('solver_success_rate', 'fraction', 'missing solver_status column'))
    latency_metrics = (('mean_solve_time_s', 'mean', None), ('median_solve_time_s', 'quantile', 0.5), ('p95_solve_time_s', 'quantile', 0.95), ('p99_solve_time_s', 'quantile', 0.99), ('max_solve_time_s', 'max', None))
    if 'solve_time_s' not in trace:
        records.extend((_pending_record(metric_id, 's', 'missing solve_time_s column') for metric_id, _, _ in latency_metrics))
    else:
        solve_time = pd.to_numeric(trace['solve_time_s'], errors='coerce').to_numpy(float)
        if np.any(np.isfinite(solve_time) & (solve_time < -tolerance)):
            raise MetricContractError('solve_time_s contains a negative latency')
        valid = eligible & np.isfinite(solve_time) & (solve_time >= 0.0)
        valid_count = int(np.count_nonzero(valid))
        status = _validity_status(valid_count, total)
        values = solve_time[valid]
        for metric_id, statistic, quantile in latency_metrics:
            if valid_count == 0:
                value = None
            elif statistic == 'mean':
                value = float(np.mean(values))
            elif statistic == 'max':
                value = float(np.max(values))
            else:
                value = float(np.quantile(values, float(quantile)))
            records.append(_record(metric_id, value, 's', float(values.sum()) if statistic == 'mean' and valid_count else None, valid_count, 'finite nonnegative eligible solver latencies', excluded_count=total - valid_count, notes='Quantiles use NumPy linear interpolation on per-step wall-clock solve time.', status=status))
    if 'mip_gap' not in trace:
        records.append(_pending_record('mean_reported_mip_gap', 'fraction', 'missing mip_gap column'))
    else:
        mip_gap = pd.to_numeric(trace['mip_gap'], errors='coerce').to_numpy(float)
        valid = eligible & np.isfinite(mip_gap)
        valid_count = int(np.count_nonzero(valid))
        records.append(_record('mean_reported_mip_gap', float(mip_gap[valid].mean()) if valid_count else None, 'fraction', float(mip_gap[valid].sum()) if valid_count else None, valid_count, 'eligible steps with a finite reported MIP gap', excluded_count=total - valid_count, notes='Some optimal solvers legitimately omit a MIP gap; exclusions remain explicit.', status=_validity_status(valid_count, total)))
    reserve_records = _slack_records(trace, eligible, column='reserve_shortage_kwh', prefix='reserve_shortage', source_unit='kWh', step_hours=dt, tolerance=tolerance, total_notes='DEPRECATED: this sampling-rate-dependent sum adds per-step kWh deficit states without multiplying by duration; it is not an exposure integral and must not be cited. Use reserve_shortage_exposure_kwh_h instead.')
    barrier_records = _slack_records(trace, eligible, column='barrier_slack_kwh', prefix='barrier_slack', source_unit='kWh', step_hours=dt, tolerance=tolerance)
    feeder_records = _slack_records(trace, eligible, column='feeder_slack_kw', prefix='feeder_slack', source_unit='kW', step_hours=dt, tolerance=tolerance)
    records.extend(reserve_records)
    records.extend(barrier_records)
    records.extend(feeder_records)
    records.extend(_energy_deficit_duration_records(trace, eligible, column='reserve_shortage_kwh', prefix='reserve_shortage', step_hours=dt, tolerance=tolerance, notes='Controller-native reserve shortage; its requirement semantics differ across M6/RB/GS and it is not the cross-method comparator.'))
    records.extend(_common_gs_reserve_records(trace, eligible, reserve_reference_frame=reserve_reference_frame, feeder_limit_kw=feeder_limit_kw, reserve_hours_eval=resolved_reserve_hours, battery_energy_kwh=resolved_battery_energy, eta_discharge=resolved_eta_discharge, soc_min=resolved_soc_min, step_hours=dt, tolerance=tolerance, reserve_reference_sha256=reserve_reference_sha256))
    reserve_total = next((record for record in reserve_records if record.metric_id == 'reserve_shortage_total_kwh'))
    reserve_rate = next((record for record in reserve_records if record.metric_id == 'reserve_shortage_active_rate'))
    records.extend([_record('reserve_shortage_energy_kwh', reserve_total.value, 'kWh', reserve_total.numerator, reserve_total.denominator, reserve_total.denominator_definition, excluded_count=reserve_total.excluded_count, notes='DEPRECATED compatibility alias of reserve_shortage_total_kwh; this is a sampling-rate-dependent sum of per-step kWh deficit states, not exposure.', status=reserve_total.status), _record('reserve_shortage_step_rate', reserve_rate.value, 'fraction', reserve_rate.numerator, reserve_rate.denominator, reserve_rate.denominator_definition, excluded_count=reserve_rate.excluded_count, notes='Compatibility alias of reserve_shortage_active_rate.', status=reserve_rate.status)])
    throughput: float | None = None
    throughput_status = 'PENDING'
    if {'p_charge_exec_kw', 'p_discharge_exec_kw'}.issubset(trace.columns):
        charge = _numeric(trace, 'p_charge_exec_kw')
        discharge = _numeric(trace, 'p_discharge_exec_kw')
        mutual = ~np.isfinite(charge) | ~np.isfinite(discharge) | (charge > tolerance) & (discharge > tolerance)
        records.append(_rate_record('charge_discharge_mutex_violation_rate', mutual, eligible, 'all pre-registered evaluation steps; missing action counts as a violation'))
        valid = eligible & np.isfinite(charge) & np.isfinite(discharge)
        complete = int(np.count_nonzero(valid)) == total
        throughput = float(((charge[valid] + discharge[valid]) * dt).sum()) if complete else None
        throughput_status = 'OK' if complete else 'PENDING'
        records.append(_record('battery_throughput_kwh', throughput, 'kWh', throughput, total, 'all pre-registered evaluation steps', excluded_count=total - int(np.count_nonzero(valid)), notes='Undefined if any eligible action is missing.', status=throughput_status))
    else:
        missing_actions = sorted({'p_charge_exec_kw', 'p_discharge_exec_kw'}.difference(trace.columns))
        records.extend([_pending_record('charge_discharge_mutex_violation_rate', 'fraction', f'missing action columns: {missing_actions}'), _pending_record('battery_throughput_kwh', 'kWh', f'missing action columns: {missing_actions}')])
    grid_cost: float | None = None
    grid_cost_status = 'PENDING'
    if {'grid_import_exec_kw', 'price_yuan_per_kwh'}.issubset(trace.columns):
        price = _numeric(trace, 'price_yuan_per_kwh')
        valid = eligible & np.isfinite(executed) & np.isfinite(price)
        complete = int(np.count_nonzero(valid)) == total
        grid_cost = float((executed[valid] * price[valid] * dt).sum()) if complete else None
        grid_cost_status = 'OK' if complete else 'PENDING'
        for metric_id, notes in (('grid_energy_purchase_cost_yuan', 'Raw metered energy-purchase cost; excludes battery-throughput evaluation cost.'), ('grid_energy_cost_yuan', 'Compatibility alias of grid_energy_purchase_cost_yuan.')):
            records.append(_record(metric_id, grid_cost, 'Yuan', grid_cost, total, 'every pre-registered step must have finite import and price', excluded_count=total - int(np.count_nonzero(valid)), notes=notes, status=grid_cost_status))
    else:
        missing_cost = sorted({'grid_import_exec_kw', 'price_yuan_per_kwh'}.difference(trace.columns))
        for metric_id in ('grid_energy_purchase_cost_yuan', 'grid_energy_cost_yuan'):
            records.append(_pending_record(metric_id, 'Yuan', f'missing cost columns: {missing_cost}'))
    resolved_lambda = _resolve_constant(trace, 'lambda_eval_yuan_per_kwh', lambda_eval_yuan_per_kwh, tolerance=tolerance)
    if resolved_lambda is None:
        records.extend([_pending_record('lambda_eval_yuan_per_kwh', 'Yuan/kWh', 'supply --lambda-eval-yuan-per-kwh or a constant trace column'), _pending_record('throughput_cost_at_lambda_eval_yuan', 'Yuan', 'lambda_eval is unavailable'), _pending_record('comparable_operating_cost_at_lambda_eval_yuan', 'Yuan', 'lambda_eval is unavailable')])
    else:
        if resolved_lambda < 0:
            raise MetricContractError('lambda_eval_yuan_per_kwh cannot be negative')
        records.append(_record('lambda_eval_yuan_per_kwh', resolved_lambda, 'Yuan/kWh', resolved_lambda, 1, 'one frozen postprocess evaluation coefficient'))
        throughput_cost = throughput * resolved_lambda if throughput is not None else None
        comparable_cost = grid_cost + throughput_cost if grid_cost is not None and throughput_cost is not None else None
        comparable_status = 'OK' if grid_cost_status == 'OK' and throughput_status == 'OK' else 'PENDING'
        records.extend([_record('throughput_cost_at_lambda_eval_yuan', throughput_cost, 'Yuan', throughput_cost, total, 'frozen trajectory throughput multiplied by lambda_eval', notes='Postprocess only; it does not change scheduler actions.', status='OK' if throughput_cost is not None else 'PENDING'), _record('comparable_operating_cost_at_lambda_eval_yuan', comparable_cost, 'Yuan', comparable_cost, total, 'grid energy purchase cost plus lambda_eval times battery throughput', notes='Kept separate from the raw grid-energy cost and the optimization objective.', status=comparable_status)])
    coverage_triplets = (('ev', 'ev_actual_kw', 'ev_lower_kw', 'ev_upper_kw'), ('pv', 'pv_actual_kw', 'pv_lower_kw', 'pv_upper_kw'))
    for prefix, actual_col, lower_col, upper_col in coverage_triplets:
        records.extend(_interval_records(trace, eligible, prefix=prefix, actual_col=actual_col, lower_col=lower_col, upper_col=upper_col, tolerance=tolerance, step_hours=dt, rolling_window_hours=rolling_miss_window_hours))
    soc_metric_units = {'soc_lower_bound_touch_rate': 'fraction', 'soc_upper_bound_touch_rate': 'fraction', 'soc_lower_bound_violation_rate': 'fraction', 'soc_upper_bound_violation_rate': 'fraction', 'soc_observed_min': 'soc', 'soc_observed_max': 'soc'}
    if 'soc_post' not in trace or resolved_soc_min is None or resolved_soc_max is None:
        missing_soc: list[str] = []
        if 'soc_post' not in trace:
            missing_soc.append('soc_post')
        if resolved_soc_min is None:
            missing_soc.append('soc_min')
        if resolved_soc_max is None:
            missing_soc.append('soc_max')
        records.extend((_pending_record(metric_id, unit, f'missing SOC evidence: {missing_soc}') for metric_id, unit in soc_metric_units.items()))
    else:
        if not 0 <= resolved_soc_min < resolved_soc_max <= 1:
            raise MetricContractError('SOC bounds must satisfy 0 <= min < max <= 1')
        soc = pd.to_numeric(trace['soc_post'], errors='coerce').to_numpy(float)
        valid_soc = np.isfinite(soc)
        denominator_mask = eligible & valid_soc
        denominator = int(np.count_nonzero(denominator_mask))
        soc_status = _validity_status(denominator, total)
        for metric_id, events in (('soc_lower_bound_touch_rate', valid_soc & np.isclose(soc, resolved_soc_min, atol=tolerance, rtol=0.0)), ('soc_upper_bound_touch_rate', valid_soc & np.isclose(soc, resolved_soc_max, atol=tolerance, rtol=0.0)), ('soc_lower_bound_violation_rate', valid_soc & (soc < resolved_soc_min - tolerance)), ('soc_upper_bound_violation_rate', valid_soc & (soc > resolved_soc_max + tolerance))):
            records.append(_finite_rate_record(metric_id, events, valid_soc, eligible, 'eligible steps with finite post-action SOC', notes='SOC bound checks use soc_post once per closed-loop step.'))
        records.extend([_record('soc_observed_min', float(np.min(soc[denominator_mask])) if denominator else None, 'soc', None, denominator, 'finite eligible post-action SOC observations', excluded_count=total - denominator, status=soc_status), _record('soc_observed_max', float(np.max(soc[denominator_mask])) if denominator else None, 'soc', None, denominator, 'finite eligible post-action SOC observations', excluded_count=total - denominator, status=soc_status)])
    records.extend(_aci_records(trace, forecast_frame, alpha_min=aci_alpha_min, alpha_max=aci_alpha_max, tolerance=max(tolerance, 1e-12)))
    metrics = pd.DataFrame([asdict(record) for record in records])
    duplicate_ids = metrics.loc[metrics['metric_id'].duplicated(), 'metric_id'].tolist()
    if duplicate_ids:
        raise MetricContractError(f'duplicate metric IDs: {duplicate_ids}')
    missing_endpoint_metrics = REQUIRED_ENDPOINT_METRIC_IDS.difference(metrics['metric_id'])
    if missing_endpoint_metrics:
        raise MetricContractError(f'endpoint metric catalog is incomplete: {sorted(missing_endpoint_metrics)}')
    return metrics

def assert_metric_denominators(metrics: pd.DataFrame, expected_evaluation_steps: int) -> None:
    required_full = {'prefilter_pressure_rate', 'filter_activation_rate', 'filter_trigger_rate', 'filter_correction_rate', 'filter_capacity_limited_zero_correction_rate', 'filter_capacity_limited_residual_rate', 'grid_safety_violation_rate', 'missing_executed_import_rate'}
    rows = metrics.set_index('metric_id')
    missing = required_full.difference(rows.index)
    if missing:
        raise MetricContractError(f'missing full-denominator metrics: {sorted(missing)}')
    for metric_id in required_full:
        observed = rows.loc[metric_id, 'denominator']
        if not math.isclose(float(observed), float(expected_evaluation_steps), abs_tol=0.0):
            raise MetricContractError(f'{metric_id} denominator {observed} != pre-registered steps {expected_evaluation_steps}')
    conditional = rows.loc['conditional_filter_closure_rate']
    pressure = rows.loc['prefilter_pressure_rate']
    if not math.isclose(float(conditional['denominator']), float(pressure['numerator']), abs_tol=0.0):
        raise MetricContractError('conditional closure denominator must equal pressure-event count')

def _unique_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        return None
    return str(values[0])

def _allclose_mismatch(left: pd.Series, right: pd.Series, tolerance: float) -> np.ndarray:
    left_num = pd.to_numeric(left, errors='coerce').to_numpy(float)
    right_num = pd.to_numeric(right, errors='coerce').to_numpy(float)
    both_nan = np.isnan(left_num) & np.isnan(right_num)
    return ~(both_nan | np.isclose(left_num, right_num, atol=tolerance, rtol=0.0, equal_nan=False))

def _text_mismatch(left: pd.Series, right: pd.Series) -> np.ndarray:
    return left.fillna('<NA>').astype(str).to_numpy() != right.fillna('<NA>').astype(str).to_numpy()

def _first_mismatch_payload(field: str, mismatch: np.ndarray, left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object] | None:
    indices = np.flatnonzero(mismatch)
    if len(indices) == 0:
        return None
    index = int(indices[0])
    payload: dict[str, object] = {'row': index, 'left': _json_scalar(left.iloc[index][field]), 'right': _json_scalar(right.iloc[index][field])}
    if 'step' in left:
        payload['step'] = _json_scalar(left.iloc[index]['step'])
    if 'timestamp' in left:
        payload['timestamp'] = str(left.iloc[index]['timestamp'])
    return payload

def _json_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=' ')
    return value

# EN: Execute or verify paired `F0` (uncorrected baseline) and `F1` (REF-corrected) runs on identical inputs and forecasts.
# 中文：在完全一致的输入与预测序列上运行或核验 `F0`（无修正基准臂）与 `F1`（REF 修正臂）配对闭环对比。
def audit_pairing(left_frame: pd.DataFrame, right_frame: pd.DataFrame, *, mode: str='closed_loop', tolerance: float=1e-09, left_input_hash: str | None=None, right_input_hash: str | None=None, require_activation: bool=True, require_external_hash: bool=True, certificate_schema_version: int=TRACE_SCHEMA_VERSION) -> PairingAuditResult:
    if mode not in {'closed_loop', 'shadow'}:
        raise MetricContractError("pairing mode must be 'closed_loop' or 'shadow'")
    if tolerance < 0:
        raise MetricContractError('pairing tolerance must be nonnegative')
    if certificate_schema_version not in {1, 2}:
        raise MetricContractError('pairing certificate schema must be 1 or 2')
    left = normalize_trace_schema(left_frame)
    right = normalize_trace_schema(right_frame)
    notes: list[str] = []
    mismatch_counts: dict[str, int] = {}
    first_mismatches: dict[str, dict[str, object]] = {}
    row_count_match = len(left) == len(right)
    common_rows = min(len(left), len(right))
    left = left.iloc[:common_rows].reset_index(drop=True)
    right = right.iloc[:common_rows].reset_index(drop=True)
    if not row_count_match:
        mismatch_counts['row_count'] = abs(len(left_frame) - len(right_frame))
        first_mismatches['row_count'] = {'left': len(left_frame), 'right': len(right_frame)}
    timestamps_match = 'timestamp' in left and 'timestamp' in right
    if timestamps_match:
        mismatch = _text_mismatch(left['timestamp'], right['timestamp'])
        mismatch_counts['timestamp'] = int(np.count_nonzero(mismatch))
        payload = _first_mismatch_payload('timestamp', mismatch, left, right)
        if payload:
            first_mismatches['timestamp'] = payload
        timestamps_match = not mismatch.any()
    else:
        notes.append('timestamp column absent from one or both traces')
    scheduler_left = _unique_text(left, 'scheduler_id')
    scheduler_right = _unique_text(right, 'scheduler_id')
    scheduler_match = scheduler_left is not None and scheduler_left == scheduler_right
    if not scheduler_match:
        mismatch_counts['scheduler_id'] = common_rows
        first_mismatches['scheduler_id'] = {'left': scheduler_left, 'right': scheduler_right}
    external_hashes_match = True
    external_hash_evidence_present = False
    if left_input_hash is not None or right_input_hash is not None:
        external_hash_evidence_present = left_input_hash is not None and right_input_hash is not None
        explicit_hashes_match = external_hash_evidence_present and str(left_input_hash).casefold() == str(right_input_hash).casefold()
        external_hashes_match = external_hashes_match and explicit_hashes_match
        if not explicit_hashes_match:
            mismatch_counts['external_input_hash'] = 1
            first_mismatches['external_input_hash'] = {'left': left_input_hash, 'right': right_input_hash}
    for column in HASH_COLUMNS:
        in_left = column in left
        in_right = column in right
        if in_left and in_right:
            external_hash_evidence_present = True
            mismatch = _text_mismatch(left[column], right[column])
            mismatch_counts[column] = int(np.count_nonzero(mismatch))
            payload = _first_mismatch_payload(column, mismatch, left, right)
            if payload:
                first_mismatches[column] = payload
            external_hashes_match = external_hashes_match and (not mismatch.any())
        elif in_left or in_right:
            external_hashes_match = False
            key = f'missing_pair_{column}'
            mismatch_counts[key] = 1
            first_mismatches[key] = {'left_present': in_left, 'right_present': in_right}
    if not external_hash_evidence_present:
        notes.append('no comparable external-input hash evidence is present')
        if require_external_hash:
            external_hashes_match = False
            mismatch_counts['missing_external_hash_evidence'] = 1
            first_mismatches['missing_external_hash_evidence'] = {'required': True, 'accepted_sources': ['paired input files', *HASH_COLUMNS]}
    exogenous_match = True
    for canonical in EXOGENOUS_ALIASES:
        in_left = canonical in left
        in_right = canonical in right
        if not in_left and (not in_right):
            if canonical in REQUIRED_EXOGENOUS_FIELDS:
                exogenous_match = False
                key = f'missing_exogenous_{canonical}'
                mismatch_counts[key] = 1
                first_mismatches[key] = {'left_present': False, 'right_present': False}
            continue
        if not in_left or not in_right:
            exogenous_match = False
            key = f'missing_pair_{canonical}'
            mismatch_counts[key] = 1
            first_mismatches[key] = {'left_present': in_left, 'right_present': in_right}
            continue
        mismatch = _allclose_mismatch(left[canonical], right[canonical], tolerance)
        mismatch_counts[canonical] = int(np.count_nonzero(mismatch))
        payload = _first_mismatch_payload(canonical, mismatch, left, right)
        if payload:
            first_mismatches[canonical] = payload
        exogenous_match = exogenous_match and (not mismatch.any())
    required_pair_fields = ('soc_pre', 'p_charge_nom_kw', 'p_discharge_nom_kw', 'net_exchange_nom_kw', 'grid_import_nom_kw')
    missing_required = [field for field in required_pair_fields if field not in left or field not in right]
    if missing_required:
        raise MetricContractError(f'pair audit missing nominal fields: {missing_required}')
    if 'delta_charge_kw' not in right or 'delta_discharge_kw' not in right:
        raise MetricContractError('right/filter-on trace lacks correction columns')
    delta_charge = pd.to_numeric(right['delta_charge_kw'], errors='coerce').to_numpy(float)
    delta_discharge = pd.to_numeric(right['delta_discharge_kw'], errors='coerce').to_numpy(float)
    invalid_corrections = ~np.isfinite(delta_charge) | ~np.isfinite(delta_discharge)
    mismatch_counts['right_nonfinite_corrections'] = int(np.count_nonzero(invalid_corrections))
    if invalid_corrections.any():
        index = int(np.flatnonzero(invalid_corrections)[0])
        first_mismatches['right_nonfinite_corrections'] = {'row': index, 'step': _json_scalar(right.iloc[index]['step']), 'delta_charge_kw': _json_scalar(right.iloc[index]['delta_charge_kw']), 'delta_discharge_kw': _json_scalar(right.iloc[index]['delta_discharge_kw'])}
    corrections = np.flatnonzero(np.isfinite(delta_charge) & np.isfinite(delta_discharge) & ((np.abs(delta_charge) > tolerance) | (np.abs(delta_discharge) > tolerance)))
    correction_index = int(corrections[0]) if len(corrections) else None
    if 'filter_triggered' in right:
        trigger_mask = _strict_bool_mask(right, 'filter_triggered')
    elif 'filter_active' in right:
        trigger_mask = _strict_bool_mask(right, 'filter_active')
    else:
        filter_enabled = right['filter_id'].astype(str).str.casefold().ne('none').to_numpy() if 'filter_id' in right else np.ones(common_rows, dtype=bool)
        feeder_limit = pd.to_numeric(right['feeder_safe_kw'], errors='coerce').to_numpy(float) if 'feeder_safe_kw' in right else np.full(common_rows, 200.0)
        nominal_import = pd.to_numeric(right['grid_import_nom_kw'], errors='coerce').to_numpy(float)
        trigger_mask = filter_enabled & np.isfinite(feeder_limit) & np.isfinite(nominal_import) & (nominal_import > feeder_limit + tolerance)
    correction_mask = np.zeros(common_rows, dtype=bool)
    correction_mask[corrections] = True
    correction_without_trigger = correction_mask & ~trigger_mask
    correction_without_trigger_count = int(np.count_nonzero(correction_without_trigger))
    if certificate_schema_version >= 2 or correction_without_trigger_count:
        mismatch_counts['right_correction_without_trigger'] = correction_without_trigger_count
    if correction_without_trigger.any():
        index = int(np.flatnonzero(correction_without_trigger)[0])
        first_mismatches['right_correction_without_trigger'] = {'row': index, 'step': _json_scalar(right.iloc[index]['step'])}
    trigger_indexes = np.flatnonzero(trigger_mask)
    trigger_index = int(trigger_indexes[0]) if len(trigger_indexes) else None
    if mode == 'shadow':
        compare_rows = common_rows
    elif correction_index is None:
        compare_rows = common_rows
    else:
        compare_rows = correction_index + 1
    plant_pair_fields = ('plant_charge_kw', 'plant_discharge_kw', 'plant_net_exchange_kw', 'plant_grid_kw', 'plant_curtail_kw', 'common_discharge_clip_kw')
    compared_fields = list(required_pair_fields)
    for field in plant_pair_fields:
        if field in left or field in right:
            if field not in left or field not in right:
                mismatch_counts[f'missing_pair_{field}'] = 1
                first_mismatches[f'missing_pair_{field}'] = {'left_present': field in left, 'right_present': field in right}
            else:
                compared_fields.append(field)
    for field in compared_fields:
        mismatch = _allclose_mismatch(left.loc[:compare_rows - 1, field], right.loc[:compare_rows - 1, field], tolerance) if compare_rows else np.zeros(0, dtype=bool)
        key = f'paired_{field}'
        mismatch_counts[key] = int(np.count_nonzero(mismatch))
        if mismatch.any():
            payload = _first_mismatch_payload(field, np.pad(mismatch, (0, common_rows - len(mismatch))), left, right)
            if payload:
                first_mismatches[key] = payload
    if 'delta_charge_kw' in left and 'delta_discharge_kw' in left:
        left_delta_charge = pd.to_numeric(left['delta_charge_kw'], errors='coerce').to_numpy(float)
        left_delta_discharge = pd.to_numeric(left['delta_discharge_kw'], errors='coerce').to_numpy(float)
        mismatch = ~np.isfinite(left_delta_charge) | ~np.isfinite(left_delta_discharge) | (np.abs(left_delta_charge) > tolerance) | (np.abs(left_delta_discharge) > tolerance)
        mismatch_counts['left_filter_off_corrections'] = int(np.count_nonzero(mismatch))
        if mismatch.any():
            index = int(np.flatnonzero(mismatch)[0])
            first_mismatches['left_filter_off_corrections'] = {'row': index, 'step': _json_scalar(left.iloc[index]['step']), 'delta_charge_kw': _json_scalar(left.iloc[index]['delta_charge_kw']), 'delta_discharge_kw': _json_scalar(left.iloc[index]['delta_discharge_kw'])}
    post_divergence = 0
    if correction_index is not None and 'soc_pre' in left and (correction_index + 1 < common_rows):
        mismatch = _allclose_mismatch(left.loc[correction_index + 1:, 'soc_pre'], right.loc[correction_index + 1:, 'soc_pre'], tolerance)
        post_divergence = int(np.count_nonzero(mismatch))
        if post_divergence:
            notes.append('post-activation state divergence is reported but is not a closed-loop pairing failure')

    def event_at(index: int | None) -> tuple[int | None, str | None, dict[str, object] | None]:
        if index is None:
            return (None, None, None)
        step_value = int(left.iloc[index]['step'])
        timestamp_value = str(left.iloc[index]['timestamp']) if 'timestamp' in left else None
        return (step_value, timestamp_value, {'step': step_value, 'timestamp': timestamp_value, 'left_grid_import_nom_kw': _json_scalar(left.iloc[index]['grid_import_nom_kw']), 'right_grid_import_nom_kw': _json_scalar(right.iloc[index]['grid_import_nom_kw']), 'right_delta_charge_kw': _json_scalar(right.iloc[index]['delta_charge_kw']), 'right_delta_discharge_kw': _json_scalar(right.iloc[index]['delta_discharge_kw']), 'right_grid_import_exec_kw': _json_scalar(right.iloc[index].get('grid_import_exec_kw')), 'left_soc_pre': _json_scalar(left.iloc[index]['soc_pre']), 'right_soc_pre': _json_scalar(right.iloc[index]['soc_pre'])})
    trigger_step, trigger_timestamp, trigger_snapshot = event_at(trigger_index)
    correction_step, correction_timestamp, correction_snapshot = event_at(correction_index)
    activation_step = correction_step
    activation_timestamp = correction_timestamp
    event_snapshot = correction_snapshot
    if correction_index is None:
        if trigger_index is not None:
            notes.append('filter triggered but applied no nonzero correction; full nominal/state trajectory remains paired')
    if correction_index is None and require_activation:
        mismatch_counts['required_filter_activation'] = 1
        first_mismatches['required_filter_activation'] = {'left': None, 'right': 'no correction above tolerance'}
    fatal_keys = {key for key, count in mismatch_counts.items() if count and (not (mode == 'closed_loop' and key.startswith('post_activation_')))}
    if not row_count_match:
        fatal_keys.add('row_count')
    if not timestamps_match:
        fatal_keys.add('timestamp')
    if not scheduler_match:
        fatal_keys.add('scheduler_id')
    if not external_hashes_match:
        fatal_keys.add('external_input_hash')
    if not exogenous_match:
        fatal_keys.add('exogenous')
    verdict = 'PASS' if not fatal_keys else 'FAIL'
    if verdict == 'PASS' and correction_index is None:
        verdict = 'PASS_NO_ACTIVATION'
        notes.append('pair is identical, but this trace does not exercise a filter intervention')
    return PairingAuditResult(schema_version=certificate_schema_version, verdict=verdict, mode=mode, tolerance=float(tolerance), row_count_left=len(left_frame), row_count_right=len(right_frame), external_hash_evidence_present=external_hash_evidence_present, external_hashes_match=external_hashes_match, timestamps_match=timestamps_match, scheduler_match=scheduler_match, exogenous_match=exogenous_match, first_trigger_step=trigger_step, first_trigger_timestamp=trigger_timestamp, first_correction_step=correction_step, first_correction_timestamp=correction_timestamp, first_activation_step=activation_step, first_activation_timestamp=activation_timestamp, compared_prefix_rows=compare_rows, mismatch_counts=mismatch_counts, first_mismatches=first_mismatches, post_correction_state_divergence_rows=post_divergence, post_activation_state_divergence_rows=post_divergence, trigger_event_snapshot=trigger_snapshot, correction_event_snapshot=correction_snapshot, event_snapshot=event_snapshot, notes=tuple(notes))

def audit_shadow_filter(nominal_source: pd.DataFrame, shadow_filtered: pd.DataFrame, *, tolerance: float=1e-09, left_input_hash: str | None=None, right_input_hash: str | None=None) -> PairingAuditResult:
    return audit_pairing(nominal_source, shadow_filtered, mode='shadow', tolerance=tolerance, left_input_hash=left_input_hash, right_input_hash=right_input_hash, require_activation=True)

def write_pairing_audit(result: PairingAuditResult, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
__all__ = ['COMMON_GS_RESERVE_METRIC_UNITS', 'EXOGENOUS_ALIASES', 'MetricContractError', 'MetricRecord', 'REQUIRED_ENDPOINT_METRIC_IDS', 'PairingAuditResult', 'TRACE_ALIASES', 'TRACE_COLUMNS', 'TRACE_SCHEMA_VERSION', 'assert_metric_denominators', 'audit_pairing', 'audit_shadow_filter', 'compute_common_gs_reserve_step_table', 'compute_trace_metrics', 'infer_step_hours', 'normalize_trace_schema', 'write_pairing_audit']
