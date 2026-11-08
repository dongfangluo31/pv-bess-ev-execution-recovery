# EN: Causal train/calibration/test split builder, UTC-to-UTC+8 PVOD station alignment, and TOU price mapping.
# 中文：因果历史/校准/测试切分构造、PVOD 光伏序列时区对齐折算与分时电价映射流水线。

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4
import numpy as np
import pandas as pd
REVISION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REVISION_ROOT.parent
CONFIG_ROOT = REVISION_ROOT / 'configs'
LEGACY_DATA_ROOT = Path(__import__('os').environ.get('PV_STATION_DATA', str(REVISION_ROOT / 'inputs'))).resolve()
DEFAULT_EV_PATH = LEGACY_DATA_ROOT / 'station_hourly_ev_load.csv'
DEFAULT_PRICE_PATH = LEGACY_DATA_ROOT / '28182251' / 'Dataset' / 'Time-of-use_Price.csv'
DEFAULT_PV_PATH = LEGACY_DATA_ROOT / 'PVOD' / 'station01.csv'
DEFAULT_PV_METADATA_PATH = LEGACY_DATA_ROOT / 'PVOD' / 'metadata.csv'
TARGET_TIMEZONE = 'Asia/Shanghai'
SOURCE_TIMEZONE = 'UTC'
PV_CALENDAR_MAPPING = 'same_month_day_hour_after_utc_to_asia_shanghai'

class DataContractError(ValueError):
    pass

@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_mapping(cls, value: Mapping[str, str]) -> 'TimeWindow':
        start = _naive_timestamp(value['start'])
        end = _naive_timestamp(value['end'])
        if not start < end:
            raise DataContractError(f'invalid half-open window [{start}, {end})')
        if (end - start) % pd.Timedelta(hours=1) != pd.Timedelta(0):
            raise DataContractError('window boundaries must lie on the hourly grid')
        return cls(start, end)

    @property
    def hours(self) -> int:
        return int((self.end - self.start) / pd.Timedelta(hours=1))

    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, inclusive='left', freq='h')

@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    label: str
    forecast_history: TimeWindow
    calibration: TimeWindow
    test: TimeWindow
    embargo: TimeWindow | None
    primary_evidence: bool
    sensitivity_only: bool

    def role_windows(self, include_embargo: bool=True) -> tuple[tuple[str, TimeWindow], ...]:
        rows: list[tuple[str, TimeWindow]] = [('forecast_history', self.forecast_history), ('calibration', self.calibration)]
        if include_embargo and self.embargo is not None:
            rows.append(('embargo', self.embargo))
        rows.append(('test', self.test))
        return tuple(rows)

@dataclass(frozen=True)
class PreparedDataset:
    inputs: pd.DataFrame
    pv_provenance: pd.DataFrame
    manifest: dict[str, Any]

def _naive_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise DataContractError(f'timezone-aware timestamp is not allowed in CSV contract: {value}')
    return timestamp

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f'cannot load protocol config {path}: {exc}') from exc

def load_split_specs(path: Path | None=None) -> dict[str, SplitSpec]:
    config_path = path or CONFIG_ROOT / 'splits.json'
    raw = _load_json(config_path)
    if raw.get('interval_convention') != 'left_closed_right_open':
        raise DataContractError('split config must state left-closed/right-open intervals')
    if raw.get('target_timezone') != TARGET_TIMEZONE:
        raise DataContractError('unexpected target timezone in split config')
    if raw.get('source_timezone') != SOURCE_TIMEZONE:
        raise DataContractError('unexpected source timezone policy in split config')
    specs: dict[str, SplitSpec] = {}
    for split_id, value in raw.get('splits', {}).items():
        forecast_history = TimeWindow.from_mapping(value['forecast_history'])
        calibration = TimeWindow.from_mapping(value['calibration'])
        test = TimeWindow.from_mapping(value['test'])
        embargo = TimeWindow.from_mapping(value['embargo']) if 'embargo' in value else None
        if forecast_history.end != calibration.start or forecast_history.hours < 168:
            raise DataContractError(f'{split_id}: forecast_history must provide at least 168 hours and end at calibration start')
        if calibration.end > test.start:
            raise DataContractError(f'{split_id}: calibration overlaps or follows test')
        if embargo is not None:
            if calibration.end > embargo.start or embargo.end > test.start:
                raise DataContractError(f'{split_id}: embargo overlaps calibration or test')
            if embargo.start != calibration.end or embargo.end != test.start:
                raise DataContractError(f'{split_id}: embargo must exactly fill the declared gap')
        specs[split_id] = SplitSpec(split_id=split_id, label=str(value['label']), forecast_history=forecast_history, calibration=calibration, test=test, embargo=embargo, primary_evidence=bool(value.get('primary_evidence', False)), sensitivity_only=bool(value.get('sensitivity_only', False)))
    if not specs:
        raise DataContractError('split config contains no splits')
    return specs

def load_split_spec(split_id: str, path: Path | None=None) -> SplitSpec:
    try:
        return load_split_specs(path)[split_id]
    except KeyError as exc:
        raise DataContractError(f'unknown split_id {split_id!r}') from exc

def _read_ev_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataContractError(f'EV source does not exist: {path}')
    frame = pd.read_csv(path)
    required = {'time', 'ev_load_kw'}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f'EV source missing columns: {sorted(missing)}')
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame['time'], errors='raise')
    if frame['timestamp'].dt.tz is not None:
        raise DataContractError('EV source timestamp must be naive')
    if frame['timestamp'].duplicated().any():
        raise DataContractError('EV source contains duplicate hourly timestamps')
    if not frame['timestamp'].is_monotonic_increasing:
        raise DataContractError('EV source must be sorted by timestamp')
    values = pd.to_numeric(frame['ev_load_kw'], errors='coerce')
    if values.isna().any() or (values < 0).any():
        raise DataContractError('EV load must be finite and nonnegative')
    frame['ev_load_kw'] = values.astype(float)
    if 'ev_energy_kwh' in frame:
        energy = pd.to_numeric(frame['ev_energy_kwh'], errors='coerce')
        if energy.isna().any() or (energy < 0).any():
            raise DataContractError('EV energy must be finite and nonnegative')
        frame['ev_energy_kwh'] = energy.astype(float)
    else:
        frame['ev_energy_kwh'] = frame['ev_load_kw']
    return frame[['timestamp', 'ev_load_kw', 'ev_energy_kwh']]

def _select_complete_window(source: pd.DataFrame, window: TimeWindow, role: str) -> pd.DataFrame:
    expected = window.index()
    indexed = source.set_index('timestamp')
    selected = indexed.reindex(expected)
    if selected.isna().any(axis=None):
        missing = selected.index[selected.isna().any(axis=1)]
        preview = ', '.join((str(value) for value in missing[:5]))
        raise DataContractError(f'{role} is missing {len(missing)} EV hours; first: {preview}')
    selected.index.name = 'timestamp'
    return selected.reset_index()

def _attach_forecast_source_provenance(inputs: pd.DataFrame) -> pd.DataFrame:
    result = inputs.copy()
    role_by_time = result.set_index('timestamp')['split_role']
    eligible = ~result['split_role'].eq('forecast_history')
    for predictor, lag_hours in (('p24', 24), ('p168', 168)):
        source_time = result['timestamp'] - pd.Timedelta(hours=lag_hours)
        source_role = source_time.map(role_by_time)
        missing = eligible & source_role.isna()
        if missing.any():
            first = result.loc[missing, 'timestamp'].iloc[0]
            raise DataContractError(f'{predictor} lacks a past observation for target {first}; forecast_history is too short or discontinuous')
        result[f'{predictor}_source_timestamp'] = source_time.where(eligible, pd.NaT)
        result[f'{predictor}_source_role'] = source_role.where(eligible, pd.NA)
        if not (result.loc[eligible, f'{predictor}_source_timestamp'].to_numpy() < result.loc[eligible, 'timestamp'].to_numpy()).all():
            raise DataContractError(f'{predictor} attempted a non-past forecast read')
    return result

def _clock_minutes(value: object, *, end: bool) -> int:
    text = str(value).strip()
    hour_text, minute_text = text.split(':', maxsplit=1)
    minute = int(hour_text) * 60 + int(minute_text)
    if end and text == '23:59':
        return 24 * 60
    if not 0 <= minute <= 24 * 60:
        raise DataContractError(f'invalid time-of-use boundary {text!r}')
    return minute

def _load_price_periods(path: Path) -> list[tuple[int, int, float]]:
    if not path.is_file():
        raise DataContractError(f'price source does not exist: {path}')
    frame = pd.read_csv(path)
    required = {'Time Period', 'Electricity Price(Yuan/kWh)'}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f'price source missing columns: {sorted(missing)}')
    periods: list[tuple[int, int, float]] = []
    for _, row in frame.iterrows():
        start_text, end_text = str(row['Time Period']).split('-', maxsplit=1)
        start = _clock_minutes(start_text, end=False)
        end = _clock_minutes(end_text, end=True)
        price = float(row['Electricity Price(Yuan/kWh)'])
        if not start < end or not math.isfinite(price) or price < 0:
            raise DataContractError(f'invalid time-of-use row: {row.to_dict()}')
        periods.append((start, end, price))
    return periods

def prices_for_timestamps(timestamps: Iterable[pd.Timestamp], price_path: Path=DEFAULT_PRICE_PATH) -> np.ndarray:
    periods = _load_price_periods(price_path)
    prices: list[float] = []
    for timestamp in timestamps:
        timestamp = _naive_timestamp(timestamp)
        minute = timestamp.hour * 60 + timestamp.minute
        matches = [price for start, end, price in periods if start <= minute < end]
        if len(matches) != 1:
            raise DataContractError(f'timestamp {timestamp} matches {len(matches)} time-of-use intervals')
        prices.append(matches[0])
    return np.asarray(prices, dtype=float)

def _station_metadata(path: Path, station_id: str) -> tuple[float, dict[str, object]]:
    if not path.is_file():
        raise DataContractError(f'PV metadata does not exist: {path}')
    frame = pd.read_csv(path)
    id_col = next((name for name in frame.columns if name.strip().lower() == 'station_id'), None)
    cap_col = next((name for name in frame.columns if name.strip().lower() == 'capacity'), None)
    if id_col is None or cap_col is None:
        raise DataContractError('PV metadata lacks Station_ID/Capacity')
    match = frame[frame[id_col].astype(str).str.casefold() == station_id.casefold()]
    if len(match) != 1:
        raise DataContractError(f'expected exactly one metadata row for {station_id}, found {len(match)}')
    row = match.iloc[0]
    capacity_kw = float(row[cap_col])
    if not math.isfinite(capacity_kw) or capacity_kw <= 0:
        raise DataContractError('PV installed capacity must be positive')
    selected: dict[str, object] = {'station_id': station_id, 'capacity_kw': capacity_kw}
    for source_name, target_name in (('Longitude', 'longitude'), ('Latitude', 'latitude')):
        if source_name in row.index:
            selected[target_name] = float(row[source_name])
    return (capacity_kw, selected)

def map_pvod_hourly(target_rows: pd.DataFrame, *, pv_path: Path=DEFAULT_PV_PATH, metadata_path: Path=DEFAULT_PV_METADATA_PATH, station_id: str='station01', pv_rated_kw: float=300.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if pv_rated_kw <= 0 or not math.isfinite(pv_rated_kw):
        raise DataContractError('pv_rated_kw must be finite and positive')
    if not pv_path.is_file():
        raise DataContractError(f'PV source does not exist: {pv_path}')
    raw = pd.read_csv(pv_path, usecols=['date_time', 'power'])
    raw['source_timestamp'] = pd.to_datetime(raw['date_time'], errors='raise')
    if raw['source_timestamp'].dt.tz is not None:
        raise DataContractError('PVOD CSV must contain naive timestamps representing UTC')
    raw['source_local_timestamp'] = raw['source_timestamp'].dt.tz_localize('UTC').dt.tz_convert(TARGET_TIMEZONE).dt.tz_localize(None)
    raw['raw_power_mw'] = pd.to_numeric(raw['power'], errors='coerce')
    if raw[['source_timestamp', 'raw_power_mw']].isna().any(axis=None):
        raise DataContractError('PVOD contains missing/non-numeric timestamp or power')
    if raw['source_timestamp'].duplicated().any():
        raise DataContractError('PVOD contains duplicate 15-minute timestamps')
    if (raw['raw_power_mw'] < 0).any():
        raise DataContractError('PVOD power contains negative values')
    capacity_kw, station_meta = _station_metadata(metadata_path, station_id)
    capacity_mw = capacity_kw / 1000.0
    raw['source_hour'] = raw['source_timestamp'].dt.floor('h')
    raw['source_local_hour'] = raw['source_local_timestamp'].dt.floor('h')
    grouped_rows: list[dict[str, object]] = []
    for source_hour, group in raw.groupby('source_hour', sort=True):
        group = group.sort_values('source_timestamp')
        offsets = tuple((int((value - source_hour).total_seconds() / 60) for value in group['source_timestamp']))
        if len(group) != 4 or offsets != (0, 15, 30, 45):
            raise DataContractError(f'PV source hour {source_hour} is not a complete 00/15/30/45 quartet: {offsets}')
        grouped_rows.append({'source_hour': source_hour, 'source_local_hour': group['source_local_hour'].iloc[0], 'source_hour_count': 1, 'source_record_count': 4, 'source_timestamps': json.dumps([value.isoformat(sep=' ') for value in group['source_timestamp']], ensure_ascii=False), 'source_hours': json.dumps([source_hour.isoformat(sep=' ')]), 'source_timestamps_local': json.dumps([value.isoformat(sep=' ') for value in group['source_local_timestamp']], ensure_ascii=False), 'source_hours_local': json.dumps([group['source_local_hour'].iloc[0].isoformat(sep=' ')]), 'source_earliest_timestamp': group['source_timestamp'].iloc[0], 'source_latest_timestamp': group['source_timestamp'].iloc[-1], 'source_local_earliest_timestamp': group['source_local_timestamp'].iloc[0], 'source_local_latest_timestamp': group['source_local_timestamp'].iloc[-1], 'source_power_mean_mw': float(group['raw_power_mw'].mean()), 'source_power_min_mw': float(group['raw_power_mw'].min()), 'source_power_max_mw': float(group['raw_power_mw'].max())})
    hourly = pd.DataFrame(grouped_rows)
    hourly['calendar_key'] = list(zip(hourly['source_local_hour'].dt.month, hourly['source_local_hour'].dt.day, hourly['source_local_hour'].dt.hour))
    if hourly['calendar_key'].duplicated().any():
        duplicate = hourly.loc[hourly['calendar_key'].duplicated(), 'calendar_key'].iloc[0]
        raise DataContractError(f'ambiguous PV calendar key across source years: {duplicate}')
    lookup = hourly.set_index('calendar_key')
    hourly['climatology_key'] = list(zip(hourly['source_local_hour'].dt.month, hourly['source_local_hour'].dt.hour))
    climatology_rows: list[dict[str, object]] = []
    for climatology_key, group in hourly.groupby('climatology_key', sort=True):
        source_times: list[str] = []
        source_times_local: list[str] = []
        for encoded in group['source_timestamps']:
            source_times.extend(json.loads(encoded))
        for encoded in group['source_timestamps_local']:
            source_times_local.extend(json.loads(encoded))
        climatology_rows.append({'climatology_key': climatology_key, 'source_hour': pd.NaT, 'source_local_hour': pd.NaT, 'source_hour_count': len(group), 'source_record_count': int(group['source_record_count'].sum()), 'source_timestamps': json.dumps(source_times, ensure_ascii=False), 'source_timestamps_local': json.dumps(source_times_local, ensure_ascii=False), 'source_hours': json.dumps([value.isoformat(sep=' ') for value in group['source_hour']], ensure_ascii=False), 'source_hours_local': json.dumps([value.isoformat(sep=' ') for value in group['source_local_hour']], ensure_ascii=False), 'source_earliest_timestamp': group['source_earliest_timestamp'].min(), 'source_latest_timestamp': group['source_latest_timestamp'].max(), 'source_local_earliest_timestamp': group['source_local_earliest_timestamp'].min(), 'source_local_latest_timestamp': group['source_local_latest_timestamp'].max(), 'source_power_mean_mw': float(group['source_power_mean_mw'].mean()), 'source_power_min_mw': float(group['source_power_min_mw'].min()), 'source_power_max_mw': float(group['source_power_max_mw'].max())})
    climatology_lookup = pd.DataFrame(climatology_rows).set_index('climatology_key')
    required = {'timestamp', 'split_role'}
    missing = required.difference(target_rows.columns)
    if missing:
        raise DataContractError(f'target rows missing columns: {sorted(missing)}')
    target = target_rows[['timestamp', 'split_role']].copy()
    target['timestamp'] = pd.to_datetime(target['timestamp'], errors='raise')
    if target['timestamp'].dt.tz is not None:
        raise DataContractError('target timestamps must be naive')
    source_hash = _sha256_file(pv_path)
    metadata_hash = _sha256_file(metadata_path)
    mapped: list[dict[str, object]] = []
    for row in target.itertuples(index=False):
        key = (row.timestamp.month, row.timestamp.day, row.timestamp.hour)
        mapping_category = 'exact'
        mapping_day_offset: int | None = 0
        if key in lookup.index:
            source = lookup.loc[[key]].iloc[0]
        else:
            source = None
            calendar_anchor = pd.Timestamp(year=2000, month=row.timestamp.month, day=row.timestamp.day, hour=row.timestamp.hour)
            for day_offset in (-1, 1, -2, 2, -3, 3):
                adjacent = calendar_anchor + pd.Timedelta(days=day_offset)
                adjacent_key = (adjacent.month, adjacent.day, adjacent.hour)
                if adjacent_key in lookup.index:
                    source = lookup.loc[[adjacent_key]].iloc[0]
                    mapping_category = 'plus_minus_day'
                    mapping_day_offset = day_offset
                    break
            if source is None:
                climatology_key = (key[0], key[2])
                if climatology_key in climatology_lookup.index:
                    source = climatology_lookup.loc[[climatology_key]].iloc[0]
                    mapping_category = 'climatology'
                    mapping_day_offset = None
            if source is None:
                source = pd.Series({'source_hour': pd.NaT, 'source_local_hour': pd.NaT, 'source_hour_count': 0, 'source_record_count': 0, 'source_timestamps': '[]', 'source_hours': '[]', 'source_timestamps_local': '[]', 'source_hours_local': '[]', 'source_earliest_timestamp': pd.NaT, 'source_latest_timestamp': pd.NaT, 'source_local_earliest_timestamp': pd.NaT, 'source_local_latest_timestamp': pd.NaT, 'source_power_mean_mw': 0.0, 'source_power_min_mw': 0.0, 'source_power_max_mw': 0.0})
                mapping_category = 'zero'
                mapping_day_offset = None
        source_mean_mw = float(source['source_power_mean_mw'])
        normalized_unclipped = source_mean_mw / capacity_mw
        normalized = float(np.clip(normalized_unclipped, 0.0, 1.0))
        mapped.append({'timestamp': row.timestamp, 'split_role': row.split_role, 'target_timezone': TARGET_TIMEZONE, 'source_station': station_id, 'source_timestamp_hour': source['source_hour'], 'source_local_timestamp_hour': source['source_local_hour'], 'source_hours_json': source['source_hours'], 'source_timestamps_15min_json': source['source_timestamps'], 'source_hours_local_json': source['source_hours_local'], 'source_timestamps_15min_local_json': source['source_timestamps_local'], 'source_earliest_timestamp': source['source_earliest_timestamp'], 'source_latest_timestamp': source['source_latest_timestamp'], 'source_local_earliest_timestamp': source['source_local_earliest_timestamp'], 'source_local_latest_timestamp': source['source_local_latest_timestamp'], 'source_timezone': SOURCE_TIMEZONE, 'source_local_timezone': TARGET_TIMEZONE, 'timezone_conversion_applied': True, 'calendar_mapping': PV_CALENDAR_MAPPING, 'mapping_category': mapping_category, 'mapping_day_offset': mapping_day_offset, 'mapping_semantics': 'exact means an exact month-day-hour key after UTC-to-local conversion in the historical, non-colocated calendar proxy; it does not mean co-temporal measurement', 'non_colocated_source': True, 'uses_target_future_data': False, 'source_hour_count': int(source['source_hour_count']), 'source_record_count': int(source['source_record_count']), 'source_resolution_minutes': 15, 'source_power_unit': 'MW', 'source_power_mean_mw': source_mean_mw, 'source_power_min_mw': float(source['source_power_min_mw']), 'source_power_max_mw': float(source['source_power_max_mw']), 'source_capacity_kw': capacity_kw, 'source_capacity_mw': capacity_mw, 'normalized_power_unclipped_pu': normalized_unclipped, 'normalized_power_pu': normalized, 'clip_applied': not math.isclose(normalized, normalized_unclipped, abs_tol=0.0), 'pv_rated_scaled_kw': pv_rated_kw, 'pv_power_kw': normalized * pv_rated_kw, 'source_file': str(pv_path.resolve()), 'source_file_sha256': source_hash, 'metadata_file': str(metadata_path.resolve()), 'metadata_file_sha256': metadata_hash})
    provenance = pd.DataFrame(mapped)
    mapped_pv = provenance[['timestamp', 'pv_power_kw']].copy()
    policy = {'source_station': station_id, 'source_timezone': SOURCE_TIMEZONE, 'target_timezone': TARGET_TIMEZONE, 'source_local_timezone': TARGET_TIMEZONE, 'timezone_conversion_applied': True, 'calendar_mapping': PV_CALENDAR_MAPPING, 'mapping_categories': ['exact', 'plus_minus_day', 'climatology', 'zero'], 'mapping_category_counts': {category: int((provenance['mapping_category'] == category).sum()) for category in ('exact', 'plus_minus_day', 'climatology', 'zero')}, 'mapping_category_counts_by_role': {str(role): {category: int(((provenance['split_role'] == role) & (provenance['mapping_category'] == category)).sum()) for category in ('exact', 'plus_minus_day', 'climatology', 'zero')} for role in provenance['split_role'].unique()}, 'fallback_order': ['exact', 'plus_minus_day', 'climatology', 'zero'], 'exact_category_semantics': 'exact historical local month-day-hour key after UTC conversion; non-colocated and not a co-temporal measurement', 'non_colocated_source': True, 'co_temporal_measurement_claimed': False, 'uses_target_future_data': False, 'source_power_unit': 'MW', 'source_capacity_kw': capacity_kw, 'pv_rated_scaled_kw': pv_rated_kw, 'station_metadata': station_meta, 'source_file_sha256': source_hash, 'metadata_file_sha256': metadata_hash}
    return (mapped_pv, provenance, policy)

def _source_manifest_entry(path: Path, role: str) -> dict[str, object]:
    resolved = path.resolve()
    return {'role': role, 'path': str(resolved), 'bytes': resolved.stat().st_size, 'sha256': _sha256_file(resolved), 'read_only_input': True}

def build_split_dataset(split_id: str, *, ev_multiplier: float=1.0, pv_multiplier: float=1.0, pv_rated_kw: float=300.0, include_embargo: bool=True, split_config_path: Path | None=None, ev_path: Path=DEFAULT_EV_PATH, price_path: Path=DEFAULT_PRICE_PATH, pv_path: Path=DEFAULT_PV_PATH, pv_metadata_path: Path=DEFAULT_PV_METADATA_PATH) -> PreparedDataset:
    for name, value in (('ev_multiplier', ev_multiplier), ('pv_multiplier', pv_multiplier)):
        if not math.isfinite(value) or value < 0:
            raise DataContractError(f'{name} must be finite and nonnegative')
    spec = load_split_spec(split_id, split_config_path)
    ev_source = _read_ev_source(ev_path)
    role_frames: list[pd.DataFrame] = []
    for role, window in spec.role_windows(include_embargo=include_embargo):
        frame = _select_complete_window(ev_source, window, role)
        frame.insert(1, 'split_role', role)
        role_frames.append(frame)
    inputs = pd.concat(role_frames, ignore_index=True)
    if inputs['timestamp'].duplicated().any():
        raise DataContractError(f'{split_id}: selected roles overlap')
    inputs.insert(0, 'split_id', split_id)
    inputs = _attach_forecast_source_provenance(inputs)
    inputs['ev_load_kw_source'] = inputs['ev_load_kw']
    inputs['ev_energy_kwh_source'] = inputs['ev_energy_kwh']
    inputs['ev_multiplier'] = float(ev_multiplier)
    inputs['ev_load_kw'] = inputs['ev_load_kw_source'] * ev_multiplier
    inputs['ev_energy_kwh'] = inputs['ev_energy_kwh_source'] * ev_multiplier
    inputs['price_yuan_per_kwh'] = prices_for_timestamps(inputs['timestamp'], price_path)
    pv_values, pv_provenance, pv_policy = map_pvod_hourly(inputs[['timestamp', 'split_role']], pv_path=pv_path, metadata_path=pv_metadata_path, pv_rated_kw=pv_rated_kw)
    if not inputs['timestamp'].reset_index(drop=True).equals(pv_values['timestamp'].reset_index(drop=True)):
        raise DataContractError('PV mapping changed target row order')
    inputs['pv_power_kw_source'] = pv_values['pv_power_kw'].to_numpy()
    inputs['pv_multiplier'] = float(pv_multiplier)
    inputs['pv_power_kw'] = inputs['pv_power_kw_source'] * pv_multiplier
    inputs['pv_energy_kwh'] = inputs['pv_power_kw']
    inputs['net_load_without_bess_kw'] = inputs['ev_load_kw'] - inputs['pv_power_kw']
    inputs['grid_load_without_bess_kw'] = inputs['net_load_without_bess_kw'].clip(lower=0.0)
    inputs['pv_curtail_without_bess_kw'] = (-inputs['net_load_without_bess_kw']).clip(lower=0.0)
    inputs['grid_cost_without_bess_yuan'] = inputs['grid_load_without_bess_kw'] * inputs['price_yuan_per_kwh']
    inputs['target_timezone'] = TARGET_TIMEZONE
    inputs['timestamp_storage'] = 'naive_iso8601'
    pv_provenance.insert(0, 'split_id', split_id)
    pv_provenance['pv_multiplier'] = float(pv_multiplier)
    pv_provenance['pv_power_kw_after_multiplier'] = pv_provenance['pv_power_kw'] * pv_multiplier
    manifest: dict[str, Any] = {'schema_version': 1, 'split_id': split_id, 'label': spec.label, 'interval_convention': 'left_closed_right_open', 'target_timezone': TARGET_TIMEZONE, 'timestamp_storage': 'naive_iso8601', 'source_timezone': SOURCE_TIMEZONE, 'pv_source_timezone': SOURCE_TIMEZONE, 'ev_source_timezone': TARGET_TIMEZONE, 'primary_evidence': spec.primary_evidence, 'sensitivity_only': spec.sensitivity_only, 'windows': {role: {'start': window.start.isoformat(sep=' '), 'end': window.end.isoformat(sep=' '), 'hours': window.hours} for role, window in spec.role_windows(include_embargo=include_embargo)}, 'row_counts': {role: int((inputs['split_role'] == role).sum()) for role in inputs['split_role'].unique()}, 'forecast_error_eligibility': {'predictors': ['P24', 'P168'], 'forecast_history_hours': spec.forecast_history.hours, 'forecast_history_counts_as_calibration': False, 'calibration_error_count': spec.calibration.hours, 'calibration_targets_all_have_past_sources': True, 'test_first_step_requires_truth_warmup': False, 'embargo_observations_available_to_forecaster': spec.embargo is not None, 'embargo_observations_in_conformal_calibration': False}, 'ev_multiplier': float(ev_multiplier), 'pv_multiplier': float(pv_multiplier), 'pv_policy': pv_policy, 'source_files': [_source_manifest_entry(ev_path, 'hourly_ev_load'), _source_manifest_entry(price_path, 'time_of_use_price'), _source_manifest_entry(pv_path, 'pvod_station_power'), _source_manifest_entry(pv_metadata_path, 'pvod_station_metadata'), _source_manifest_entry(split_config_path or CONFIG_ROOT / 'splits.json', 'split_protocol')]}
    prepared = PreparedDataset(inputs=inputs, pv_provenance=pv_provenance, manifest=manifest)
    verify_prepared_dataset(prepared, spec=spec, include_embargo=include_embargo)
    return prepared

def verify_prepared_dataset(prepared: PreparedDataset, *, spec: SplitSpec | None=None, include_embargo: bool=True) -> None:
    inputs = prepared.inputs
    provenance = prepared.pv_provenance
    required = {'split_id', 'split_role', 'timestamp', 'ev_load_kw', 'pv_power_kw', 'price_yuan_per_kwh'}
    missing = required.difference(inputs.columns)
    if missing:
        raise DataContractError(f'prepared inputs missing columns: {sorted(missing)}')
    if inputs['timestamp'].duplicated().any():
        raise DataContractError('prepared inputs contain duplicate timestamps')
    if inputs['timestamp'].dt.tz is not None:
        raise DataContractError('prepared input timestamps must be naive')
    numeric = inputs[['ev_load_kw', 'pv_power_kw', 'price_yuan_per_kwh']].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise DataContractError('prepared core inputs must be finite and nonnegative')
    if spec is None:
        split_ids = inputs['split_id'].unique()
        if len(split_ids) != 1:
            raise DataContractError('prepared dataset must contain exactly one split_id')
        spec = load_split_spec(str(split_ids[0]))
    for role, window in spec.role_windows(include_embargo=include_embargo):
        observed = pd.DatetimeIndex(inputs.loc[inputs['split_role'] == role, 'timestamp'])
        expected = window.index()
        if not observed.equals(expected):
            raise DataContractError(f'{spec.split_id}/{role}: expected {len(expected)} exact hourly rows, got {len(observed)}')
    calibration_mask = inputs['split_role'].eq('calibration')
    test_mask = inputs['split_role'].eq('test')
    if int(calibration_mask.sum()) != spec.calibration.hours:
        raise DataContractError('calibration residual count differs from the registered window')
    for predictor in ('p24', 'p168'):
        source_column = f'{predictor}_source_timestamp'
        role_column = f'{predictor}_source_role'
        if source_column not in inputs or role_column not in inputs:
            raise DataContractError(f'missing {predictor} forecast-source provenance')
        for mask, label in ((calibration_mask, 'calibration'), (test_mask, 'test')):
            sources = pd.to_datetime(inputs.loc[mask, source_column], errors='raise')
            targets = pd.to_datetime(inputs.loc[mask, 'timestamp'], errors='raise')
            if len(sources) != int(mask.sum()) or sources.isna().any():
                raise DataContractError(f'{predictor}: incomplete {label} forecast sources')
            if not (sources.to_numpy() < targets.to_numpy()).all():
                raise DataContractError(f'{predictor}: {label} forecast reads current/future truth')
        if pd.isna(inputs.loc[test_mask, source_column].iloc[0]):
            raise DataContractError(f'{predictor}: test first step requires an undeclared warm-up')
    if spec.embargo is not None and include_embargo:
        first_test = inputs.loc[test_mask].iloc[0]
        if first_test['p24_source_role'] != 'embargo' or first_test['p168_source_role'] != 'embargo':
            raise DataContractError('clean-gap test forecasts may use observed embargo history, while calibration remains separate')
    if len(provenance) != len(inputs):
        raise DataContractError('hourly PV provenance row count does not match inputs')
    if not inputs['timestamp'].reset_index(drop=True).equals(provenance['timestamp'].reset_index(drop=True)):
        raise DataContractError('hourly PV provenance timestamps are misaligned')
    required_provenance = {'timestamp', 'source_timestamp_hour', 'source_earliest_timestamp', 'source_latest_timestamp', 'source_local_timestamp_hour', 'source_local_earliest_timestamp', 'source_local_latest_timestamp', 'source_timestamps_15min_local_json', 'source_hours_local_json', 'source_power_mean_mw', 'normalized_power_pu', 'pv_power_kw', 'source_file_sha256', 'mapping_category', 'uses_target_future_data'}
    missing_provenance = required_provenance.difference(provenance.columns)
    if missing_provenance:
        raise DataContractError(f'hourly PV provenance missing columns: {sorted(missing_provenance)}')
    allowed_categories = {'exact', 'plus_minus_day', 'climatology', 'zero'}
    observed_categories = set(provenance['mapping_category'].astype(str).unique())
    if not observed_categories.issubset(allowed_categories):
        raise DataContractError(f'unknown PV mapping category: {observed_categories}')
    exact_or_adjacent = provenance['mapping_category'].isin(['exact', 'plus_minus_day'])
    climatology = provenance['mapping_category'].eq('climatology')
    zero = provenance['mapping_category'].eq('zero')
    if not (provenance.loc[exact_or_adjacent, 'source_record_count'] == 4).all():
        raise DataContractError('exact/plus-minus-day PV mappings must cite four 15-minute records')
    if not ((provenance.loc[climatology, 'source_record_count'] >= 4) & (provenance.loc[climatology, 'source_record_count'] % 4 == 0)).all():
        raise DataContractError('climatology mappings must cite complete historical hours')
    if not (provenance.loc[zero, 'source_record_count'] == 0).all():
        raise DataContractError('zero fallback cannot claim source measurements')
    if not (provenance['source_timezone'] == SOURCE_TIMEZONE).all():
        raise DataContractError('PV source timezone policy changed')
    if not provenance['timezone_conversion_applied'].astype(bool).all():
        raise DataContractError('PVOD UTC timestamps must be converted before local calendar mapping')
    if not (provenance['calendar_mapping'] == PV_CALENDAR_MAPPING).all():
        raise DataContractError('unexpected PV calendar mapping')
    if not provenance['non_colocated_source'].astype(bool).all():
        raise DataContractError('PV non-colocation disclosure is missing')
    if provenance['uses_target_future_data'].astype(bool).any():
        raise DataContractError('PV mapping cannot read target-future observations')
    nonzero = ~zero
    source_latest = pd.to_datetime(provenance.loc[nonzero, 'source_latest_timestamp'], errors='raise')
    target_time = pd.to_datetime(provenance.loc[nonzero, 'timestamp'], errors='raise').dt.tz_localize(TARGET_TIMEZONE).dt.tz_convert('UTC').dt.tz_localize(None)
    if not (source_latest.to_numpy() < target_time.to_numpy()).all():
        raise DataContractError('PV calendar proxy contains a source time at/after the target time')
    for endpoint in ('earliest', 'latest'):
        raw_clock = pd.to_datetime(provenance.loc[nonzero, f'source_{endpoint}_timestamp'], errors='raise')
        expected_local = raw_clock.dt.tz_localize('UTC').dt.tz_convert(TARGET_TIMEZONE).dt.tz_localize(None)
        local_clock = pd.to_datetime(provenance.loc[nonzero, f'source_local_{endpoint}_timestamp'], errors='raise')
        if not np.array_equal(expected_local.to_numpy(), local_clock.to_numpy()):
            raise DataContractError('PV raw UTC and converted local provenance disagree')
    hashes = provenance['source_file_sha256'].astype(str)
    if not hashes.str.fullmatch('[0-9a-f]{64}').all() or hashes.nunique() != 1:
        raise DataContractError('PV source hash is missing or changes within one prepared split')
    expected_scaled = provenance['normalized_power_pu'].to_numpy(float) * provenance['pv_rated_scaled_kw'].to_numpy(float)
    if not np.allclose(expected_scaled, provenance['pv_power_kw'], atol=1e-12, rtol=0.0):
        raise DataContractError('PV normalization/scaling identity failed')

def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        frame.to_csv(temporary, index=False, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _atomic_write_json(value: Mapping[str, object], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def write_prepared_dataset(prepared: PreparedDataset, output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {'combined': output_dir / 'case_inputs.csv', 'pv_provenance': output_dir / 'pv_hourly_provenance.csv'}
    for role in prepared.inputs['split_role'].unique():
        outputs[str(role)] = output_dir / f'{role}.csv'
    manifest_path = output_dir / 'dataset_manifest.json'
    targets = [*outputs.values(), manifest_path]
    collisions = [path for path in targets if path.exists()]
    if collisions:
        raise FileExistsError('refusing a partial prepared-dataset write; target paths already exist: ' + ', '.join((str(path) for path in collisions)))
    _atomic_write_csv(prepared.inputs, outputs['combined'])
    _atomic_write_csv(prepared.pv_provenance, outputs['pv_provenance'])
    for role in prepared.inputs['split_role'].unique():
        role_path = outputs[str(role)]
        _atomic_write_csv(prepared.inputs.loc[prepared.inputs['split_role'] == role].reset_index(drop=True), role_path)
    manifest = dict(prepared.manifest)
    manifest['output_files'] = [{'role': role, 'path': path.name, 'bytes': path.stat().st_size, 'sha256': _sha256_file(path)} for role, path in outputs.items()]
    _atomic_write_json(manifest, manifest_path)
    outputs['manifest'] = manifest_path
    return outputs
__all__ = ['CONFIG_ROOT', 'DataContractError', 'LEGACY_DATA_ROOT', 'PV_CALENDAR_MAPPING', 'PreparedDataset', 'SOURCE_TIMEZONE', 'SplitSpec', 'TARGET_TIMEZONE', 'TimeWindow', 'build_split_dataset', 'load_split_spec', 'load_split_specs', 'map_pvod_hourly', 'prices_for_timestamps', 'verify_prepared_dataset', 'write_prepared_dataset']
