# EN: Experiment-matrix Cartesian expansion, SHA-256 `RunSpec` hashing, and parameter consistency checks.
# 中文：实验矩阵笛卡尔展开、`RunSpec` 配置哈希绑定与实验参数一致性校验。

from __future__ import annotations
from dataclasses import asdict, dataclass
import csv
import hashlib
from io import StringIO
from itertools import product
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4
from .data_pipeline import CONFIG_ROOT, load_split_specs
from .metrics import REQUIRED_ENDPOINT_METRIC_IDS, TRACE_COLUMNS
MATRIX_SCHEMA_VERSION = 2

class MatrixContractError(ValueError):
    pass

@dataclass(frozen=True)
class RunSpec:
    schema_version: int
    spec_id: str
    stage: str
    run_kind: str
    split_id: str
    condition_id: str
    arm_id: str
    scheduler_id: str
    filter_id: str
    calibration_id: str
    paper_method: str | None
    ev_multiplier: float
    pv_multiplier: float
    test_start: str
    test_end: str
    step_minutes: int
    expected_test_steps: int
    expected_trace_rows: int
    expected_milp_solves: int
    reused_milp_solves: int
    expected_power_flow_calls: int
    formal_evidence: bool
    deterministic_mpc_seed_applicable: bool
    verification_seed: int
    parameters_json: str
    parameter_notes: str
    reuse_dependency: str
    required_outputs_json: str
    config_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixContractError(f'cannot load matrix config {path}: {exc}') from exc

def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()

def _slug(*parts: str) -> str:
    text = '__'.join(parts).lower()
    return ''.join((character if character.isalnum() else '_' for character in text)).strip('_')

def _parameter_combinations(parameters: Mapping[str, Sequence[object]] | None) -> list[dict[str, object]]:
    if not parameters:
        return [{}]
    keys = list(parameters)
    values: list[Sequence[object]] = []
    for key in keys:
        axis = parameters[key]
        if not isinstance(axis, list) or not axis:
            raise MatrixContractError(f'parameter axis {key!r} must be a nonempty JSON list')
        values.append(axis)
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]

# EN: Expand JSON grid axes into hash-bound `RunSpec` records with deterministic `spec_id` digests.
# 中文：将 JSON 实验网格笛卡尔展开为带 SHA-256 配置摘要绑定的 `RunSpec` 运行规范列表。
def expand_experiment_matrix(*, matrix_path: Path | None=None, splits_path: Path | None=None, stages: Sequence[str] | None=None) -> list[RunSpec]:
    raw = _load_json(matrix_path or CONFIG_ROOT / 'experiment_matrix.json')
    if raw.get('schema_version') != MATRIX_SCHEMA_VERSION:
        raise MatrixContractError('unsupported experiment matrix schema')
    split_specs = load_split_specs(splits_path)
    arms = raw.get('arms', {})
    conditions = raw.get('conditions', {})
    seed = int(raw['seed'])
    deterministic_seed = bool(raw.get('deterministic_mpc_seed_applicable', False))
    stage_filter = set(stages) if stages else None
    specs: list[RunSpec] = []
    seen: set[str] = set()
    for grid in raw.get('grids', []):
        stage = str(grid['stage'])
        if stage_filter is not None and stage not in stage_filter:
            continue
        combinations = _parameter_combinations(grid.get('parameters'))
        step_minutes = int(grid.get('step_minutes', 60))
        if step_minutes <= 0 or 60 % step_minutes != 0:
            raise MatrixContractError(f'{stage}: step_minutes must be a positive divisor of 60')
        milp_per_step = int(grid.get('milp_solves_per_step', 1))
        reused_per_step = int(grid.get('reused_milp_solves_per_step', 0))
        power_flow_per_step = int(grid.get('power_flow_calls_per_step', 0))
        trace_rows_per_step = int(grid.get('trace_rows_per_step', 1))
        if min(milp_per_step, reused_per_step, power_flow_per_step, trace_rows_per_step) < 0:
            raise MatrixContractError(f'{stage}: expected work counts cannot be negative')
        for split_id in grid['splits']:
            if split_id not in split_specs:
                raise MatrixContractError(f'unknown split in {stage}: {split_id}')
            split = split_specs[split_id]
            expected_steps = split.test.hours * 60 // step_minutes
            for condition_id in grid['conditions']:
                if condition_id not in conditions:
                    raise MatrixContractError(f'unknown condition in {stage}: {condition_id}')
                condition = conditions[condition_id]
                for arm_id in grid['arms']:
                    if arm_id not in arms:
                        raise MatrixContractError(f'unknown arm in {stage}: {arm_id}')
                    arm = arms[arm_id]
                    for parameters in combinations:
                        payload = {'schema_version': MATRIX_SCHEMA_VERSION, 'stage': stage, 'run_kind': str(grid.get('run_kind', 'scheduler')), 'split_id': split_id, 'condition_id': condition_id, 'arm_id': arm_id, 'scheduler_id': arm['scheduler_id'], 'filter_id': arm['filter_id'], 'calibration_id': arm['calibration_id'], 'paper_method': arm.get('paper_method'), 'ev_multiplier': float(condition['ev_multiplier']), 'pv_multiplier': float(condition['pv_multiplier']), 'test_start': split.test.start.isoformat(sep=' '), 'test_end': split.test.end.isoformat(sep=' '), 'step_minutes': step_minutes, 'expected_test_steps': expected_steps, 'expected_trace_rows': expected_steps * trace_rows_per_step, 'expected_milp_solves': expected_steps * milp_per_step, 'reused_milp_solves': expected_steps * reused_per_step, 'expected_power_flow_calls': expected_steps * power_flow_per_step, 'formal_evidence': bool(condition.get('formal_evidence', True)), 'deterministic_mpc_seed_applicable': deterministic_seed, 'verification_seed': seed, 'parameters_json': _canonical_json(parameters), 'parameter_notes': str(grid.get('parameter_notes', '')), 'reuse_dependency': str(grid.get('reuse_dependency', '')), 'required_outputs_json': _canonical_json(grid.get('required_outputs', []))}
                        config_hash = _canonical_hash(payload)
                        spec_id = f'{_slug(stage, split_id, condition_id, arm_id)}_{config_hash[:10]}'
                        if spec_id in seen:
                            raise MatrixContractError(f'duplicate expanded matrix spec: {spec_id}')
                        seen.add(spec_id)
                        specs.append(RunSpec(**payload, spec_id=spec_id, config_sha256=config_hash))
    return specs
