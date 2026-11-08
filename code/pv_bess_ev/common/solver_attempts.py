# EN: Durable fsync-backed CSV solver-call logger recording every MILP invocation, runtime, status, and MIP gap.
# 中文：带 fsync 持久化的外部 MILP 求解逐次调用日志记录器（记录耗时、状态码与 MIP 间隙）。

from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Iterator, TypeVar
SOLVER_ATTEMPT_COLUMNS = ('run_id', 'run_kind', 'attempt_id', 'attempt_index', 'decision_step', 'decision_timestamp', 'station_id', 'station_index', 'profile_id', 'variant_id', 'target_window', 'started_utc', 'completed_utc', 'outcome', 'status', 'solver_success', 'solver_message', 'exception', 'exception_type', 'solve_time_s', 'mip_gap', 'method', 'controller')
_OUTCOMES = {'success', 'return_failure', 'exception'}
_T = TypeVar('_T')

class SolverAttemptLogError(RuntimeError):
    pass

@dataclass(frozen=True)
class SolverAttemptContext:
    logger: 'SolverAttemptLogger'
    decision_step: int | None
    decision_timestamp: object | None
    station_id: object | None
    station_index: int | None
    profile_id: object | None
    variant_id: object | None
    target_window: object | None
_ACTIVE_SOLVER_ATTEMPT_CONTEXT: ContextVar[SolverAttemptContext | None] = ContextVar('active_solver_attempt_context', default=None)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _csv_value(value: object | None) -> object:
    if value is None:
        return ''
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value

def _optional_float(value: object | None) -> str | float:
    if value is None:
        return ''
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return ''
    return result if math.isfinite(result) else ''

# EN: Maintain contiguous 1-based attempt_index sequences (`<run_id>:milp:<08d>`) across all MILP solver calls.
# 中文：维护从 1 开始连续递增的求解序号，完整记录每次 MILP 求解调用及其返回状态。
class SolverAttemptLogger:

    def __init__(self, path: Path, run_id: str, run_kind: str) -> None:
        self.path = Path(path).resolve()
        self.run_id = str(run_id)
        self.run_kind = str(run_kind)
        if not self.run_id:
            raise ValueError('run_id must be nonempty')
        if not self.run_kind:
            raise ValueError('run_kind must be nonempty')
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_or_validate()

    def _initialize_or_validate(self) -> None:
        with self._lock:
            if not self.path.exists() or self.path.stat().st_size == 0:
                with self.path.open('w', encoding='utf-8', newline='') as handle:
                    csv.DictWriter(handle, fieldnames=SOLVER_ATTEMPT_COLUMNS, lineterminator='\n').writeheader()
                    handle.flush()
                    os.fsync(handle.fileno())
                return
            self._read_rows()

    def _read_rows(self) -> list[dict[str, str]]:
        with self.path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOLVER_ATTEMPT_COLUMNS:
                raise SolverAttemptLogError(f'solver attempt log schema mismatch: {self.path}')
            rows = list(reader)
        indexes: list[int] = []
        for row_number, row in enumerate(rows, start=2):
            if row['run_id'] != self.run_id or row['run_kind'] != self.run_kind:
                raise SolverAttemptLogError(f'solver attempt identity mismatch at CSV row {row_number}')
            if row['outcome'] not in _OUTCOMES:
                raise SolverAttemptLogError(f'unknown solver attempt outcome at CSV row {row_number}')
            try:
                index = int(row['attempt_index'])
            except ValueError as exc:
                raise SolverAttemptLogError(f'invalid attempt_index at CSV row {row_number}') from exc
            if index < 1:
                raise SolverAttemptLogError(f'attempt_index must be positive at CSV row {row_number}')
            expected_id = f'{self.run_id}:milp:{index:08d}'
            if row['attempt_id'] != expected_id:
                raise SolverAttemptLogError(f'attempt_id/index mismatch at CSV row {row_number}')
            indexes.append(index)
        if sorted(indexes) != list(range(1, len(rows) + 1)):
            raise SolverAttemptLogError('solver attempt indexes are not contiguous')
        return rows

    def _reserve_attempt_index(self) -> int:
        with self._lock:
            return len(self._read_rows()) + 1

    def _append(self, row: dict[str, object]) -> None:
        if tuple(row) != SOLVER_ATTEMPT_COLUMNS:
            raise SolverAttemptLogError('internal solver attempt row schema mismatch')
        with self._lock:
            durable_rows = self._read_rows()
            expected_index = len(durable_rows) + 1
            if int(row['attempt_index']) != expected_index:
                raise SolverAttemptLogError('solver attempt completion order is inconsistent with its index')
            with self.path.open('a', encoding='utf-8', newline='') as handle:
                csv.DictWriter(handle, fieldnames=SOLVER_ATTEMPT_COLUMNS, lineterminator='\n').writerow(row)
                handle.flush()
                os.fsync(handle.fileno())

    def call(self, function: Callable[[], _T], *, context: SolverAttemptContext, method: object, controller: object) -> _T:
        attempt_index = self._reserve_attempt_index()
        attempt_id = f'{self.run_id}:milp:{attempt_index:08d}'
        started_utc = _utc_now()
        started = time.perf_counter()
        try:
            result = function()
        except BaseException as exc:
            completed_utc = _utc_now()
            elapsed = time.perf_counter() - started
            exception_type = f'{type(exc).__module__}.{type(exc).__qualname__}'
            exception_text = f'{exception_type}: {exc}'
            row = self._row(context=context, attempt_id=attempt_id, attempt_index=attempt_index, started_utc=started_utc, completed_utc=completed_utc, outcome='exception', status='', solver_success='', solver_message='', exception=exception_text, exception_type=exception_type, solve_time_s=elapsed, mip_gap='', method=method, controller=controller)
            try:
                self._append(row)
            except BaseException as log_exc:
                if hasattr(exc, 'add_note'):
                    exc.add_note(f'solver attempt logging also failed: {log_exc!r}')
            raise
        completed_utc = _utc_now()
        elapsed = time.perf_counter() - started
        external_success = bool(getattr(result, 'success', False)) and getattr(result, 'x', None) is not None
        row = self._row(context=context, attempt_id=attempt_id, attempt_index=attempt_index, started_utc=started_utc, completed_utc=completed_utc, outcome='success' if external_success else 'return_failure', status=_csv_value(getattr(result, 'status', None)), solver_success=str(bool(getattr(result, 'success', False))).lower(), solver_message=_csv_value(getattr(result, 'message', None)), exception='', exception_type='', solve_time_s=elapsed, mip_gap=_optional_float(getattr(result, 'mip_gap', None)), method=method, controller=controller)
        self._append(row)
        return result

    def _row(self, *, context: SolverAttemptContext, attempt_id: str, attempt_index: int, started_utc: str, completed_utc: str, outcome: str, status: object, solver_success: object, solver_message: object, exception: str, exception_type: str, solve_time_s: float, mip_gap: object, method: object, controller: object) -> dict[str, object]:
        return {'run_id': self.run_id, 'run_kind': self.run_kind, 'attempt_id': attempt_id, 'attempt_index': attempt_index, 'decision_step': _csv_value(context.decision_step), 'decision_timestamp': _csv_value(context.decision_timestamp), 'station_id': _csv_value(context.station_id), 'station_index': _csv_value(context.station_index), 'profile_id': _csv_value(context.profile_id), 'variant_id': _csv_value(context.variant_id), 'target_window': _csv_value(context.target_window), 'started_utc': started_utc, 'completed_utc': completed_utc, 'outcome': outcome, 'status': status, 'solver_success': solver_success, 'solver_message': solver_message, 'exception': exception, 'exception_type': exception_type, 'solve_time_s': solve_time_s, 'mip_gap': mip_gap, 'method': _csv_value(method), 'controller': _csv_value(controller)}

    def summary(self, expected_milp_solves: int | None=None) -> dict[str, object]:
        if expected_milp_solves is not None:
            if isinstance(expected_milp_solves, bool) or not isinstance(expected_milp_solves, int):
                raise TypeError('expected_milp_solves must be an integer or None')
            if expected_milp_solves < 0:
                raise ValueError('expected_milp_solves must be nonnegative')
        with self._lock:
            rows = self._read_rows()
            success_count = sum((row['outcome'] == 'success' for row in rows))
            return_failure_count = sum((row['outcome'] == 'return_failure' for row in rows))
            exception_count = sum((row['outcome'] == 'exception' for row in rows))
            gaps = []
            for row in rows:
                if not row['mip_gap']:
                    continue
                try:
                    value = float(row['mip_gap'])
                except ValueError as exc:
                    raise SolverAttemptLogError('invalid mip_gap in attempt log') from exc
                if not math.isfinite(value):
                    raise SolverAttemptLogError('non-finite mip_gap in attempt log')
                gaps.append(value)
            attempt_count = len(rows)
            expected_match = None if expected_milp_solves is None else attempt_count == expected_milp_solves
            log_sha256 = _sha256_file(self.path)
        failure_count = return_failure_count + exception_count
        return {'run_id': self.run_id, 'run_kind': self.run_kind, 'solver_attempt_count': attempt_count, 'expected_milp_solves': expected_milp_solves, 'expected_match': expected_match, 'expected_milp_solves_match': expected_match, 'external_solve_count_matches_expected': expected_match, 'solver_success_count': success_count, 'solver_failure_count': failure_count, 'solver_return_failure_count': return_failure_count, 'solver_exception_count': exception_count, 'max_reported_mip_gap': max(gaps) if gaps else None, 'exact_external_solve_count_supported': True, 'solver_attempt_log_path': str(self.path), 'solver_attempt_log_sha256': log_sha256, 'solver_attempts_sha256': log_sha256}

@contextmanager
def solver_attempt_scope(logger: SolverAttemptLogger, *, decision_step: int | None, decision_timestamp: object | None, station_id: object | None=None, station_index: int | None=None, profile_id: object | None=None, variant_id: object | None=None, target_window: object | None=None) -> Iterator[SolverAttemptLogger]:
    if not isinstance(logger, SolverAttemptLogger):
        raise TypeError('logger must be a SolverAttemptLogger')
    if decision_step is not None and (isinstance(decision_step, bool) or not isinstance(decision_step, int)):
        raise TypeError('decision_step must be an integer or None')
    if station_index is not None and (isinstance(station_index, bool) or not isinstance(station_index, int)):
        raise TypeError('station_index must be an integer or None')
    context = SolverAttemptContext(logger=logger, decision_step=decision_step, decision_timestamp=decision_timestamp, station_id=station_id, station_index=station_index, profile_id=profile_id, variant_id=variant_id, target_window=target_window)
    token = _ACTIVE_SOLVER_ATTEMPT_CONTEXT.set(context)
    try:
        yield logger
    finally:
        _ACTIVE_SOLVER_ATTEMPT_CONTEXT.reset(token)

def active_solver_attempt_context() -> SolverAttemptContext | None:
    return _ACTIVE_SOLVER_ATTEMPT_CONTEXT.get()

def call_external_milp(function: Callable[[], _T], *, method: object, controller: object) -> _T:
    context = _ACTIVE_SOLVER_ATTEMPT_CONTEXT.get()
    if context is None:
        return function()
    return context.logger.call(function, context=context, method=method, controller=controller)
__all__ = ['SOLVER_ATTEMPT_COLUMNS', 'SolverAttemptContext', 'SolverAttemptLogError', 'SolverAttemptLogger', 'active_solver_attempt_context', 'call_external_milp', 'solver_attempt_scope']
