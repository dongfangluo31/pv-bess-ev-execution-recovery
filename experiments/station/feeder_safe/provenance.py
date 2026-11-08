# EN: Atomic JSON/CSV writer, SHA-256 file manifest generator, and run-directory lifecycle state tracker.
# 中文：原子化 JSON/CSV 写入、SHA-256 文件清单生成与运行目录状态机管理。

from __future__ import annotations
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4
REVISION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get('PV_STATION_OUTPUT', str(REVISION_ROOT / 'outputs'))).resolve()
DEFAULT_RUN_ROOT = PROJECT_ROOT / 'runs' / 'station_runs'
DEFAULT_TEMP_ROOT = PROJECT_ROOT / 'tmp' / 'station_runs'
RUN_SCHEMA_VERSION = 1
CANONICAL_SOURCE_SUFFIXES = ('.py', '.json', '.toml', '.txt', '.md', '.ps1', '.sv', '.xdc', '.tcl', '.sby')
CANONICAL_SOURCE_EXCLUDED_PARTS = ('__pycache__', '.git', '.test_tmp', 'build', 'dist')
RUN_SUBDIRECTORIES = ('config', 'manifests', 'inputs', 'traces', 'solver', 'metrics', 'audits', 'logs')
_RUN_ID = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
_FINAL_STATES = {'COMPLETE', 'FAILED'}

class ProvenanceError(RuntimeError):
    pass

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_within_output_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    workspace = Path(os.environ.get('PV_STATION_OUTPUT', str(REVISION_ROOT / 'outputs'))).resolve().resolve()
    if resolved != workspace and resolved.is_relative_to(workspace):
        return resolved
    raise ProvenanceError(f'Outputs must remain inside {workspace}, got {resolved}')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def _atomic_text(path: Path, text: str, *, replace: bool=False) -> None:
    ensure_within_output_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not replace):
        raise FileExistsError(f'refusing to overwrite {path}')
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_write_json(path: Path, value: Mapping[str, object], *, replace: bool=False) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2), replace=replace)

def file_manifest_rows(paths: Iterable[Path], *, base: Path | None=None, role: str='artifact') -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((Path(value).resolve() for value in paths), key=lambda value: str(value).casefold()):
        if not path.is_file():
            raise ProvenanceError(f'manifest input is not a file: {path}')
        display = path.relative_to(base.resolve()).as_posix() if base and path.is_relative_to(base.resolve()) else str(path)
        rows.append({'role': role, 'path': display, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return rows

def tree_manifest_rows(root: Path, *, suffixes: Sequence[str] | None=None, excluded_parts: Sequence[str]=('__pycache__', '.git', '.test_tmp', 'build', 'dist'), role: str='code') -> list[dict[str, object]]:
    root = root.resolve()
    suffix_set = {value.casefold() for value in suffixes} if suffixes else None
    excluded_part_set = {value.casefold() for value in excluded_parts}
    files = []
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if not path.is_file() or any((part.casefold() in excluded_part_set for part in relative.parts)):
            continue
        if suffix_set is not None and path.suffix.casefold() not in suffix_set:
            continue
        files.append(path)
    return file_manifest_rows(files, base=root, role=role)

def strict_tree_manifest_rows(root: Path, *, suffixes: Sequence[str], excluded_parts: Sequence[str], role: str) -> list[dict[str, object]]:
    resolved_root = Path(root).resolve(strict=True)
    suffix_set = {value.casefold() for value in suffixes}
    excluded_set = {value.casefold() for value in excluded_parts}
    rows: list[dict[str, object]] = []
    for path in resolved_root.rglob('*'):
        relative = path.relative_to(resolved_root)
        if any((part.casefold() in excluded_set for part in relative.parts)):
            continue
        is_junction = getattr(path, 'is_junction', None)
        if path.is_symlink() or bool(is_junction and is_junction()):
            raise ProvenanceError(f'strict tree manifest refuses symlink or junction: {path}')
        if not path.is_file() or path.suffix.casefold() not in suffix_set:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise ProvenanceError(f'strict tree manifest path escapes root: {path} -> {resolved}')
        rows.append({'role': role, 'path': relative.as_posix(), 'bytes': resolved.stat().st_size, 'sha256': sha256_file(resolved)})
    _reject_casefold_duplicate_manifest_paths(rows)
    rows.sort(key=lambda row: str(row['path']).casefold())
    return rows

def _reject_casefold_duplicate_manifest_paths(rows: Sequence[Mapping[str, object]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        stored_path = str(row.get('path', ''))
        key = stored_path.casefold()
        previous = seen.get(key)
        if previous is not None:
            raise ProvenanceError(f'canonical source identity contains casefold-duplicate paths: {previous!r} and {stored_path!r}')
        seen[key] = stored_path

def revision_source_manifest_rows(*, root: Path=REVISION_ROOT, role: str='revision_code') -> list[dict[str, object]]:
    resolved_root = Path(root).resolve(strict=True)
    excluded = {value.casefold() for value in CANONICAL_SOURCE_EXCLUDED_PARTS}
    for path in resolved_root.rglob('*'):
        relative = path.relative_to(resolved_root)
        if any((part.casefold() in excluded for part in relative.parts)):
            continue
        is_junction = getattr(path, 'is_junction', None)
        if path.is_symlink() or bool(is_junction and is_junction()):
            raise ProvenanceError(f'canonical revision source identity refuses symlink or junction: {path}')
    return strict_tree_manifest_rows(resolved_root, suffixes=CANONICAL_SOURCE_SUFFIXES, excluded_parts=CANONICAL_SOURCE_EXCLUDED_PARTS, role=role)

def write_manifest_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProvenanceError('cannot write an empty file manifest')
    columns = list(rows[0].keys())
    for row in rows:
        if list(row.keys()) != columns:
            raise ProvenanceError('manifest rows must have a stable column order')
    output: list[str] = []
    from io import StringIO
    buffer = StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    output.append(buffer.getvalue())
    ensure_within_output_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_bytes(''.join(output).encode('utf-8'))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def verify_manifest_csv(path: Path, *, base: Path | None=None) -> list[str]:
    discrepancies: list[str] = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw = Path(row['path'])
        candidate = raw if raw.is_absolute() else (base or path.parent) / raw
        if not candidate.is_file():
            discrepancies.append(f'missing:{candidate}')
            continue
        actual_bytes = candidate.stat().st_size
        actual_hash = sha256_file(candidate)
        if actual_bytes != int(row['bytes']):
            discrepancies.append(f"bytes:{candidate}:{actual_bytes}!={row['bytes']}")
        if actual_hash.casefold() != row['sha256'].casefold():
            discrepancies.append(f"sha256:{candidate}:{actual_hash}!={row['sha256']}")
    return discrepancies

def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions

def _highs_version() -> str | None:
    try:
        from scipy.optimize._highspy import _core
        return '.'.join((str(value) for value in (_core.HIGHS_VERSION_MAJOR, _core.HIGHS_VERSION_MINOR, _core.HIGHS_VERSION_PATCH)))
    except Exception:
        return None

def _git_metadata(root: Path) -> dict[str, object]:

    def call(*args: str) -> str | None:
        try:
            completed = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True, check=False, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None
    commit = call('rev-parse', 'HEAD')
    status = call('status', '--porcelain=v1')
    return {'repository_detected': commit is not None, 'commit': commit, 'dirty': None if status is None else bool(status), 'status_porcelain': status}

def capture_environment(*, seed: int=20260706, command: Sequence[str] | None=None) -> dict[str, object]:
    command_vector = list(command if command is not None else sys.argv)
    declared_parent_pid = os.environ.get('FEEDER_SAFE_LAUNCH_PARENT_PID')
    try:
        declared_parent_pid_value = int(declared_parent_pid) if declared_parent_pid is not None else None
    except ValueError:
        declared_parent_pid_value = None
    process_execution = {'role': os.environ.get('FEEDER_SAFE_PROCESS_ROLE', 'in_process'), 'launch_id': os.environ.get('FEEDER_SAFE_LAUNCH_ID'), 'pid': os.getpid(), 'parent_pid': os.getppid(), 'declared_parent_pid': declared_parent_pid_value, 'command_sha256': canonical_json_sha256(command_vector), 'declared_command_sha256': os.environ.get('FEEDER_SAFE_LAUNCH_COMMAND_SHA256')}
    return {'schema_version': 1, 'captured_at_utc': utc_now(), 'python': {'version': sys.version, 'version_info': [sys.version_info.major, sys.version_info.minor, sys.version_info.micro], 'executable': sys.executable, 'base_executable': getattr(sys, '_base_executable', None), 'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'implementation': platform.python_implementation()}, 'platform': {'platform': platform.platform(), 'system': platform.system(), 'release': platform.release(), 'version': platform.version(), 'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpu_count': os.cpu_count()}, 'packages': _package_versions(('numpy', 'pandas', 'scipy', 'matplotlib', 'pandapower', 'python-docx', 'pyserial')), 'solvers': {'scipy_milp_backend': 'HiGHS', 'highs_version': _highs_version(), 'explicit_mip_feasibility_tolerance': 1e-09, 'default_primal_feasibility_tolerance': 1e-07, 'default_dual_feasibility_tolerance': 1e-07}, 'seed_policy': {'verification_and_bootstrap_seed': int(seed), 'deterministic_mpc_seed_applicable': False, 'pythonhashseed': os.environ.get('PYTHONHASHSEED')}, 'time_policy': {'target_timezone': 'Asia/Shanghai', 'target_csv_timestamps': 'naive', 'pv_source_timezone': 'UTC', 'pv_calendar_timezone': 'Asia/Shanghai', 'pv_timezone_conversion_applied': True, 'pv_calendar_mapping': 'same_month_day_hour_after_utc_to_asia_shanghai'}, 'storage_policy': {'run_root': str(DEFAULT_RUN_ROOT), 'temporary_root': str(DEFAULT_TEMP_ROOT), 'temp_environment': os.environ.get('TEMP'), 'tmp_environment': os.environ.get('TMP'), 'tmpdir_environment': os.environ.get('TMPDIR'), 'python_pycache_prefix': os.environ.get('PYTHONPYCACHEPREFIX'), 'matplotlib_config_dir': os.environ.get('MPLCONFIGDIR'), 'numba_cache_dir': os.environ.get('NUMBA_CACHE_DIR')}, 'process_execution': process_execution, 'working_directory': str(Path.cwd()), 'command': command_vector, 'git': _git_metadata(PROJECT_ROOT)}

@dataclass
class RunDirectory:
    root: Path
    run_id: str

    @classmethod
    def create(cls, run_id: str, run_spec: Mapping[str, object], *, run_root: Path=DEFAULT_RUN_ROOT, seed: int=20260706) -> 'RunDirectory':
        if not _RUN_ID.fullmatch(run_id):
            raise ProvenanceError(f'invalid run_id {run_id!r}')
        run_root = ensure_within_output_root(run_root)
        root = ensure_within_output_root(run_root / run_id)
        if root.exists():
            raise FileExistsError(f'completed or partial run directory cannot be reused: {root}')
        root.mkdir(parents=True, exist_ok=False)
        for name in RUN_SUBDIRECTORIES:
            (root / name).mkdir()
        run = cls(root=root, run_id=run_id)
        atomic_write_json(root / 'config' / 'run_spec.json', dict(run_spec))
        atomic_write_json(root / 'manifests' / 'environment.json', capture_environment(seed=seed))
        code_rows = revision_source_manifest_rows(root=REVISION_ROOT)
        write_manifest_csv(root / 'manifests' / 'code_files.csv', code_rows)
        _atomic_text(root / 'anomalies.csv', 'recorded_at_utc,anomaly_id,severity,stage,status,message,evidence\n')
        run._write_state({'schema_version': RUN_SCHEMA_VERSION, 'run_id': run_id, 'state': 'CREATED', 'created_at_utc': utc_now(), 'updated_at_utc': utc_now(), 'run_spec_sha256': canonical_json_sha256(run_spec), 'history': [{'state': 'CREATED', 'at_utc': utc_now()}]})
        return run

    @property
    def state_path(self) -> Path:
        return self.root / 'RUN_STATE.json'

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise ProvenanceError(f'missing run state: {self.state_path}')
        return json.loads(self.state_path.read_text(encoding='utf-8'))

    def _write_state(self, state: Mapping[str, object]) -> None:
        atomic_write_json(self.state_path, dict(state), replace=self.state_path.exists())

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ProvenanceError(f'path escapes run directory: {relative_path}')
        return candidate

    def transition(self, new_state: str, *, detail: str | None=None) -> None:
        new_state = new_state.upper()
        allowed = {'CREATED': {'RUNNING', 'FAILED'}, 'RUNNING': {'COMPLETE', 'FAILED'}, 'FAILED': set(), 'COMPLETE': set()}
        state = self._read_state()
        current = str(state['state'])
        if new_state not in allowed.get(current, set()):
            raise ProvenanceError(f'invalid run-state transition {current}->{new_state}')
        now = utc_now()
        event: dict[str, object] = {'state': new_state, 'at_utc': now}
        if detail:
            event['detail'] = detail
        state['state'] = new_state
        state['updated_at_utc'] = now
        state.setdefault('history', []).append(event)
        self._write_state(state)

    def write_json(self, relative_path: str | Path, value: Mapping[str, object]) -> Path:
        path = self.resolve(relative_path)
        atomic_write_json(path, dict(value))
        return path

    def record_input_manifest(self, paths: Mapping[str, Path]) -> Path:
        rows: list[dict[str, object]] = []
        for role, path in paths.items():
            rows.extend(file_manifest_rows([path], base=self.root, role=role))
        output = self.root / 'manifests' / 'input_files.csv'
        write_manifest_csv(output, rows)
        return output

    def record_anomaly(self, anomaly_id: str, severity: str, stage: str, message: str, *, evidence: str='', status: str='OPEN') -> None:
        state = self._read_state()
        if state['state'] in _FINAL_STATES:
            raise ProvenanceError('cannot append anomalies after a run is final')
        path = self.root / 'anomalies.csv'
        with path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow([utc_now(), anomaly_id, severity, stage, status, message, evidence])

    def finalize(self, *, required_outputs: Sequence[str | Path]=(), completion_fields: Mapping[str, object] | None=None, pre_complete_check: Callable[[], None] | None=None) -> Path:
        state = self._read_state()
        if state['state'] != 'RUNNING':
            raise ProvenanceError('run must be RUNNING before it can be finalized')
        reserved_completion_fields = {'schema_version', 'run_id', 'completed_at_utc', 'state_file_sha256', 'output_manifest_sha256', 'artifact_count', 'immutable'}
        if completion_fields:
            reserved = reserved_completion_fields.intersection(completion_fields)
            if reserved:
                raise ProvenanceError(f'completion fields cannot replace reserved keys: {sorted(reserved)}')
        missing = [str(path) for path in required_outputs if not self.resolve(path).is_file()]
        if missing:
            raise ProvenanceError(f'required run outputs are missing: {missing}')
        exclusions = {self.state_path.resolve(), (self.root / 'RUN_COMPLETE.json').resolve(), (self.root / 'manifests' / 'output_files.csv').resolve()}
        files = [path for path in self.root.rglob('*') if path.is_file() and path.resolve() not in exclusions]
        rows = file_manifest_rows(files, base=self.root, role='run_artifact')
        manifest_path = self.root / 'manifests' / 'output_files.csv'
        write_manifest_csv(manifest_path, rows)
        manifest_sha256 = sha256_file(manifest_path)
        if pre_complete_check is not None:
            pre_complete_check()
        post_check_files = [path for path in self.root.rglob('*') if path.is_file() and path.resolve() not in exclusions]
        post_check_rows = file_manifest_rows(post_check_files, base=self.root, role='run_artifact')
        if post_check_rows != rows:
            raise ProvenanceError('run artifacts changed after the pre-COMPLETE callback')
        if sha256_file(manifest_path) != manifest_sha256:
            raise ProvenanceError('output manifest changed after the pre-COMPLETE callback')
        self.transition('COMPLETE')
        completion = {'schema_version': RUN_SCHEMA_VERSION, 'run_id': self.run_id, 'completed_at_utc': utc_now(), 'state_file_sha256': sha256_file(self.state_path), 'output_manifest_sha256': sha256_file(manifest_path), 'artifact_count': len(rows), 'immutable': True}
        if completion_fields:
            completion.update(dict(completion_fields))
        complete_path = self.root / 'RUN_COMPLETE.json'
        atomic_write_json(complete_path, completion)
        return complete_path

def seal_failed_run(run: RunDirectory, *, detail: str, role: str='failed_run_artifact') -> Path:
    state = run._read_state()
    current = str(state.get('state', '')).upper()
    if current in {'CREATED', 'RUNNING'}:
        run.transition('FAILED', detail=detail)
    elif current != 'FAILED':
        raise ProvenanceError(f'cannot seal final state {current!r} as FAILED')
    failed_manifest = run.resolve('manifests/failed_output_files.csv')
    failed_marker = run.resolve('RUN_FAILED.json')
    if failed_manifest.exists() or failed_marker.exists():
        raise FileExistsError('failed-run evidence is already sealed')
    exclusions = {run.state_path.resolve(), failed_manifest.resolve(), failed_marker.resolve()}
    files = [path for path in run.root.rglob('*') if path.is_file() and path.resolve() not in exclusions]
    if not files:
        raise ProvenanceError('failed run has no artifacts to seal')
    write_manifest_csv(failed_manifest, file_manifest_rows(files, base=run.root, role=role))
    atomic_write_json(failed_marker, {'schema_version': RUN_SCHEMA_VERSION, 'run_id': run.run_id, 'state': 'FAILED', 'immutable': True, 'detail': detail, 'state_file_sha256': sha256_file(run.state_path), 'failed_output_manifest_sha256': sha256_file(failed_manifest), 'artifact_count': len(files)})
    return failed_marker

def make_run_id(spec_id: str, *, when: datetime | None=None) -> str:
    if not _RUN_ID.fullmatch(spec_id):
        raise ProvenanceError(f'invalid spec_id {spec_id!r}')
    stamp = (when or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%SZ')
    return f'{stamp}_{spec_id}'
__all__ = ['CANONICAL_SOURCE_EXCLUDED_PARTS', 'CANONICAL_SOURCE_SUFFIXES', 'DEFAULT_RUN_ROOT', 'DEFAULT_TEMP_ROOT', 'ProvenanceError', 'RUN_SCHEMA_VERSION', 'RunDirectory', 'atomic_write_json', 'canonical_json_sha256', 'capture_environment', 'ensure_within_output_root', 'file_manifest_rows', 'make_run_id', 'revision_source_manifest_rows', 'seal_failed_run', 'sha256_file', 'tree_manifest_rows', 'verify_manifest_csv', 'write_manifest_csv']
