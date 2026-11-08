# EN: Full-corpus UART hardware-in-the-loop (HIL) vector replay, bit-exactness verification, and round-trip latency statistics.
# 中文：全量测试向量串口硬件在环（HIL）重放、逐位精确性比对与往返时延分布统计。

from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable
from .fixed import FilterInput, FilterResult, apply_ref_filter
from .host import PortIdentity, SequenceMismatchError, SerialSession, discover_port_identity
from .protocol import ProtocolError
OFFICIAL_VECTOR_COUNT = 5009
MAX_LSB_ERROR = 1
MAX_CORE_CYCLES = 32
RESULT_VALUE_FIELDS = ('reduce_charge', 'add_discharge', 'executed_charge', 'executed_discharge', 'executed_import', 'remaining_overlimit')

def _signed(word: str) -> int:
    value = int(word, 16)
    return value - (1 << 32) if value & 1 << 31 else value

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _percentile(values: list[int], percent: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int((percent * len(ordered) + 99) // 100))
    return ordered[min(rank - 1, len(ordered) - 1)]

def load_corpus(vectors_path: Path, summary_path: Path, *, official: bool, diagnostic_limit: int | None=None) -> tuple[list[FilterInput], dict[str, Any]]:
    if official and diagnostic_limit is not None:
        raise ValueError('official 5,009-vector replay forbids any limit')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    actual_sha = _sha256(vectors_path)
    if summary.get('sha256') != actual_sha:
        raise ValueError(f"vector corpus SHA-256 mismatch: summary={summary.get('sha256')} actual={actual_sha}")
    requests: list[FilterInput] = []
    with vectors_path.open('r', encoding='ascii') as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 13:
                raise ValueError(f'malformed vector line {line_number}: expected 13 fields, got {len(fields)}')
            values = [_signed(word) for word in fields]
            request = FilterInput(*values[:6])
            golden = apply_ref_filter(request)
            embedded = (golden.reduce_charge, golden.add_discharge, golden.executed_charge, golden.executed_discharge, golden.executed_import, golden.remaining_overlimit, golden.status)
            if tuple(values[6:]) != embedded:
                raise ValueError(f'embedded golden mismatch at vector line {line_number}')
            requests.append(request)
    if summary.get('total_vectors') != len(requests):
        raise ValueError(f"vector count mismatch: summary={summary.get('total_vectors')} actual={len(requests)}")
    if official and len(requests) != OFFICIAL_VECTOR_COUNT:
        raise ValueError(f'official replay requires exactly {OFFICIAL_VECTOR_COUNT} vectors, got {len(requests)}')
    if diagnostic_limit is not None:
        if diagnostic_limit <= 0:
            raise ValueError('diagnostic limit must be positive')
        requests = requests[:diagnostic_limit]
    return (requests, {**summary, 'actual_sha256': actual_sha, 'replay_vectors': len(requests)})

def _comparison(expected: FilterResult, received: FilterResult) -> dict[str, Any]:
    differences = {name: abs(getattr(expected, name) - getattr(received, name)) for name in RESULT_VALUE_FIELDS}
    max_lsb = max(differences.values(), default=0)
    return {'differences_lsb': differences, 'max_abs_lsb': max_lsb, 'status_match': received.status == expected.status, 'latency_cycles': received.latency_cycles, 'latency_within_limit': 0 <= received.latency_cycles <= MAX_CORE_CYCLES, 'bit_exact': received == expected, 'accepted': max_lsb <= MAX_LSB_ERROR and received.status == expected.status and (0 <= received.latency_cycles <= MAX_CORE_CYCLES)}

def replay_corpus(requests: Iterable[FilterInput], *, session: Any, identity: PortIdentity, timeout_s: float, output_dir: Path, corpus: dict[str, Any], bitstream_path: Path, jtag_device: str, official: bool) -> dict[str, Any]:
    if not bitstream_path.is_file():
        raise ValueError(f'bitstream does not exist: {bitstream_path}')
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = output_dir / 'replay_ledger.jsonl'
    request_list = list(requests)
    counts = {'attempted': 0, 'responses': 0, 'accepted': 0, 'bit_exact': 0, 'value_limit_failures': 0, 'status_failures': 0, 'latency_failures': 0, 'timeouts': 0, 'dropped': 0, 'crc_errors': 0, 'protocol_errors': 0, 'sequence_errors': 0, 'out_of_order': 0, 'unexpected_errors': 0}
    round_trips: list[int] = []
    core_cycles: list[int] = []
    max_lsb = 0
    session_error: str | None = None
    with ledger_path.open('w', encoding='utf-8', newline='\n') as ledger:
        try:
            with session as active:
                for sequence, request in enumerate(request_list, start=1):
                    counts['attempted'] += 1
                    row: dict[str, Any] = {'vector_index': sequence, 'sent_sequence': sequence, 'request_q16_16': asdict(request), 'expected_q16_16': asdict(apply_ref_filter(request))}
                    try:
                        expected = apply_ref_filter(request)
                        received, elapsed_ns = active.transact(request, sequence, timeout_s)
                        comparison = _comparison(expected, received)
                        counts['responses'] += 1
                        counts['accepted'] += int(comparison['accepted'])
                        counts['bit_exact'] += int(comparison['bit_exact'])
                        counts['value_limit_failures'] += int(comparison['max_abs_lsb'] > MAX_LSB_ERROR)
                        counts['status_failures'] += int(not comparison['status_match'])
                        counts['latency_failures'] += int(not comparison['latency_within_limit'])
                        max_lsb = max(max_lsb, comparison['max_abs_lsb'])
                        round_trips.append(elapsed_ns)
                        core_cycles.append(received.latency_cycles)
                        row.update({'outcome': 'PASS' if comparison['accepted'] else 'FAIL', 'received_q16_16': asdict(received), 'round_trip_ns': elapsed_ns, **comparison})
                    except TimeoutError as exc:
                        counts['timeouts'] += 1
                        counts['dropped'] += 1
                        row.update(outcome='FAIL', error_type='TIMEOUT_OR_DROP', error=str(exc))
                    except SequenceMismatchError as exc:
                        counts['sequence_errors'] += 1
                        counts['out_of_order'] += 1
                        row.update(outcome='FAIL', error_type='SEQUENCE_OR_ORDER', received_sequence=exc.received, error=str(exc))
                    except ProtocolError as exc:
                        counts['protocol_errors'] += 1
                        if 'crc' in str(exc).casefold():
                            counts['crc_errors'] += 1
                        row.update(outcome='FAIL', error_type='CRC' if 'crc' in str(exc).casefold() else 'PROTOCOL', error=str(exc))
                    except Exception as exc:
                        counts['unexpected_errors'] += 1
                        row.update(outcome='FAIL', error_type=type(exc).__name__, error=str(exc))
                    ledger.write(json.dumps(row, sort_keys=True) + '\n')
                    ledger.flush()
        except Exception as exc:
            session_error = f'{type(exc).__name__}: {exc}'
            ledger.write(json.dumps({'outcome': 'FAIL', 'error_type': 'SESSION', 'error': session_error}, sort_keys=True) + '\n')
    complete_count = counts['attempted'] == len(request_list)
    no_failures = counts['accepted'] == len(request_list) and counts['responses'] == len(request_list) and (not session_error)
    official_count_ok = not official or len(request_list) == OFFICIAL_VECTOR_COUNT
    verdict = 'PASS' if complete_count and no_failures and official_count_ok else 'FAIL'
    summary = {'schema_version': 2, 'suite': 'board_hil_replay', 'verdict': verdict, 'mode': 'OFFICIAL_5009' if official else 'DIAGNOSTIC', 'acceptance': {'max_abs_lsb': MAX_LSB_ERROR, 'max_core_cycles': MAX_CORE_CYCLES, 'zero_crc_errors': True, 'zero_drops': True, 'zero_out_of_order': True}, 'counts': counts, 'expected_vectors': len(request_list), 'official_count_ok': official_count_ok, 'max_abs_lsb': max_lsb, 'core_cycles': {'min': min(core_cycles) if core_cycles else None, 'max': max(core_cycles) if core_cycles else None}, 'round_trip_ns': {'min': min(round_trips) if round_trips else None, 'median': int(statistics.median(round_trips)) if round_trips else None, 'p95': _percentile(round_trips, 95), 'max': max(round_trips) if round_trips else None}, 'corpus': corpus, 'bitstream': {'path': str(bitstream_path.resolve()), 'sha256': _sha256(bitstream_path), 'bytes': bitstream_path.stat().st_size}, 'jtag_device_metadata': jtag_device, 'uart': {**identity.evidence_dict(), 'protocol': '115200-8N1-little-endian-CRC16-CCITT-FALSE'}, 'session_error': session_error, 'board_programmed_by_this_tool': False, 'jtag_opened_by_this_tool': False, 'programming_action': 'NOT_PERFORMED', 'ledger': str(ledger_path.resolve())}
    (output_dir / 'replay_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--vectors-summary', type=Path, required=True)
    parser.add_argument('--bitstream', type=Path, required=True)
    parser.add_argument('--jtag-device', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--port')
    parser.add_argument('--baud', type=int, choices=(115200,), default=115200)
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--diagnostic-limit', type=int)
    args = parser.parse_args()
    official = not args.diagnostic
    requests, corpus = load_corpus(args.vectors, args.vectors_summary, official=official, diagnostic_limit=args.diagnostic_limit)
    identity = discover_port_identity(args.port)
    session = SerialSession(identity, baud=args.baud)
    summary = replay_corpus(requests, session=session, identity=identity, timeout_s=args.timeout, output_dir=args.output_dir, corpus=corpus, bitstream_path=args.bitstream, jtag_device=args.jtag_device, official=official)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary['verdict'] != 'PASS':
        raise SystemExit(1)
if __name__ == '__main__':
    main()
