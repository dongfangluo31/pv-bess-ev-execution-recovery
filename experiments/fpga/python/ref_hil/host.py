# EN: Serial UART transaction driver for live Artix-7 FPGA hardware-in-the-loop (HIL) single-vector evaluation.
# 中文：面向 Artix-7 FPGA 开发板实时串口硬件在环（HIL）单向量交互测试的主机端通信驱动。

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Callable
from .fixed import FilterInput, FilterResult, apply_ref_filter, q_to_float
from .protocol import RESPONSE_PAYLOAD_BYTES, SYNC, ProtocolError, decode_response, encode_request
CH340_IDS = {(6790, 29987), (6790, 21795)}

@dataclass(frozen=True)
class PortIdentity:
    device: str
    vid: int | None
    pid: int | None
    serial_number: str | None = None
    manufacturer: str | None = None
    product: str | None = None
    description: str | None = None

    @property
    def vid_pid(self) -> str | None:
        if self.vid is None or self.pid is None:
            return None
        return f'{self.vid:04x}:{self.pid:04x}'

    def evidence_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['vid_pid'] = self.vid_pid
        return data

class SequenceMismatchError(RuntimeError):

    def __init__(self, sent: int, received: int):
        self.sent = sent
        self.received = received
        super().__init__(f'sequence mismatch: sent {sent}, received {received}')

def _load_serial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError('pyserial is required for live UART HIL transactions; install pyserial==3.5') from exc
    return (serial, list_ports)

def _identity_from_port(port: Any) -> PortIdentity:
    return PortIdentity(device=str(port.device), vid=getattr(port, 'vid', None), pid=getattr(port, 'pid', None), serial_number=getattr(port, 'serial_number', None), manufacturer=getattr(port, 'manufacturer', None), product=getattr(port, 'product', None), description=getattr(port, 'description', None))

def discover_port_identity(explicit: str | None=None) -> PortIdentity:
    _, list_ports = _load_serial()
    ports = list(list_ports.comports())
    if explicit:
        matches = [port for port in ports if str(port.device).casefold() == str(explicit).casefold()]
        if len(matches) != 1:
            raise RuntimeError(f'explicit UART {explicit!r} was not uniquely enumerated; available={[str(port.device) for port in ports]}')
    else:
        matches = [port for port in ports if (port.vid, port.pid) in CH340_IDS]
        if len(matches) != 1:
            raise RuntimeError(f'expected exactly one CH340 UART by VID/PID, found {[str(port.device) for port in matches]}')
    identity = _identity_from_port(matches[0])
    if (identity.vid, identity.pid) not in CH340_IDS:
        raise RuntimeError(f'UART {identity.device} is not an approved CH340 VID/PID: {identity.vid_pid}')
    return identity

def discover_port(explicit: str | None=None) -> str:
    return discover_port_identity(explicit).device

def _read_exact(port: Any, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size and time.monotonic() < deadline:
        chunk = port.read(size - len(data))
        if chunk:
            data.extend(chunk)
    if len(data) != size:
        raise TimeoutError(f'UART timeout after {len(data)}/{size} bytes')
    return bytes(data)

def read_frame(port: Any, timeout_s: float) -> bytes:
    deadline = time.monotonic() + timeout_s
    sync = bytearray()
    while time.monotonic() < deadline:
        byte = port.read(1)
        if not byte:
            continue
        sync = (sync + byte)[-2:]
        if bytes(sync) == SYNC:
            break
    else:
        raise TimeoutError('UART response sync timeout')
    fixed = _read_exact(port, 8, deadline)
    payload_len = int.from_bytes(fixed[2:4], 'little')
    if payload_len != RESPONSE_PAYLOAD_BYTES:
        raise ProtocolError(f'unexpected response payload length {payload_len}; expected {RESPONSE_PAYLOAD_BYTES}')
    remainder = _read_exact(port, payload_len + 2, deadline)
    return SYNC + fixed + remainder

def transact_on_open_port(port: Any, request: FilterInput, sequence: int, timeout_s: float) -> tuple[FilterResult, int]:
    frame = encode_request(sequence, request)
    started = time.perf_counter_ns()
    port.write(frame)
    port.flush()
    response = read_frame(port, timeout_s)
    elapsed_ns = time.perf_counter_ns() - started
    response_sequence, result = decode_response(response)
    if response_sequence != sequence:
        raise SequenceMismatchError(sequence, response_sequence)
    return (result, elapsed_ns)

class SerialSession:

    def __init__(self, identity: PortIdentity, baud: int=115200, read_timeout_s: float=0.02, serial_factory: Callable[..., Any] | None=None) -> None:
        self.identity = identity
        self.baud = baud
        self.read_timeout_s = read_timeout_s
        self._serial_factory = serial_factory
        self._port: Any | None = None

    def __enter__(self) -> 'SerialSession':
        if self._port is not None:
            raise RuntimeError('serial session is already open')
        if self._serial_factory is None:
            serial, _ = _load_serial()
            factory = serial.Serial
        else:
            factory = self._serial_factory
        self._port = factory(self.identity.device, baudrate=self.baud, timeout=self.read_timeout_s)
        self._port.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None

    def transact(self, request: FilterInput, sequence: int, timeout_s: float) -> tuple[FilterResult, int]:
        if self._port is None:
            raise RuntimeError('serial session is not open')
        return transact_on_open_port(self._port, request, sequence, timeout_s)

def transact(request: FilterInput, sequence: int, port_name: str, baud: int, timeout_s: float) -> tuple[FilterResult, int]:
    identity = PortIdentity(port_name, None, None)
    with SerialSession(identity, baud=baud) as session:
        return session.transact(request, sequence, timeout_s)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port')
    parser.add_argument('--baud', type=int, choices=(115200,), default=115200)
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--sequence', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--nominal-import', required=True)
    parser.add_argument('--nominal-charge', required=True)
    parser.add_argument('--nominal-discharge', required=True)
    parser.add_argument('--max-reduce-charge', required=True)
    parser.add_argument('--max-add-discharge', required=True)
    parser.add_argument('--feeder-limit', required=True)
    args = parser.parse_args()
    request = FilterInput.from_kw([args.nominal_import, args.nominal_charge, args.nominal_discharge, args.max_reduce_charge, args.max_add_discharge, args.feeder_limit])
    golden = apply_ref_filter(request)
    identity: PortIdentity | None = None
    if args.dry_run:
        result = golden
        elapsed_ns = None
    else:
        identity = discover_port_identity(args.port)
        with SerialSession(identity, baud=args.baud) as session:
            result, elapsed_ns = session.transact(request, args.sequence, args.timeout)
        if result != golden:
            raise RuntimeError(f'FPGA mismatch\nexpected={golden}\nreceived={result}')
    output = {'port_identity': identity.evidence_dict() if identity else None, 'sequence': args.sequence, 'request_q16_16': asdict(request), 'result_q16_16': asdict(result), 'result_kw': {key: q_to_float(value) for key, value in asdict(result).items() if key not in {'status', 'latency_cycles'}}, 'round_trip_ns': elapsed_ns, 'bit_exact_with_golden': result == golden, 'board_programmed_by_this_tool': False}
    print(json.dumps(output, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
