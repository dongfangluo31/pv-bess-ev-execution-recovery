# EN: Binary UART HIL frame codec (0x55 0xAA sync, little-endian Q16.16 payload, CRC-16/CCITT-FALSE).
# 中文：串口硬件在环（HIL）二进制帧编解码器（0x55 0xAA 帧同步头、小端序 Q16.16 载荷与 CRC-16/CCITT-FALSE 校验）。

from __future__ import annotations
from dataclasses import dataclass
import struct
from .fixed import FilterInput, FilterResult
SYNC = b'U\xaa'
VERSION = 1
REQUEST_TYPE = 16
RESPONSE_TYPE = 144
REQUEST_PAYLOAD_BYTES = 24
RESPONSE_PAYLOAD_BYTES = 32

class ProtocolError(ValueError):
    pass

@dataclass(frozen=True)
class DecodedFrame:
    version: int
    message_type: int
    sequence: int
    payload: bytes

def crc16_ccitt_false(data: bytes, initial: int=65535) -> int:
    crc = initial & 65535
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (crc << 1 ^ 4129) & 65535 if crc & 32768 else crc << 1 & 65535
    return crc

def encode_frame(message_type: int, sequence: int, payload: bytes) -> bytes:
    if not 0 <= message_type <= 255:
        raise ValueError('message_type must fit in one byte')
    if not 0 <= sequence <= 4294967295:
        raise ValueError('sequence must fit in u32')
    if len(payload) > 65535:
        raise ValueError('payload is too large')
    body = struct.pack('<BBHI', VERSION, message_type, len(payload), sequence) + payload
    crc = crc16_ccitt_false(body)
    return SYNC + body + struct.pack('<H', crc)

def decode_frame(frame: bytes) -> DecodedFrame:
    if len(frame) < 12:
        raise ProtocolError('frame shorter than fixed overhead')
    if frame[:2] != SYNC:
        raise ProtocolError('bad sync')
    version, message_type, payload_len, sequence = struct.unpack_from('<BBHI', frame, 2)
    expected_len = 12 + payload_len
    if len(frame) != expected_len:
        raise ProtocolError(f'length mismatch: expected {expected_len}, got {len(frame)}')
    body = frame[2:-2]
    received_crc = struct.unpack_from('<H', frame, len(frame) - 2)[0]
    expected_crc = crc16_ccitt_false(body)
    if received_crc != expected_crc:
        raise ProtocolError(f'CRC mismatch: expected 0x{expected_crc:04X}, got 0x{received_crc:04X}')
    if version != VERSION:
        raise ProtocolError(f'unsupported protocol version {version}')
    return DecodedFrame(version, message_type, sequence, frame[10:-2])

def encode_request(sequence: int, request: FilterInput) -> bytes:
    payload = struct.pack('<6i', request.nominal_import, request.nominal_charge, request.nominal_discharge, request.max_reduce_charge, request.max_add_discharge, request.feeder_limit)
    return encode_frame(REQUEST_TYPE, sequence, payload)

def decode_request(frame: bytes) -> tuple[int, FilterInput]:
    decoded = decode_frame(frame)
    if decoded.message_type != REQUEST_TYPE:
        raise ProtocolError('not a request frame')
    if len(decoded.payload) != REQUEST_PAYLOAD_BYTES:
        raise ProtocolError('request payload must be 24 bytes')
    return (decoded.sequence, FilterInput(*struct.unpack('<6i', decoded.payload)))

def encode_response(sequence: int, result: FilterResult) -> bytes:
    payload = struct.pack('<6i2I', result.reduce_charge, result.add_discharge, result.executed_charge, result.executed_discharge, result.executed_import, result.remaining_overlimit, result.status, result.latency_cycles)
    return encode_frame(RESPONSE_TYPE, sequence, payload)

def decode_response(frame: bytes) -> tuple[int, FilterResult]:
    decoded = decode_frame(frame)
    if decoded.message_type != RESPONSE_TYPE:
        raise ProtocolError('not a response frame')
    if len(decoded.payload) != RESPONSE_PAYLOAD_BYTES:
        raise ProtocolError('response payload must be 32 bytes')
    fields = struct.unpack('<6i2I', decoded.payload)
    return (decoded.sequence, FilterResult(*fields))
