# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

from pathlib import Path
import gzip, hashlib, json, math, os, time
WORK = Path(os.environ['PV_NETWORK_OUTPUT']).resolve()
STATIONS = ('S1_Bus', 'S2_ParkB', 'S3_Mall', 'S4_ExpwC', 'S5_Tech', 'S6_Indus', 'S7_Whole')
ARMS = ('SCALAR_SHARED', 'NETWORK_JOINT', 'JOINT_H24_NETWORK_MPC')
CAPS = {a: dict(MPC=21 if a != 'JOINT_H24_NETWORK_MPC' else 0, JOINT_MIP=3 if a == 'JOINT_H24_NETWORK_MPC' else 0, PROJECTION_LP=180 if a == 'NETWORK_JOINT' else 0, AC=10800, J7=180 if a == 'NETWORK_JOINT' else 0, J8=3 if a == 'JOINT_H24_NETWORK_MPC' else 0) for a in ARMS}

def safe(v):
    if isinstance(v, float) and (not math.isfinite(v)):
        return '+inf' if v > 0 else '-inf' if v < 0 else 'NaN'
    if isinstance(v, complex):
        return dict(real=v.real, imag=v.imag)
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if hasattr(v, 'tolist'):
        return safe(v.tolist())
    if hasattr(v, 'item'):
        return safe(v.item())
    return v

def encoded(v):
    return json.dumps(safe(v), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(v):
    return hashlib.sha256(encoded(v)).hexdigest()

def record(p):
    p = Path(p).resolve()
    with p.open('rb') as f:
        h = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h)

def require(r):
    if record(r['path']) != r:
        raise ValueError('Bound evidence changed ' + r['path'])
    return Path(r['path'])

def read(p):
    p = Path(p)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, v):
    p = Path(p).resolve()
    if not p.is_relative_to(WORK):
        raise ValueError('D output scope')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        if p.suffix == '.gz':
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as z:
                z.write(encoded(v) + b'\n')
        else:
            f.write(encoded(v) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    return record(p)

class Journal:

    def __init__(self, path, run, arm):
        self.path = Path(path)
        self.f = self.path.open('xb')
        self.run = run
        self.arm = arm
        self.counts = {k: 0 for k in CAPS[arm]}
        self.seq = 0

    def event(self, event_name, second, **payload):
        self.seq += 1
        obj = dict(seq=self.seq, run_id=self.run, arm=self.arm, event=event_name, second=second, wall_ns=time.perf_counter_ns(), **payload)
        self.f.write(encoded(obj) + b'\n')
        self.f.flush()
        os.fsync(self.f.fileno())
        return obj

    def attempt(self, kind, payload):
        if kind not in self.counts or self.counts[kind] >= CAPS[self.arm][kind]:
            raise RuntimeError('Frozen call cap ' + kind)
        self.counts[kind] += 1
        i = self.counts[kind]
        self.event('BEFORE_CALL', int(payload.get('second', payload.get('minute', 0) * 60)), kind=kind, call_id=i, references=payload)
        return i

    def close(self):
        self.f.close()
