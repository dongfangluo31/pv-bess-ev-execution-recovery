# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
from snapshot_contract import safe, STATIONS, BUS_MAP, PF_OPTIONS, LP_OPTIONS
ROOT = Path(__file__).resolve().parent
LIB = ROOT
BATCH = Path(os.environ['PV_NETWORK_OUTPUT']).resolve()
ARMS = ('SCALAR_SHARED', 'NETWORK_JOINT')
MAXIMUM = dict(MPC=42, LP=180, AC=360)

def load(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def utc():
    return datetime.now(timezone.utc).isoformat()

def write(p, obj):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(safe(obj), f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def record(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(1048576), b''):
            h.update(x)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())

def require(ref):
    if record(ref['path']) != {k: ref[k] for k in ('path', 'bytes', 'sha256')}:
        raise ValueError('Identity changed: ' + ref['path'])

def digest(obj):
    return hashlib.sha256(json.dumps(safe(obj), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
