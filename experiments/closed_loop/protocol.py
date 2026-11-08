# EN: Strict JSON/JSONL serialization, `allow_nan=False` validation, and SHA-256 input-record helpers.
# 中文：严格禁止 NaN 的 JSON/JSONL 读写序列化与输入文件 SHA-256 记录工具。

from pathlib import Path
import hashlib, json, os, time
from datetime import datetime, timezone

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def record(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=file_sha256(p))

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

class Log:

    def __init__(self, path, role):
        self.file = Path(path).open('x', encoding='utf-8')
        self.role = role
        self.seq = 0

    def event(self, kind, sim_s, **body):
        self.seq += 1
        row = dict(seq=self.seq, role=self.role, pid=os.getpid(), event=kind, sim_s=sim_s, wall_ns=time.perf_counter_ns(), utc=datetime.now(timezone.utc).isoformat(), **body)
        self.file.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        return row

    def close(self):
        self.file.close()

def emit(value):
    print(json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False), flush=True)

def finite_message(value):
    json.dumps(value, allow_nan=False)
    return value
