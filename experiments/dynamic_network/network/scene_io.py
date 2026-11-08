# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

from records import read as load, write, record, require, safe, STATIONS
from snapshot_contract import BUS_MAP, PF_OPTIONS, LP_OPTIONS
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parent
LIB = ROOT

def digest(x):
    return hashlib.sha256(json.dumps(safe(x), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
