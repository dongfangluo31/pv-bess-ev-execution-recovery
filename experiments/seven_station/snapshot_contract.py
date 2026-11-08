# EN: Causal step-opening packet schema and physical admissibility contract.
# 中文：步初因果数据包结构校验与物理可行域契约定义模块。

from pathlib import Path
import json, math, hashlib
STATIONS = ['S1_Bus', 'S2_ParkB', 'S3_Mall', 'S4_ExpwC', 'S5_Tech', 'S6_Indus', 'S7_Whole']
from location_binding import station_bus_map
BUS_MAP = station_bus_map()
POWER_TOL = 1e-07
VOLTAGE_TOL = 1e-10
PF_OPTIONS = dict(algorithm='nr', calculate_voltage_angles=True, init='flat', max_iteration=100, tolerance_mva=1e-10, enforce_q_lims=False, numba=False)
LP_OPTIONS = dict(time_limit=5.0, primal_feasibility_tolerance=1e-09, dual_feasibility_tolerance=1e-09, presolve=True, disp=False)

def require(v, m):
    if not v:
        raise RuntimeError(m)

def safe(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else '+inf' if v > 0 else '-inf' if v < 0 else 'nan'
    if isinstance(v, complex):
        return dict(real=safe(v.real), imag=safe(v.imag))
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if hasattr(v, 'tolist'):
        return safe(v.tolist())
    if hasattr(v, 'item'):
        return safe(v.item())
    raise TypeError(type(v).__name__)

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(safe(value), f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        f.write('\n')

def record(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
