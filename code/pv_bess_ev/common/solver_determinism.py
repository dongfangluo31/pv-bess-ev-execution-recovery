# EN: Frozen single-thread HiGHS determinism options (seed 20260706) and 15-decimal initial SOC canonicalization.
# 中文：HiGHS 单线程确定性求解参数（固定随机种子 20260706）与 15 位小数初始 SOC 规范化函数。

from __future__ import annotations
import math
HIGHS_RANDOM_SEED = 20260706
HIGHS_THREADS = 1
HIGHS_PARALLEL = False
SOC_CANONICAL_DECIMALS = 15
SOC_CANONICAL_MAX_DELTA = 1e-12

# EN: Canonicalize solver input SOC to 15 decimal places (within 1e-12 guard) so cross-platform float drift cannot flip MILP branches.
# 中文：在 1e-12 容差保护内将输入 SOC 规整至 15 位小数，消除跨平台浮点尾数差异导致的分支定界路径分叉。
def canonicalize_solver_soc(value: float, *, soc_min: float, soc_max: float, decimals: int=SOC_CANONICAL_DECIMALS) -> float:
    numeric = float(value)
    lower = float(soc_min)
    upper = float(soc_max)
    if not all((math.isfinite(item) for item in (numeric, lower, upper))):
        raise ValueError('SOC canonicalization inputs must be finite')
    if lower > upper:
        raise ValueError('SOC canonicalization bounds are reversed')
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals != 15:
        raise ValueError('formal solver SOC canonicalization is frozen at 15 decimals')
    if numeric < lower - SOC_CANONICAL_MAX_DELTA or numeric > upper + SOC_CANONICAL_MAX_DELTA:
        raise ValueError('SOC lies outside the canonicalization guard')
    clipped = min(max(numeric, lower), upper)
    canonical = round(clipped, decimals)
    if abs(canonical - lower) <= 10.0 ** (-decimals):
        canonical = lower
    elif abs(canonical - upper) <= 10.0 ** (-decimals):
        canonical = upper
    if abs(canonical - numeric) > SOC_CANONICAL_MAX_DELTA:
        raise ValueError('SOC canonicalization delta exceeds the frozen guard')
    return float(canonical)

def highs_determinism_options(*, random_seed: int=HIGHS_RANDOM_SEED, threads: int=HIGHS_THREADS, parallel: bool=HIGHS_PARALLEL) -> dict[str, int | bool]:
    if isinstance(random_seed, bool) or int(random_seed) != HIGHS_RANDOM_SEED:
        raise ValueError('formal HiGHS random seed is frozen at 20260706')
    if isinstance(threads, bool) or int(threads) != HIGHS_THREADS:
        raise ValueError('formal HiGHS thread count is frozen at one')
    if type(parallel) is not bool or parallel is not HIGHS_PARALLEL:
        raise ValueError('formal HiGHS parallel mode is frozen off')
    return {'random_seed': HIGHS_RANDOM_SEED, 'threads': HIGHS_THREADS, 'parallel': HIGHS_PARALLEL}
