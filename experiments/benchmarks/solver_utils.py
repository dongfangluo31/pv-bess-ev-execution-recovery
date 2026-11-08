# EN: HiGHS linear-programming wrapper and per-case output writer for external control benchmarks.
# 中文：外部公开控制基准算例（BOPTEST RC 与 IEEE 14 节点机会约束最优潮流）的 HiGHS 线性规划求解与结果写出工具。

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import scipy.optimize

BASE_SEED = 20260706
BENCH_LABELS = {
    "boptest1": "BOPTEST_TC1_RC_BUILDING",
    "roald14": "ROALD17_CC_DC_OPF_IEEE14",
}


def lp(c, A_ub, b_ub, bounds, A_eq=None, b_eq=None):
    # EN: Solve linear program via SciPy HiGHS simplex/interior-point backend and record wall-clock duration.
    # 中文：调用 SciPy HiGHS 后端求解线性规划并记录单次求解墙钟耗时。
    t0 = time.perf_counter()
    res = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )
    wall = time.perf_counter() - t0
    return res, wall


def write_case_outputs(output_root: Path, label: str, bench: str, params: dict, seed: int, solver_rows: list[dict], trajectories, metrics: dict) -> Path:
    case_dir = output_root / label
    case_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "bench": bench,
        "bench_label": BENCH_LABELS[bench],
        "label": label,
        "params": params,
        "seed": seed,
        "base_seed": BASE_SEED,
    }
    (case_dir / "config.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trajectories.to_csv(case_dir / "trajectories.csv", index=False)
    with (case_dir / "solver_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attempt", "status", "nit", "num_vars", "num_constraints", "wall_s"])
        writer.writeheader()
        for idx, row in enumerate(solver_rows, start=1):
            writer.writerow({
                "attempt": row.get("attempt", idx),
                "status": row.get("status", ""),
                "nit": row.get("nit", ""),
                "num_vars": row.get("num_vars", ""),
                "num_constraints": row.get("num_constraints", ""),
                "wall_s": row.get("wall_s", ""),
            })
    return case_dir
