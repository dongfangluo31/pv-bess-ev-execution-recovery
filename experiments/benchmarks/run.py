# EN: Command-line runner for the BOPTEST RC building thermal MPC and Roald-2017 IEEE 14-bus CC-DC-OPF benchmarks.
# 中文：BOPTEST RC 建筑热力 MPC 与 Roald-2017 IEEE 14 节点机会约束直流最优潮流（CC-DC-OPF）外部基准实验入口。

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from solver_utils import BASE_SEED
import boptest_rc
import ieee14_dcopf


def build_jobs() -> list[tuple[str, dict, int, str]]:
    jobs: list[tuple[str, dict, int, str]] = []
    for i, (theta, sigma_fc) in enumerate([(t, s) for t in (0.0, 0.25, 0.5, 0.75) for s in (1.0, 2.0, 3.0)]):
        jobs.append(("boptest1", {"theta": theta, "sigma_fc": sigma_fc}, BASE_SEED + i, f"boptest1_theta{theta}_sf{sigma_fc}"))
    for i, eps in enumerate((0.005, 0.01, 0.02, 0.05)):
        jobs.append(("roald14", {"epsilon": eps, "sigma_w_pct": 10.0, "margin": 0.0}, BASE_SEED + 100 + i, f"roald14_eps{eps}"))
    for j, sw in enumerate((5.0, 7.5, 12.5, 14.25)):
        jobs.append(("roald14", {"epsilon": 0.01, "sigma_w_pct": sw, "margin": 0.0}, BASE_SEED + 200 + j, f"roald14_sw{sw}"))
    for k, mg in enumerate((0.02, 0.05)):
        jobs.append(("roald14", {"epsilon": 0.01, "sigma_w_pct": 10.0, "margin": mg}, BASE_SEED + 300 + k, f"roald14_margin{mg}"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BOPTEST RC and IEEE 14-bus CC-DC-OPF published benchmarks.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"), help="Directory for benchmark trajectories and summary.csv.")
    parser.add_argument("--case", choices=[j[3] for j in build_jobs()], help="Optional single case label to run.")
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in build_jobs() if args.case is None or j[3] == args.case]

    summary_rows: list[dict] = []
    for bench, params, seed, label in jobs:
        if bench == "boptest1":
            _, metrics = boptest_rc.run(params, seed, label, output_root)
            summary_rows.append({
                "case_label": label,
                "benchmark": metrics["benchmark"],
                "seed": seed,
                "status": "optimal" if metrics["failed_solves"] == 0 else "partial_fallback",
                "theta_k": metrics["theta"],
                "sigma_forecast_k": metrics["sigma_fc"],
                "epsilon": "",
                "sigma_wind_pct": "",
                "line_margin": "",
                "objective_value": metrics["electrical_energy_kwh"],
                "objective_unit": "kWh_electrical",
                "realized_violation": metrics["comfort_violation_kh"],
                "violation_unit": "K_h",
                "max_line_exceedance_mw": "",
                "n_evaluations": metrics["n_steps"],
            })
        else:
            _, metrics = ieee14_dcopf.run(params, seed, label, output_root)
            summary_rows.append({
                "case_label": label,
                "benchmark": metrics["benchmark"],
                "seed": seed,
                "status": metrics["status"],
                "theta_k": "",
                "sigma_forecast_k": "",
                "epsilon": metrics["epsilon"],
                "sigma_wind_pct": metrics["sigma_w_pct"],
                "line_margin": metrics["margin"],
                "objective_value": "" if metrics["cost"] is None else metrics["cost"],
                "objective_unit": "generation_cost",
                "realized_violation": "" if metrics["realized_violation_mw"] is None else metrics["realized_violation_mw"],
                "violation_unit": "MW_expected_exceedance",
                "max_line_exceedance_mw": metrics.get("max_line_exceedance_mw", ""),
                "n_evaluations": 2000,
            })

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
