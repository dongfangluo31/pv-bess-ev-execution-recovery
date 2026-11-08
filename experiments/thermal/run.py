import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY / "code"))

from pv_bess_ev.thermal.controller import action, power_cap, predict, select_cap, thermal_step
from asset_adapter import make_net, run_ac
from summary import summary


def run_arm(common, case, arm, trace):
    state = case["initial_thermal_state"]
    soc = common["battery"]["soc_initial"]
    net = make_net()
    rows = []
    for step, requested in enumerate(case["request_kw"]):
        if arm == "STATIC_RATED":
            cap = dict(K_cap=1., thermal_feasible=None, prediction_count=0)
        else:
            cap = select_cap(lambda current, index: predict(common, state, case["ambient_c"], current))
            cap.pop("predictions", None)
        act = action(common, requested, cap["K_cap"], soc)
        ac = run_ac(net, requested, act["charge_kw"], act["discharge_kw"])
        th = thermal_step(common, state, case["ambient_c"], ac["actual_current_pu"])
        row = dict(arm=arm, step=step, start_s=step*60, ambient_c=case["ambient_c"],
                   requested_kw=requested, cap=cap, grid_cap_kw=power_cap(common, cap["K_cap"]),
                   action=act, AC=ac, thermal=th, thermal_pre=state, SOC_pre=soc, SOC_post=act["soc_post"])
        if trace:
            trace.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        rows.append(row)
        state, soc = th["thermal_post"], act["soc_post"]
    result = summary(rows, common, case)
    for key, value in result.pop("thermal_state_final").items():
        result["thermal_state_final__" + key] = value
    return result


def main():
    common = json.loads((HERE/"common.json").read_text(encoding="utf-8"))
    cases = json.loads((HERE/"cases.json").read_text(encoding="utf-8-sig"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all"] + [case["case_id"] for case in cases], default="all")
    parser.add_argument("--arm", choices=["all"] + common["arms"], default="all")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("choose a new report path")
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
    output = []
    for case in cases:
        if args.case not in ("all", case["case_id"]):
            continue
        for arm in common["arms"]:
            if args.arm not in ("all", arm):
                continue
            run_id = case["case_id"] + "__" + arm
            trace = (args.trace_dir/(run_id + ".jsonl")).open("x", encoding="utf-8") if args.trace_dir else None
            try:
                summary_row = run_arm(common, case, arm, trace)
            finally:
                if trace:
                    trace.close()
            output.append(dict(arm_id=run_id,
                               completed_steps=summary_row["completed_steps"],
                               throughput_kwh=summary_row["throughput_kwh"],
                               hotspot_peak_c=summary_row["hotspot_peak_c"], hotspot_exceed_seconds=summary_row["hotspot_exceed_seconds"],
                               final_stored_kwh=summary_row["final_stored_kwh"], predictions=summary_row["dynamic_peak_predictions"],
                               inventory_adjusted_cost_057_yuan=summary_row["inventory_adjusted_cost_0.57_yuan"]))
            print(run_id, summary_row["completed_steps"], "AC solves", flush=True)
    report = dict(status="COMPLETE", arm_count=len(output), AC_solves=sum(x["completed_steps"] for x in output),
                  thermal_steps=sum(x["completed_steps"] for x in output), dynamic_predictions=sum(x["predictions"] for x in output),
                  arms=output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, allow_nan=False)
    print(f"{len(output)} arms, {report['AC_solves']} AC solves, {report['dynamic_predictions']} dynamic predictions")


if __name__ == "__main__":
    main()
