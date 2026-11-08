import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from run import CASES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    for site in ("jpl","caltech","office001"):
        if not (args.raw_root/site).is_dir():
            parser.error(f"Missing site directory {site}")
    args.out.mkdir(parents=True,exist_ok=False)
    fields=["case","returncode","complete_29_days","actual_MPC","failed_hour_opportunities",
            "triggers","source_EV_kwh","PCC_excess_kwh","PCC_excess_seconds","maximum_grid_kw",
            "throughput_kwh","common_cost_yuan","final_energy_kwh","terminal_error_kwh"]
    failures=0
    with (args.out/"monthly_results.csv").open("x",encoding="utf-8",newline="") as fh:
        table=csv.DictWriter(fh,fieldnames=fields)
        table.writeheader()
        for case,(site,_,_) in CASES.items():
            destination=args.out/(case+".json")
            execution=subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("run.py")),
                                      "--case",case,"--raw-pages",str(args.raw_root/site),"--report",str(destination)])
            item=json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {}
            row={field:item.get(field) for field in fields}
            row["case"]=case
            row["returncode"]=execution.returncode
            table.writerow(row)
            fh.flush()
            failures+=int(execution.returncode!=0)
    return int(failures!=0)


if __name__=="__main__":
    sys.exit(main())
