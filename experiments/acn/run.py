import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import csv
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/"code"))

import numpy as np
from pv_bess_ev.common.config import ExperimentConfig, Method
from pv_bess_ev.common import model
from control import BATTERY, END, choose_minute, integrate_minute
from governor_policy import select as govern
from inventory_policy import select as inventory
from interval_plant import integrate_interval
from event_target_policy import choose_event_target
from packets import from_raw_pages, from_recorded_events, service_intervals
from service import ServiceLedger

CASES={}
for site in ("caltech","jpl","office001"):
    for arm in ("F0","REF","REF_RECOVERY"):
        CASES[site.upper()+"_"+arm]=(site,"MINUTE",arm)
    CASES[site.upper()+"_EVENT_INVENTORY"]=(site,"EVENT","INVENTORY")
    CASES[site.upper()+"_EVENT_REFERENCE_GOVERNOR"]=(site,"EVENT","GOVERNOR")
CASES["JPL_MINUTE_INVENTORY"]=("jpl","MINUTE","INVENTORY")
CASES["JPL_EVENT_ORIGINAL"]=("jpl","EVENT","ORIGINAL")


# EN: Build and solve the 3-scenario rolling-horizon MILP via single-thread HiGHS and return the non-anticipative step-0 action.
# 中文：构建并由单线程 HiGHS 求解三场景滚动时域 MILP，返回满足首步非预期性耦合的名义充放电指令。
def solve_hour(point, soc):
    values=np.asarray(point,float)
    forecast=model.HorizonData(ev_lower_kw=values,ev_point_kw=values,ev_upper_kw=values,
        pv_lower_kw=np.zeros(24),pv_point_kw=np.zeros(24),pv_upper_kw=np.zeros(24),
        price_yuan_per_kwh=np.full(24,.6),pressure=np.ones(24))
    cfg=ExperimentConfig(battery=BATTERY,horizon_steps=24,step_hours=1.,feeder_safe_kw=200.,pv_rated_kw=300.,reserve_hours=1.)
    actual=model.milp
    calls=0
    def counted(**kwargs):
        nonlocal calls
        calls+=1
        return actual(**kwargs)
    model.milp=counted
    try:
        result=model.solve_first_step(Method.GS_F0,forecast,soc,cfg)
        if result.solver_status!=0 or result.mip_gap is None or result.mip_gap>cfg.solver.mip_relative_gap+1e-12:
            raise ValueError("Unaccepted solver status or MIP gap")
        return [result.charge_kw,result.discharge_kw],calls,None
    except Exception as exc:
        return [0.,0.],calls,f"{type(exc).__name__}: {exc}"
    finally:
        model.milp=actual


def action(policy,t,soc,load,nominal,reserve):
    request=dict(t=t,soc=soc,load_kw=load,nominal=nominal,reserve=reserve)
    if policy=="GOVERNOR":
        return govern(**request)
    if policy=="INVENTORY":
        return inventory(**request)
    if policy=="ORIGINAL":
        return choose_event_target(arm="REF_RECOVERY",**request)
    return choose_minute(arm=policy,**dict(request,t=int(t)))


def run(case,events,hours,output_trace):
    started=time.monotonic()
    site,timing,policy=CASES[case]
    ledger=ServiceLedger(site)
    totals=defaultdict(float)
    failures=[]
    errors=Counter()
    soc=.5
    peak=0.
    triggers=opportunities=calls=0
    shortage=0.
    reserve=0.
    nominal=[0.,0.]
    def continue_at(t):
        reserve_seconds=70. if t%3600==0 else 10.
        return time.monotonic()-started+reserve_seconds<7200.
    for t, load, point, plant in service_intervals(events,ledger,timing=="EVENT",hours*3600,continue_at):
        if point is not None:
            opportunities+=1
            nominal,count,failure=solve_hour(point,soc)
            calls+=count
            if failure:
                failures.append(dict(hour=int(t)//3600, failure=failure,actual_calls=count))
            reserve=max(point[0]-200.,0.)
        selection=action(policy,t,soc,load,nominal,reserve)
        c,d=selection["charge_kw"],selection["discharge_kw"]
        if policy in ("F0","REF","REF_RECOVERY"):
            fixed=dict(plant,start_s=int(F(plant["start_s"])),end_s=int(F(plant["end_s"])))
            executed=integrate_minute(soc,c,d,fixed)
        else:
            executed=integrate_interval(soc,c,d,plant)
        for name,value in executed["totals"].items():
            totals[name]+=value
        for seg in executed["segments"]:
            duration=float((F(seg["end_s_exact"])-F(seg["start_s_exact"]))/3600)
            low=reserve-.95*(seg["energy_pre_kwh"]-60.)
            high=reserve-.95*(seg["energy_post_kwh"]-60.)
            if low>=0. and high>=0.:shortage+=(low+high)*duration/2
            elif low>0. or high>0.:shortage+=duration*max(low,high)**2/(2*abs(high-low))
        peak=max(peak,executed["maximum_grid_kw"])
        soc=executed["SOC_post"]
        triggers+=1
        errors.update(selection["errors"])
        if output_trace:
            output_trace.writerow([str(t),plant["end_s"],load,nominal[0],nominal[1],reserve,c,d,
                                   executed["SOC_pre"],soc,executed["maximum_grid_kw"],executed["totals"]["PCC_excess_kwh"]])
    energy=soc*600.
    cohort=list(ledger.final_rows())
    declared=sum(F(row["requested"])>0 for row in cohort)
    service=dict(month_CONNECT=len(cohort), declared_sessions=declared, undeclared_sessions=len(cohort)-declared,
        deadline_on_time_completed=sum(row["on_time_completed"] for row in cohort),
        unplug_incomplete_sessions=sum(row["unplugged_unmet"] for row in cohort),
        right_censored_sessions=sum(row["right_censored"] for row in cohort),
        requested_kwh=float(sum((F(row["requested"]) for row in cohort),F(0))),
        EV_unfinished_kwh=float(sum((F(row["remaining"]) for row in cohort),F(0))))
    status=dict(case=case,source_site=site,source_event_clock=timing,policy=policy,requested_hours=hours,
        executed_physical_seconds_exact=str(ledger.now),requested_prefix_complete=ledger.now==hours*3600,
        complete_29_days=ledger.now==END,unexecuted_full_month_hours=696-opportunities,
        triggers=triggers,actual_MPC=calls,hour_opportunities=opportunities,
        accepted_MPC=opportunities-len(failures),failed_hour_opportunities=len(failures),
        failed_hours=failures,command_refusals=dict(errors),source_month_events=sum(len(x) for x in events.values()),
        observed_service_counts=dict(ledger.counts),final_energy_kwh=energy,terminal_error_kwh=energy-300.,
        maximum_grid_kw=peak,reserve_shortage_kwh_h=shortage,source_EV_kwh=totals["demand_kwh"],
        inventory_cost_v057=totals["common_cost_yuan"]-.57*(energy-300.),
        inventory_cost_v060=totals["common_cost_yuan"]-.60*(energy-300.),**totals,**service)
    return status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",required=True,choices=sorted(CASES))
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw-pages",type=Path)
    inputs.add_argument("--recorded-events",type=Path)
    parser.add_argument("--hours",type=int,default=696,choices=range(1,697),metavar="1..696")
    parser.add_argument("--report",type=Path)
    parser.add_argument("--trace",type=Path)
    args=parser.parse_args()
    if any(p is not None and p.exists() for p in [args.report,args.trace]):parser.error("choose new output paths")
    site=CASES[args.case][0]
    events=from_raw_pages(args.raw_pages,site) if args.raw_pages else from_recorded_events(args.recorded_events,site)
    if not events:raise ValueError("No source events for this site and month")
    trace=None
    try:
        if args.trace:
            trace=args.trace.open("x",encoding="utf-8",newline="")
            writer=csv.writer(trace)
            writer.writerow(["start_s","end_s","load_kw","nominal_charge_kw","nominal_discharge_kw","reserve_kwh",
                "issued_charge_kw","issued_discharge_kw","soc_pre","soc_post","peak_grid_kw","pcc_excess_kwh"])
        else:writer=None
        result=run(args.case,events,args.hours,writer)
    finally:
        if trace:trace.close()
    if args.report:
        with args.report.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ["case","complete_29_days","triggers","actual_MPC","source_EV_kwh",
        "PCC_excess_kwh","final_energy_kwh","failed_hour_opportunities"]},allow_nan=False))
    return 0 if result["requested_prefix_complete"] else 75


if __name__=="__main__":
    sys.exit(main())
