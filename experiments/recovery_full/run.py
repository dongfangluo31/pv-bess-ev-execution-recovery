import argparse
import csv
import json
import math
import os
from bisect import bisect_right
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/"recovery"))
from station_core.config import BatteryConfig
from station_core.filter import advance_soc_checked
from station_core.plant import realize_no_export
from recovery_control import safe_target
from reserve_metric import linear_shortage

ARMS=("REFERENCE","SIMPLE","NUMERIC","SIMPLE_PREP_CAUSAL_BRAKE","NUMERIC_PREP_CAUSAL_BRAKE")


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def deliver(method,request):
    started=time.perf_counter()
    payload=json.dumps({"method":method,"request":request},allow_nan=False)
    result=None
    try:
        proc=subprocess.run([sys.executable,"-B",str(HERE/"planner.py")],input=payload,text=True,capture_output=True,
                            timeout=max(0.,60.-(time.perf_counter()-started)),check=True)
        result=json.loads(proc.stdout)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError):
        pass
    accepted=result is not None and result["accepted"]
    if accepted:
        signed=result["target"]
    else:
        packet=request["public_packet"];e=request["entry"]
        c,d,_=safe_target(0.,packet["demand_point_kw"],packet["pv_point_kw"],request["measurement"]["soc"],
            BatteryConfig(**e["battery"]),e["feeder_safe_kw"],request["held_reserve"]["R"])
        signed=d-c
    elapsed=time.perf_counter()-started
    timely=elapsed<=60
    return signed, bool(accepted and timely),timely,elapsed,(result or {}).get("warm",{}),(result or {}).get("policy_state")


def simulate(case,arm,source,trace):
    folder=HERE/"inputs"/case
    entry=json.loads((folder/"entry.json").read_text(encoding="utf8"))
    S,Q,T=(entry[k] for k in ("start_s","handoff_s","end_s"))
    feeds={int(row["decision_s"]):row for row in read_csv(folder/"released.csv")}
    plant=read_csv(folder/"plant.csv")
    starts=[int(row["start_s"]) for row in plant]
    reference=read_csv(folder/"reference_issued.csv") if arm=="REFERENCE" else None
    reference_starts=[int(x["start_s"]) for x in reference] if reference else []
    frozen={int(row["control_decision_s"]):row for row in read_csv(folder/"decisions.csv") if row["arm"]==arm} if source=="recorded" and arm!="REFERENCE" else {}
    q0=entry["known_command_state"]
    active=deepcopy(q0["last_delivered_command"])
    pending=deepcopy(q0["pending_commands"])
    seq=max([c["seq"] for c in pending]+[active["seq"] if active else 0])
    highest=active["seq"] if active else 0
    b=BatteryConfig(**entry["battery"])
    soc=entry["telemetry"]["soc"];power=entry["telemetry"]["tracked_power_kw"]
    alpha=1. if not entry["dynamics"]["lag_seconds"] else 1-math.exp(-1/entry["dynamics"]["lag_seconds"])
    ramp=entry["dynamics"]["ramp_kw_per_second"];G=entry["feeder_safe_kw"]
    dt=1/3600.;warm={};policy_state=None
    reserve=0.;totals=dict(grid_kwh=0.,curtail_kwh=0.,throughput_kwh=0.,PCC_excess_kwh=0.,PCC_seconds=0,
        reserve_exposure_kwh_h=0.,reserve_shortage_seconds=0.,ramp_seconds=0,projection_seconds=0,
        final_zero_violation_seconds=0,fallback_seconds=0,late_opportunities=0,refused_opportunities=0)
    endpoints={}
    for t in range(S,T):
        if (t-S)%60==0:
            feed=feeds[t];reserve=float(feed["reserve_kwh"]);reserve_issue=int(feed["reserve_issue_s"]);available=float(feed["latest_constituent_available_s"])
            if available>t or reserve_issue>t:raise ValueError("Future input release")
            if arm!="REFERENCE":
                if source=="recorded":
                    row=frozen[t]
                    ready=float(row["issue_s"])-t;signed=float(row["signed_kw"])
                    accepted=row["accepted"]=="1";timely=row["scheduled"]=="1"
                    seq=int(row["seq"])
                else:
                    method=arm.split("_")[0] if t<Q or "BRAKE" not in arm else "BRAKE"
                    req=dict(entry=entry,now_s=t,measurement=dict(sample_s=t,available_s=t,soc=soc,tracked_power_kw=power),
                        known_command_state=dict(as_of_s=t,last_delivered_command=active,pending_commands=pending),
                        public_packet=dict(decision_s=t,records=[{"available_s":available}],demand_point_kw=float(feed["load_kw"]),pv_point_kw=float(feed["pv_kw"])),
                        held_reserve=dict(issued_s=reserve_issue,R=reserve),own_previous_accepted_controls=warm,previous_brake_policy=policy_state)
                    signed,accepted,timely,ready,next_warm,next_policy=deliver(method,req)
                    if accepted and method=="NUMERIC":warm=next_warm
                    if method=="BRAKE":policy_state=next_policy if accepted else None
                    seq+=1
                if timely:
                    pending.append(dict(seq=seq,control_decision_s=t,issue_s=t+ready,effective_s=t+60,valid_until_s=t+120,signed_kw=signed))
                totals["late_opportunities"]+=int(not timely)
                totals["refused_opportunities"]+=int(not accepted)
        if reference is not None:
            row=reference[bisect_right(reference_starts,t)-1]
            if not int(row["start_s"])<=t<int(row["end_s"]):raise ValueError("Reference command interval gap")
            desired=float(row["signed_kw"]);fallback=row["fallback"]=="1"
        else:
            for cmd in sorted(list(pending),key=lambda c:(c["effective_s"],c["seq"])):
                if cmd["effective_s"]<=t:
                    pending.remove(cmd)
                    if cmd["valid_until_s"]>=t and cmd["seq"]>highest:
                        active=cmd;highest=cmd["seq"]
            fallback=active is None or active["valid_until_s"]<t
            desired=0. if fallback else active["signed_kw"]
        prev=power;before=soc;ramped=min(max(desired,prev-ramp),prev+ramp);lagged=prev+alpha*(ramped-prev)
        c=min(-lagged,b.charge_limit_kw,max(0.,(b.soc_max-soc)*b.energy_kwh/(b.eta_charge*dt))) if lagged<0 else 0.
        d=min(lagged,b.discharge_limit_kw,max(0.,b.eta_discharge*b.energy_kwh*(soc-b.soc_min)/dt)) if lagged>=0 else 0.
        segment=plant[bisect_right(starts,t)-1]
        if not int(segment["start_s"])<=t<int(segment["end_s"]):raise ValueError("Plant interval gap")
        outcome=realize_no_export(c,d,float(segment["ev_kw"]),float(segment["pv_kw"]))
        power=outcome.discharge_kw-outcome.charge_kw
        soc=advance_soc_checked(soc,outcome.charge_kw,outcome.discharge_kw,b,dt)
        E=soc*b.energy_kwh
        deficit0=reserve-b.eta_discharge*b.energy_kwh*(before-b.soc_min)
        deficit1=reserve-b.eta_discharge*(E-b.soc_min*b.energy_kwh)
        mean,fraction=linear_shortage(deficit0,deficit1)
        totals["grid_kwh"]+=outcome.grid_import_kw*dt;totals["curtail_kwh"]+=outcome.pv_curtail_kw*dt
        totals["throughput_kwh"]+=(outcome.charge_kw+outcome.discharge_kw)*dt
        totals["PCC_excess_kwh"]+=max(outcome.grid_import_kw-G,0.)*dt;totals["PCC_seconds"]+=int(outcome.grid_import_kw>G+1e-9)
        totals["reserve_exposure_kwh_h"]+=mean*dt;totals["reserve_shortage_seconds"]+=fraction
        totals["ramp_seconds"]+=int(abs(power-prev)>ramp+1e-9);totals["projection_seconds"]+=int(abs(power-lagged)>1e-9)
        totals["fallback_seconds"]+=int(fallback)
        if t>=T-60:totals["final_zero_violation_seconds"]+=int(abs(E-entry["final_target_kwh"])>1e-8 or abs(power)>1e-9 or abs(desired)>1e-9)
        if t+1 in (Q,Q+60,T-60,T):endpoints[t+1]={"energy_kwh":E,"power_kw":power}
        if trace:trace.writerow([t,float(segment["ev_kw"]),float(segment["pv_kw"]),desired,power,E,outcome.grid_import_kw,reserve,int(fallback)])
    net=E-entry["initial_energy_kwh"]
    op=.6*totals["grid_kwh"]+.08*totals["throughput_kwh"]
    return dict(case=case,arm=arm,command_source=source,seconds=T-S,final_E_kwh=E,final_P_W=1000*power,
        mid_q60_error_Wh=1000*(endpoints[Q+60]["energy_kwh"]-entry["mid_target_kwh"]),final_E_error_Wh=1000*(E-entry["final_target_kwh"]),
        common_cost_yuan=op,inventory_v057_yuan=op-.57*net,inventory_v060_yuan=op-.60*net,endpoints=endpoints,**totals)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=("MAY18","OPSD"),required=True)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--commands",choices=("recorded","plan"),required=True)
    parser.add_argument("--trace",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.commands=="plan" and args.arm=="REFERENCE":
        parser.error("REFERENCE is the existing accepted reference trajectory; choose recorded")
    if any(p is not None and p.exists() for p in (args.trace,args.output)):
        parser.error("Choose new output paths")
    handle=args.trace.open("x",encoding="utf8",newline="") if args.trace else None
    try:
        writer=csv.writer(handle) if handle else None
        if writer:writer.writerow(["time_s","load_kw","pv_kw","effective_kw","actual_kw","energy_kwh","grid_kw","reserve_kwh","fallback"])
        result=simulate(args.case,args.arm,args.commands,writer)
    finally:
        if handle:handle.close()
    if args.output:
        with args.output.open("x",encoding="utf8") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    main()
