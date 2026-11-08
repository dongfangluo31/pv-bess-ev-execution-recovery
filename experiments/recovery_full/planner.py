# EN: Closed-loop BESS dispatch policy and real-time constraint enforcement.
# 中文：闭环储能调度策略与实时物理约束执行模块。

import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
sys.path.insert(1,str(Path(__file__).resolve().parent.parent/"recovery"))

from station_core.config import BatteryConfig
from recovery_control import safe_target, predict_prefix


def plan(kind, request):
    entry=request["entry"]
    now=request["now_s"]
    packet=request["public_packet"]
    current=request["measurement"]
    battery=BatteryConfig(**entry["battery"])
    load=packet["demand_point_kw"]
    pv=packet["pv_point_kw"]
    reserve=request["held_reserve"]["R"]
    if kind=="BRAKE":
        from policy import propose
        result=propose(request,request.get("previous_brake_policy"))
        return {"accepted":result["accepted"],"target":result["mapping"]["signed_kw"] if result["accepted"] else None,
                "policy_state":result["policy_state"],"warm":{}}
    warm={}
    if now+60>=entry["end_s"]-60:
        raw=0.
    elif kind=="SIMPLE":
        dyn=entry["dynamics"]
        pred=predict_prefix(current,request["known_command_state"],now,battery,load,pv,dyn["ramp_kw_per_second"],dyn["lag_seconds"])
        early=now<entry["handoff_s"]
        until=entry["handoff_s"] if early else entry["end_s"]-60
        target=entry["mid_target_kwh"] if early else entry["final_target_kwh"]
        gap=target-pred["predicted_energy_kwh"]
        hours=max(60,until-(now+60))/3600
        value=-gap/(battery.eta_charge*hours) if gap>=0 else -gap*battery.eta_discharge/hours
        raw=min(battery.discharge_limit_kw,max(-battery.charge_limit_kw,value))
    elif kind=="NUMERIC":
        from full_model import problem_from_request, evaluate
        from numeric_solver import solve_once
        problem=problem_from_request(request)
        outcome,_=solve_once(problem,request.get("own_previous_accepted_controls",{}),lambda *args,**kw:None)
        if not outcome["accepted"] or not evaluate(problem,outcome["controls"])["gates"]["accepted"]:
            return {"accepted":False,"target":None,"warm":{},"policy_state":None}
        controls=outcome["controls"]
        raw=controls[0]
        warm={str(t):value for t,value in zip(problem["free_effects"],controls)}
    else:
        raise ValueError("Unknown planning method")
    c,d,_=safe_target(raw,load,pv,current["soc"],battery,entry["feeder_safe_kw"],reserve)
    return {"accepted":True,"target":d-c,"warm":warm,"policy_state":None}


if __name__=="__main__":
    job=json.load(sys.stdin)
    json.dump(plan(job["method"],job["request"]),sys.stdout,allow_nan=False)
