import argparse
import json
import os
from pathlib import Path
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

from actual_tail import execute, map_at_entry
from common_nominal_check import check_minute_controls
from equal_bridge import synthesize_equal
from station_core.config import BatteryConfig
from numeric_solver import solve_once
from public_adapter import from_public_bytes

HERE=Path(__file__).resolve().parent
DATA=HERE/'inputs'
ENTRY_SHA='76c6a602220438b2628184e0008efb5bfe1a93197e65f0e7150254f2db06c36d'

def analytic(entry,out):
    clock=entry['clock']
    cfg=dict(run_id=entry['origin_run_id'],arm='ORIGINAL_EQUAL_BRIDGE',duration_s=clock['episode_end_s'],
             feeder_safe_kw=entry['feeder_safe_kw'],positive_inventory_challenge={'offset_sign':-1},
             recovery_spec={'target_stored_energy_kwh':entry['terminal_service']['target_energy_kwh'],
                            'ramp_kw_per_second':entry['dynamics']['ramp_kw_per_second'],'lag_seconds':entry['dynamics']['lag_seconds']})
    point=entry['nominal_current_point']
    plan,_=synthesize_equal(cfg,BatteryConfig(**entry['battery']),entry['public_packet'],entry['telemetry'],
        entry['known_command_state'],clock['issue_s'],entry['released_empirical_pointwise_forecast'],
        point['load_kw'],point['pv_kw'],entry['held_nominal_reserve']['value_kwh'],out)
    if plan['witness']['status']!='reachable':
        return None,{'status':plan['witness']['status']}
    schedule={row['issue_step']:row['target_kw'] for row in plan['calendar']}
    return [schedule[clock['issue_s']+60*j] for j in range(58)],{'status':'reachable'}

def run(branch,output):
    output.mkdir(parents=True,exist_ok=False)
    blob=(DATA/'entry.json').read_bytes()
    entry=json.loads(blob)
    clock_begin=time.perf_counter()
    problem,_=from_public_bytes(blob,expected_sha256=ENTRY_SHA)
    if branch=='analytic':
        controls,solver=analytic(entry,output)
        frozen_branch='ORIGINAL_EQUAL_BRIDGE'
    else:
        design_sha='3da64197fd275789d58c858dae0d6d3a6c94a1956ae08b4f526718332cc59061'
        solver=solve_once(problem,frozen_root_design_sha256=design_sha)
        controls=solver['plan']['minute_controls_kw'] if solver['accepted'] else None
        frozen_branch='INDEPENDENT_NUMERICAL_TERMINAL'
    gate=check_minute_controls(blob,expected_snapshot_sha256=ENTRY_SHA,minute_controls_kw=controls) if controls is not None else None
    if gate is None or not gate['accepted']:controls=None
    initial=map_at_entry(entry,controls[0]) if controls is not None else None
    delivery_s=time.perf_counter()-clock_begin
    if delivery_s>60:controls=None;initial=None
    ready={'ready_wall_seconds':delivery_s,'deadline_missed':delivery_s>60}
    interface={'role2_later_public_feeds':{'packets':{'path':str(DATA/'packets.jsonl.gz')},
                    'reserve_timeline':{'path':str(DATA/'reserves.json')}},
               'role3_hidden_plant':{'file':{'path':str(DATA/'plant.jsonl.gz')}}}
    actual=execute(entry,interface,output,controls,ready,initial,frozen_branch)
    result={'branch':branch,'seconds':actual['actual_suffix_steps'],'plan_accepted':controls is not None,'in_process_plan_seconds':delivery_s,
            'solver_status':solver.get('scipy_status',solver['status']),'optimizer_calls':solver.get('counts',{}).get('optimizer_calls',0),
            'actual_common_cost_yuan':actual['totals']['common_cost_yuan'],
            'terminal_error_kwh':actual['final_energy_error_kwh'],'pcc_excess_kwh':actual['totals']['PCC_excess_kwh'],
            'terminal_service_pass':actual['terminal_energy_and_power_ok'] and actual['bridge_endpoint_ok'] and actual['final_zero_hold_ok']}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--branch',required=True,choices=('analytic','numeric'))
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    run(args.branch,args.output)
