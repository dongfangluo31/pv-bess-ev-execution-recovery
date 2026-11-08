import argparse
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=('SCALAR_SHARED','NETWORK_JOINT','JOINT_H24_NETWORK_MPC')

def run(arms,out):
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(PV_NETWORK_OUTPUT=str(out),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        NUMBA_CACHE_DIR=str(out/'cache'),MPLCONFIGDIR=str(out/'cache'))
    import pandapower
    from runner import arm_run
    from records import read,write,record
    data=read(HERE/'inputs'/'case.json')
    file=Path(pandapower.__file__).resolve().parent/'networks'/'power_system_test_case_jsons'/'case33bw.json'
    asset=record(file)
    if asset['sha256']!=data['network_asset']['sha256']:raise ValueError('Different IEEE 33-bus asset')
    data['network_asset']=asset
    case=write(out/'case.json',data)
    bg=record(HERE/'inputs'/'background.json')
    (out/'execution').mkdir()
    for plan in read(HERE/'inputs'/'plans.json'):
        if plan['arm'] not in arms:continue
        plan['source_case']=case;plan['background_shape_source_ref']=bg
        identity=write(out/f"{plan['arm']}_plan.json",plan)
        result=arm_run(out,plan,identity,None)
        metrics=result['aggregate_actual']
        print(json.dumps({'arm':result['arm'],'model_seconds':result['physical_seconds'],
            'minimum_voltage_pu':metrics['minimum_voltage_pu'],'voltage_lower_deficit_pu_h':metrics['voltage_lower_deficit_pu_h'],
            'terminal_inventory_difference_kwh':result['terminal_stored_energy_delta_kwh'],
            'inventory_adjusted_cost_yuan':result['terminal_value_corrected_cost_yuan'],
            'command_failures':len(result['control_failures']),'fatal':result['fatal']},indent=2))
        if result['physical_seconds']!=10800 or result['fatal']:raise RuntimeError('Incomplete physical run')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=(*ARMS,'all'),default='all')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(ARMS if args.arm=='all' else (args.arm,),args.output)
