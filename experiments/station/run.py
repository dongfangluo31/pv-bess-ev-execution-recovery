import argparse
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run(args):
    data=args.data.resolve()
    output=args.output.resolve()
    os.environ.update(PV_STATION_DATA=str(data),PV_STATION_OUTPUT=str(output),
                      OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                      PYTHONHASHSEED='20260706',PYTHONDONTWRITEBYTECODE='1')
    from feeder_safe.matrix import expand_experiment_matrix
    from feeder_safe.data_pipeline import build_split_dataset
    from feeder_safe.runner import config_from_run_spec,simulate_arm,simulate_paired_arms
    stage='G3_GS_PRESSURE_GRID' if args.condition!='BASE' else 'G2_LONG_STREAM' if args.split=='LONG_194D' else 'G2_CORE_SEASONAL'
    combined=[]
    for filename in ('station_matrix.json','experiment_matrix.json'):
        matrix=HERE/'configs'/filename
        if not any(g['stage']==stage for g in json.loads(matrix.read_text())['grids']):continue
        combined.extend(s for s in expand_experiment_matrix(matrix_path=matrix,stages=[stage])
                        if s.split_id==args.split and s.condition_id==args.condition)
    candidates={s.arm_id:s for s in combined}
    if not candidates:raise ValueError('The requested split and condition are not in the frozen matrix')
    arms=('GS_F0','GS_F1') if args.arm=='paired' else (args.arm,)
    for arm in arms:
        if arm not in candidates:raise ValueError(f'{arm} has no frozen configuration here')
    first=candidates[arms[0]]
    prepared=build_split_dataset(args.split,ev_multiplier=first.ev_multiplier,pv_multiplier=first.pv_multiplier,pv_rated_kw=300.,include_embargo=True)
    configs=[]
    for arm in arms:
        s=candidates[arm]
        configs.append(config_from_run_spec(s,prepared,run_id=('replay_'+s.config_sha256[:20]),run_root=output/'runs',max_steps=args.steps))
    res=simulate_paired_arms(*configs) if len(configs)==2 else simulate_arm(configs[0])
    print(json.dumps(res.to_dict(),ensure_ascii=False,default=str,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--split',choices=('WINTER_30D','SPRING_30D','SUMMER_CLEAN_30D','LONG_194D'),default='SPRING_30D')
    p.add_argument('--condition',default='BASE');p.add_argument('--arm',default='paired',choices=('M6','GS_F0','GS_F1','GS_F1_ACI','paired'))
    p.add_argument('--steps',type=int)
    a=p.parse_args();run(a)
