import argparse
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent

def run(case,out):
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(PV_NETWORK_OUTPUT=str(out),PV_NETWORK_CASE=case,PYTHONDONTWRITEBYTECODE='1',
                PYTHONHASHSEED='20260706',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                NUMBA_CACHE_DIR=str(out/'cache'),MPLCONFIGDIR=str(out/'cache'))
    import pandapower
    import dispatch
    from scene_io import MAXIMUM,load,write,record
    from runtime_gate import allow_call
    data=load(HERE/'inputs'/f'{case}.json')
    asset=Path(pandapower.__file__).resolve().parent/'networks'/'power_system_test_case_jsons'/'case33bw.json'
    identity=record(asset)
    if data['network_asset']['sha256']!=identity['sha256']:raise ValueError('The pandapower case33bw model version differs')
    data['network_asset']=identity
    write(out/'case.json',data)
    ref=record(out/'case.json')
    write(out/'INPUT_REFERENCE.json',ref)
    write(out/'PREPARED.json',{'source_files':[record(HERE/'inputs'/f'{case}.json')],'input_files':[ref],'external_dependencies':[identity]})
    rd=out/'execution';rd.mkdir()
    for name in ('mpc','lp','ac','arms','jacobian'):(rd/name).mkdir()
    counts=dict(MPC=0,LP=0,AC=0)
    state=dict(ac=0,lp=0,J7=0,forbidden_optimizer_attempts=0,context={})
    attempts=(rd/'attempts.jsonl').open('x',encoding='utf-8')
    def mark(kind,cid,context):
        allow_call(kind)
        if cid!=counts[kind]+1 or cid>MAXIMUM[kind]:raise RuntimeError('Frozen solver-call limit')
        counts[kind]=cid
        attempts.write(json.dumps({'kind':kind,'call_id':cid,'context':context})+'\n');attempts.flush()
    code=dispatch.execute(rd,counts,attempts,state,time.perf_counter(),mark)
    finish=load(rd/'RUN_COMPLETE.json')
    if code or finish.get('global_failure'):raise RuntimeError(str(finish.get('global_failure')))
    result={}
    for arm in ('SCALAR_SHARED','NETWORK_JOINT'):
        trace=[json.loads(line) for line in (rd/'arms'/f'{arm}_minutes.jsonl').read_text().splitlines()]
        if len(trace)!=180:raise RuntimeError(f'{arm}: {len(trace)}/180 minutes completed')
        metrics={'minimum_voltage_pu':min(r['minimum_voltage_pu'] for r in trace),
                 'grid_only_cost_yuan':sum(r['grid_cost_yuan'] for r in trace),
                 'throughput_kwh':sum(r['throughput_kwh'] for r in trace)}
        result[arm]=metrics
    print(json.dumps({'case':case,'backend_calls':counts,'arms':result},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True,choices=['NEAR','MIDDLE','FAR'])
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.case,a.output)
