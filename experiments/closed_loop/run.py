import argparse
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Thread
import time
import uuid

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONHASHSEED']='20260706'

from protocol import load,record,write
from relay import episode

HERE=Path(__file__).resolve().parent
CASES=('PVAWARE_SOFIE875H','PVAWARE_OPSD24H','PVAWARE_OPSD67H')

# EN: Subprocess JSON-RPC client managing companion handshake (`HELLO`), request nonces, and clean shutdown.
# 中文：子进程 JSON-RPC 通信客户端，负责握手校验（`HELLO`）、请求随机数匹配与受控退出。
class Peer:
    def __init__(self,role,cfg,log):
        self.cfg=cfg
        self.role=role
        self.out=Path(cfg['output_dir'])
        self.out.mkdir()
        config=self.out/'config.json'
        write(config,cfg)
        self.stderr=(self.out/'stderr.txt').open('x',encoding='utf-8')
        self.p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/f'{role}_process.py'),str(config)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',bufsize=1)
        self.messages=Queue()
        self.reader=Thread(target=self._read,daemon=True);self.reader.start()
        self.counter=0
        try:
            hello=self._next(90)
            if hello.get('type')!='HELLO' or hello.get('nonce')!=cfg['nonce'] or hello.get('role')!=role:
                raise RuntimeError('Invalid companion handshake')
            self.identity=hello['identity']
        except BaseException:
            self.close(forced=True)
            raise

    def _read(self):
        for line in self.p.stdout:self.messages.put(line)
        self.messages.put(None)

    def _next(self,limit):
        try:line=self.messages.get(timeout=limit)
        except Empty as exc:raise TimeoutError(self.role+' response timed out') from exc
        if line is None:raise RuntimeError(self.role+' exited before responding')
        return json.loads(line)

    def rpc(self,msg,limit=30.):
        begin=time.perf_counter();self.counter+=1
        request={**msg,'request_id':self.counter,'nonce':self.cfg['nonce']}
        self.p.stdin.write(json.dumps(request,separators=(',',':'),allow_nan=False)+'\n');self.p.stdin.flush()
        answer=self._next(limit)
        if answer.get('request_id')!=self.counter or answer.get('nonce')!=self.cfg['nonce']:
            raise RuntimeError('Protocol identity does not match the pending request')
        return answer,time.perf_counter()-begin

    def close(self,forced=False):
        if self.stderr.closed:return
        try:
            if self.p.poll() is None and not forced:self.rpc({'type':'STOP'},10)
        finally:
            if self.p.poll() is None:
                if forced:self.p.terminate()
                try:self.p.wait(timeout=8)
                except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
            self.p.stdin.close();self.reader.join(timeout=2);self.p.stdout.close();self.stderr.close()

def run(case,output,arms):
    data=HERE/'inputs'/case
    cfg=load(data/'CONTROLLER_BASE.json')
    plant=load(data/'PLANT_BASE.json')
    duration=cfg['duration_s'];batch=output.resolve()
    batch.mkdir(parents=True,exist_ok=False)
    execution=batch/'execution';execution.mkdir()
    plant['plant_profile']=record(data/'plant_profile.jsonl.gz')
    write(batch/'CONTROLLER_BASE.json',cfg);write(batch/'PLANT_BASE.json',plant)
    write(batch/'NATIVE_RELAY_SOURCE.json',{'openings':record(data/'controller_packets.jsonl.gz')})
    write(batch/'batch_plan.json',{'duration_s':duration,'per_arm_MPC_cap':duration//900,'arm_wall_seconds':10800})
    ledger=execution/'attempts.jsonl';ledger.touch(exist_ok=False)
    summary={}
    for arm in arms:
        outcome=episode(arm,batch,execution,ledger,Peer)
        events=execution/arm/'plant'/'events.jsonl'
        excess=energy=0.;n=0
        with events.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row.get('event')=='PLANT_STEP' or 'grid_kw' in row and 'soc_post' in row:
                    n+=1;excess+=max(row['grid_kw']-cfg['feeder_safe_kw'],0)/3600
                    energy=row['soc_post']*cfg['battery']['energy_kwh']
        summary[arm]={'status':outcome['status'],'steps':n,'MPC':outcome['actual_MPC_calls'],
            'closed_loop_excess_kwh':excess,'final_energy_kwh':energy,'controller_errors':outcome['errors']}
        if n!=duration or outcome['parent_failure'] or outcome['errors']:raise RuntimeError(str(summary[arm]))
    write(batch/'result.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--arm',choices=('F0_FAST','F1_FAST','both'),default='both')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.case,a.output,('F0_FAST','F1_FAST') if a.arm=='both' else (a.arm,))
