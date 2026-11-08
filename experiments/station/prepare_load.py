# EN: Deterministic input preparation and schema/hash verification script.
# 中文：确定性模型输入构造与数据结构/哈希校验脚本。

import argparse
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np

def aggregate(source,output):
    fields=['Location Information','Start Time','End Time','Transaction power/kwh',
            'is_abnormal','Electricity cost/Yuan','Transaction Amount/Yuan']
    frame=pd.read_csv(source,encoding='gbk',usecols=fields)
    frame.columns=frame.columns.str.strip()
    frame['station']=frame['Location Information'].astype(str).str.strip()
    start=pd.to_datetime(frame['Start Time'],errors='coerce')
    end=pd.to_datetime(frame['End Time'],errors='coerce')
    frame['start']=start;frame['end']=end;frame['hours']=(end-start).dt.total_seconds()/3600
    e=frame['Transaction power/kwh']
    usable=(start.notna() & end.notna() & (frame.hours>0) & (frame.hours<=24) & (e>0)
            & (e<=120) & (frame.is_abnormal==0) & np.isfinite(frame['Electricity cost/Yuan']/e)
            & np.isfinite(frame['Transaction Amount/Yuan']/e))
    frame=frame.loc[usable]
    selected=frame.station.value_counts().index[0]
    frame=frame.loc[frame.station==selected].copy()
    numeric=['hours','Transaction power/kwh']
    for _ in range(2):
        frame[numeric]=pd.read_csv(StringIO(frame[numeric].to_csv(index=False))).to_numpy()
    frame['kw']=frame['Transaction power/kwh']/frame.hours
    frame=frame.loc[(frame.hours>=1/60)&(frame.kw>0)&(frame.kw<=250)]
    dates=pd.date_range(frame.start.min().floor('h'),frame.end.max().ceil('h'),freq='h')
    load=pd.Series(0.,index=dates)
    for start,end,kw in frame[['start','end','kw']].itertuples(index=False,name=None):
        current=start.floor('h')
        while current<end:
            right=current+pd.Timedelta(hours=1)
            load.loc[current]+=kw*(min(end,right)-max(start,current)).total_seconds()/3600
            current=right
    result=load.rename_axis('time').rename('ev_energy_kwh').reset_index()
    result['ev_load_kw']=result.ev_energy_kwh
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8-sig',newline='') as f:result.to_csv(f,index=False)
    print(selected,len(frame),len(result),result.ev_energy_kwh.sum())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('transactions',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();aggregate(a.transactions,a.output)
