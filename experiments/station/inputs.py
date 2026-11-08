# EN: Deterministic input preparation and schema/hash verification script.
# 中文：确定性模型输入构造与数据结构/哈希校验脚本。

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
from zipfile import ZipFile

from prepare_load import aggregate

FIGSHARE='https://api.figshare.com/v2/articles/28182251/versions/3'
PVOD='https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62'

def unique_member(archive,name):
    hits=[s for s in archive.namelist() if Path(s.replace('\\','/')).name==name and '__MACOSX' not in s]
    if len(hits)!=1:raise ValueError(f'Expected one {name} in the supplied archive')
    return hits[0]

def extract_member(archive,name,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with archive.open(unique_member(archive,name)) as source,path.open('xb') as dest:
        shutil.copyfileobj(source,dest)

def prepare(out,pvod,china=None):
    out=out.resolve()
    if not pvod.is_file():raise ValueError('Supply the PVOD v1.0 archive obtained from '+PVOD)
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(dir=out) as folder:
        if china is None:
            with urllib.request.urlopen(FIGSHARE,timeout=45) as response:record=json.load(response)
            item=next(f for f in record['files'] if f['name']=='Dataset.zip')
            china=Path(folder)/'Dataset.zip'
            with urllib.request.urlopen(item['download_url'],timeout=120) as source,china.open('xb') as dest:
                shutil.copyfileobj(source,dest)
            if china.stat().st_size!=item['size'] or hashlib.md5(china.read_bytes()).hexdigest()!=item['computed_md5']:
                raise ValueError('Download differs from the registered Figshare file')
        with ZipFile(china) as z:
            with z.open(unique_member(z,'Charging_Data.csv')) as source:
                aggregate(source,out/'station_hourly_ev_load.csv')
            extract_member(z,'Time-of-use_Price.csv',out/'28182251'/'Dataset'/'Time-of-use_Price.csv')
        with ZipFile(pvod) as z:
            for filename in ('metadata.csv','station01.csv'):
                extract_member(z,filename,out/'PVOD'/filename)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pvod-archive',type=Path,required=True)
    p.add_argument('--china-archive',type=Path)
    a=p.parse_args();prepare(a.output,a.pvod_archive,a.china_archive)
