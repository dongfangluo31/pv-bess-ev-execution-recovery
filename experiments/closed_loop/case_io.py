# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

from pathlib import Path
import gzip, json
from protocol import record

def read_rows(reference):
    if record(reference['path']) != reference:
        raise ValueError('Frozen input identity changed')
    with gzip.open(reference['path'], 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_openings(reference, duration_s):
    rows = read_rows(reference)
    if [r['decision_s'] for r in rows] != list(range(0, duration_s, 60)):
        raise ValueError('Full control timeline required')
    return rows

def load_profile(reference, dataset, duration_s):
    rows = read_rows(reference)
    cursor = 0
    for r in rows:
        if r['start_s'] != cursor or r['end_s'] <= cursor:
            raise ValueError('Plant profile gap or order')
        cursor = r['end_s']
    if cursor != duration_s:
        raise ValueError('Complete frozen duration required')
    return rows

def step_source_quality(segment, t, dataset):
    if dataset == 'opsd':
        from source_opsd67_io import step_source_quality as quality
    elif dataset == 'sofie':
        from source_sofie_io import step_source_quality as quality
    else:
        raise ValueError('Dataset')
    return quality(segment, t)
