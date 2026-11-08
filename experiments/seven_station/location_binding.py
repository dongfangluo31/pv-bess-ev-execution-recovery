# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

import json
import os
from pathlib import Path

def station_bus_map():
    case=os.environ['PV_NETWORK_CASE']
    if case not in ('NEAR','MIDDLE','FAR'):raise ValueError('Unknown location')
    return json.loads((Path(__file__).parent/'inputs'/f'{case}.json').read_text())['bus_map']
