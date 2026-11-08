# EN: Module search-path setup for the 3-hour second-by-second IEEE 33-bus dynamic network simulation.
# 中文：IEEE 33 节点配电网 3 小时逐秒动态闭环仿真的模块搜索路径配置。

from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ['PV_NETWORK_OUTPUT']).resolve()

def wire():
    for p in (ROOT / 'core', ROOT / 'network', ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return ROOT
