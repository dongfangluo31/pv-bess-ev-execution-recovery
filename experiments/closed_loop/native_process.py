# EN: Parent supervisor orchestrating asynchronous controller/plant IPC processes, slot deadlines, and fallback holds.
# 中文：异步双进程闭环主控协调器：管理控制器与物理对象进程通信、时隙截止期判定及超时保持回退。

import os
import sys

def own_identity():
    return {'pid':os.getpid(),'image_path':sys.executable}
