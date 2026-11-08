# EN: Real-time control-slot wall-clock deadline and causal command-activation timing rules.
# 中文：实时控制时隙墙钟截止期与指令因果生效时刻契约定义。

import math

def fixed_slot_delivery(command, t, rpc_wall_seconds):
    if not isinstance(t, int) or t < 0 or t % 60:
        raise ValueError('Only fixed minute decisions')
    issue = command['issue_s']
    if not all((math.isfinite(x) for x in (issue, rpc_wall_seconds))) or rpc_wall_seconds < 0 or issue < t:
        raise ValueError('Invalid actual compute timestamps')
    if command['control_decision_s'] != t or command['dispatch_slot_s'] != t + 60 or command['valid_until_s'] != t + 120 or (command['lifetime_reference'] != 'fixed_dispatch_slot'):
        raise ValueError('Controller fixed-slot contract changed')
    c = dict(command)
    c['ready_s'] = max(issue, t + rpc_wall_seconds)
    c['effective_s'] = t + 60
    return (c, c['ready_s'] > c['effective_s'])
