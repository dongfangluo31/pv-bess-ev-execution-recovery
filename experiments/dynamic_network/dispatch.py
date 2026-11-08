# EN: Closed-loop BESS dispatch policy and real-time constraint enforcement.
# 中文：闭环储能调度策略与实时物理约束执行模块。

from copy import deepcopy
import math

class Dispatch:

    def __init__(self, initial):
        self.held = deepcopy(initial)
        self.held_id = 'FIXED_SOURCE_STEP401_ACTUAL'
        self.valid_until = 60
        self.queue = {}
        self.busy_until_ns = 0
        self.events = []

    def action(self, minute):
        t = 60 * minute
        applied = None
        if minute in self.queue:
            q = self.queue.pop(minute)
            if q['effect_s'] != t or q['ready_ns'] > t * 10 ** 9 or q['valid_until_s'] != t + 60:
                raise ValueError('Invalid dispatch queue')
            self.held = deepcopy(q['command'])
            self.held_id = q['command_id']
            self.valid_until = q['valid_until_s']
            applied = q['command_id']
        expired = t >= self.valid_until
        command = {s: dict(charge_kw=0.0, discharge_kw=0.0) for s in self.held} if expired else deepcopy(self.held)
        return (command, dict(held_command_id='COMMON_EXPIRED_ZERO' if expired else self.held_id, valid_until_s=self.valid_until, expired=expired, newly_applied_command_id=applied))

    def opening(self, minute):
        t = minute * 60
        tick = t * 10 ** 9
        available = self.busy_until_ns <= tick
        return dict(planned_decision_s=t, engine_start_model_ns=max(tick, self.busy_until_ns), previous_busy_until_ns=self.busy_until_ns, control_opportunity_available=available)

    def finish(self, minute, opening, total_wall_ns, command, reason):
        if type(total_wall_ns) is not int or total_wall_ns < 0:
            raise ValueError('Integer full work duration')
        t = minute * 60
        ready_ns = opening['engine_start_model_ns'] + total_wall_ns
        ready = ready_ns / 1000000000.0
        self.busy_until_ns = ready_ns
        admitted = bool(opening['control_opportunity_available'] and reason is None and (command is not None) and (ready_ns <= (t + 60) * 10 ** 9))
        q = None
        if admitted:
            q = dict(command_id=f'command:{minute}', decision_s=t, ready_ns=ready_ns, ready_s=ready, effect_s=t + 60, valid_until_s=t + 120, command=deepcopy(command))
            self.queue[minute + 1] = q
        row = dict(minute=minute, ready_ns=ready_ns, ready_s=ready, busy_until_ns=self.busy_until_ns, full_work_wall_ns=total_wall_ns, admitted=admitted, command=q, reason=reason if reason else None if admitted else 'MISSED_FIXED_SLOT_OR_NO_PROPOSAL', discarded_late_proposal=command is not None and ready_ns > (t + 60) * 10 ** 9, **opening)
        self.events.append(row)
        return row
