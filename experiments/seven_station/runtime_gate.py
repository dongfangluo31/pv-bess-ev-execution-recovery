import json,os,time
ARM_STARTED=None
ROUND_STARTED=None
ARM_COUNTS={}
J7=0
def arm_begin(arm, started):
    global ARM_STARTED, ROUND_STARTED, ARM_COUNTS, J7
    ARM_STARTED = started
    ROUND_STARTED = None
    ARM_COUNTS = {'MPC': 0, 'LP': 0, 'AC': 0}
    J7 = 0

def round_begin(minute, started):
    global ROUND_STARTED
    if ARM_STARTED is None:
        raise RuntimeError('Arm gate missing')
    if time.perf_counter() - ARM_STARTED >= 600:
        raise TimeoutError('Original600s arm work budget reached')
    ROUND_STARTED = started

def allow_call(kind):
    if ARM_STARTED is None or ROUND_STARTED is None:
        raise RuntimeError('Original arm/round context absent')
    if time.perf_counter() - ARM_STARTED >= 600:
        raise TimeoutError('Original600s arm budget reached before backend')
    if time.perf_counter() - ROUND_STARTED >= 60:
        raise TimeoutError('Original60s round deadline reached before backend')
    per_arm = {'MPC': 21, 'LP': 180, 'AC': 180}
    if ARM_COUNTS[kind] >= per_arm[kind]:
        raise RuntimeError('Per-arm call cap reached')
    ARM_COUNTS[kind] += 1

def given_j7_before(attempts, arm, minute, request, source_ac, run_id):
    global J7
    if ARM_STARTED is None or ROUND_STARTED is None:
        raise RuntimeError('J7 context missing')
    if time.perf_counter() - ARM_STARTED >= 600 or time.perf_counter() - ROUND_STARTED >= 60:
        raise TimeoutError('Original time budget reached before J7')
    if J7 >= 180 or arm != 'NETWORK_JOINT':
        raise RuntimeError('J7 hard budget/arm rejected')
    J7 += 1
    event = {'event': 'BEFORE_GIVEN_LINEAR_SYSTEM', 'kind': 'ALGEBRA_NOT_OPTIMIZER', 'run_id': run_id, 'arm': arm, 'minute': minute, 'linear_id': J7, 'source_ac': source_ac, 'request': request}
    attempts.write(json.dumps(event, separators=(',', ':')) + '\n')
    attempts.flush()
    os.fsync(attempts.fileno())
    return J7
