# EN: Deterministic SciPy SLSQP trajectory optimizer solving constrained terminal SOC recovery.
# 中文：求解带物理路径约束终端储能状态恢复问题的确定性 SciPy SLSQP 数值优化器。

import time
import numpy as np
from full_model import evaluate, initial_guess

def solve_once(problem, warm, journal):
    start = time.perf_counter()
    counts = dict(optimizer_calls=0, model_calls_started=0, model_calls_completed=0, modeled_steps_requested=0, modeled_steps_completed=0, iterations=0)
    history = []
    cache_x = None
    cache = None
    result = None
    error = None

    def model(x):
        nonlocal cache_x, cache
        if time.perf_counter() - start > 60:
            raise TimeoutError('Numeric child complete budget exhausted')
        if cache_x is None or not np.array_equal(cache_x, x):
            counts['model_calls_started'] += 1
            counts['modeled_steps_requested'] += problem['seconds']
            journal('MODEL_STARTED', counts=dict(counts))
            cache = evaluate(problem, x)
            cache_x = np.array(x, copy=True)
            counts['model_calls_completed'] += 1
            counts['modeled_steps_completed'] += problem['seconds']
            journal('MODEL_RETURNED', counts=dict(counts))
        return cache

    def obj(x):
        return model(x)['objective']

    def oj(x):
        return model(x)['objective_jac']

    def eq(x):
        return model(x)['equality']

    def ej(x):
        return model(x)['equality_jac']

    def ge(x):
        return model(x)['ineq']

    def gj(x):
        return model(x)['ineq_jac']

    def callback(x):
        v = model(x)
        counts['iterations'] += 1
        row = dict(iteration=counts['iterations'], elapsed=time.perf_counter() - start, objective=v['objective'], max_equality=float(np.max(np.abs(v['equality']))), minimum_margin=float(np.min(v['ineq'])), nominal_gate=v['gates']['accepted'])
        history.append(row)
        journal('ITERATION', **row)
    seed = initial_guess(problem, warm)
    d = problem['entry']['battery']
    if not len(seed):
        return (dict(status='NO_FREE_CONTROLS_FIXED_ZERO_TAIL', accepted=False, normal_return=False, controls=None, counts=counts, error=None), None)
    try:
        from scipy.optimize import minimize
        counts['optimizer_calls'] = 1
        journal('OPTIMIZER_STARTED', method='SLSQP', maxiter=20, ftol=1e-12)
        result = minimize(obj, seed, method='SLSQP', jac=oj, bounds=[(-d['charge_limit_kw'], d['discharge_limit_kw'])] * len(seed), constraints=[dict(type='eq', fun=eq, jac=ej), dict(type='ineq', fun=ge, jac=gj)], callback=callback, options=dict(maxiter=20, ftol=1e-12, disp=False))
        journal('OPTIMIZER_RETURNED', status=int(result.status), success=bool(result.success))
        x = np.asarray(result.x, dtype=float)
        if x.shape != seed.shape or not np.isfinite(x).all():
            raise ValueError('Nonfinite normal returned controls')
        v = model(x)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        v = cache
    elapsed = time.perf_counter() - start
    normal = result is not None and error is None
    accepted = bool(normal and v['gates']['accepted'] and (elapsed <= 60))
    output = dict(status='ACCEPTED_FEASIBLE_CANDIDATE' if accepted and result.success else 'ACCEPTED_FEASIBLE_UNCONVERGED_CANDIDATE' if accepted else 'REFUSED_RETAINED', accepted=accepted, normal_return=normal, controls=np.asarray(result.x).tolist() if normal else None, cached_controls_diagnostic=None if normal or cache_x is None else cache_x.tolist(), counts=counts, history=history, error=error, elapsed_seconds=elapsed, initial_controls=seed.tolist(), warm_start_from_own_timely_accepted_only=bool(warm), success=None if result is None else bool(result.success), solver_status=None if result is None else int(result.status), solver_message=None if result is None else str(result.message), gates=None if v is None else v['gates'], objective=None if v is None else v['objective'], optimality_claim=False, global_infeasibility_claim=False)
    return (output, v if accepted else None)
