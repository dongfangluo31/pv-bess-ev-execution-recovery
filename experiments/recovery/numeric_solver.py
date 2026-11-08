# EN: Deterministic SciPy SLSQP trajectory optimizer solving constrained terminal SOC recovery.
# 中文：求解带物理路径约束终端储能状态恢复问题的确定性 SciPy SLSQP 数值优化器。

import time
import numpy as np
from numeric_core import evaluate, initial_guess
METHOD = 'SINGLE_SHOT_SLSQP20_MINUTE_COMMANDS'
MAXITER = 20
FTOL = 1e-12
WALL_SECONDS = 60.0
PRICE = 0.6
THROUGHPUT_PRICE = 0.08

class BudgetExpired(RuntimeError):
    pass

def candidate_acceptance(result, gates, elapsed_seconds, error=None):
    finite = False
    if result is not None:
        try:
            x = np.asarray(result.x, dtype=float)
            finite = bool(x.shape == (58,) and np.isfinite(x).all())
        except (AttributeError, TypeError, ValueError):
            finite = False
    accepted = bool(result is not None and finite and (error is None) and (gates is not None) and gates['accepted'] and (0 <= elapsed_seconds <= WALL_SECONDS))
    status = 'REFUSED_NUMERICAL_ATTEMPT'
    if accepted:
        status = 'ACCEPTED_FEASIBLE_CANDIDATE' if bool(result.success) else 'ACCEPTED_FEASIBLE_UNCONVERGED_CANDIDATE'
    return dict(accepted=accepted, status=status, normally_returned_finite_vector=finite, local_optimality_claim=False, global_optimality_claim=False, global_infeasibility_claim=False)

def solve_once(problem, *, frozen_root_design_sha256):
    if not isinstance(frozen_root_design_sha256, str) or len(frozen_root_design_sha256) != 64 or any((c not in '0123456789abcdef' for c in frozen_root_design_sha256)):
        raise ValueError('Frozen root design identity required; no preparation-stage solve')
    if problem.require_no_projection is not True or problem.free_minutes != 58:
        raise ValueError('Frozen comparison requires 58 free commands and no nominal extra projection')
    started = time.perf_counter()
    counts = dict(optimizer_calls=0, model_evaluations=0, model_steps=0, objective_requests=0, objective_jacobian_requests=0, equality_requests=0, inequality_requests=0, equality_jacobian_requests=0, inequality_jacobian_requests=0, iterations=0, actual_plant_calls=0, future_truth_reads=0)
    cached_x = None
    cached = None
    history = []

    def within_budget():
        if time.perf_counter() - started > WALL_SECONDS:
            raise BudgetExpired('Original 60 second complete preparation budget exceeded')

    def model(x):
        nonlocal cached_x, cached
        within_budget()
        if cached_x is None or not np.array_equal(cached_x, x):
            counts['model_evaluations'] += 1
            counts['model_steps'] += problem.seconds
            cached = evaluate(problem, x)
            cached_x = np.asarray(x, dtype=float).copy()
            within_budget()
        return cached

    def objective(x):
        counts['objective_requests'] += 1
        v = model(x)
        return PRICE * v['grid_energy_kwh'] + THROUGHPUT_PRICE * v['throughput_kwh']

    def objective_jac(x):
        counts['objective_jacobian_requests'] += 1
        v = model(x)
        return PRICE * v['grid_energy_jac'] + THROUGHPUT_PRICE * v['throughput_jac']

    def eq(x):
        counts['equality_requests'] += 1
        return model(x)['equality']

    def eq_jac(x):
        counts['equality_jacobian_requests'] += 1
        return model(x)['equality_jac']

    def ge(x):
        counts['inequality_requests'] += 1
        return model(x)['ineq']

    def ge_jac(x):
        counts['inequality_jacobian_requests'] += 1
        return model(x)['ineq_jac']

    def callback(x):
        counts['iterations'] += 1
        v = model(x)
        history.append(dict(iteration=counts['iterations'], elapsed_seconds=time.perf_counter() - started, terminal_energy_error_kwh=float(v['equality'][0]), bridge_power_kw=float(v['equality'][1]), minimum_margin=float(np.min(v['ineq'])), nominal_cost_yuan=PRICE * v['grid_energy_kwh'] + THROUGHPUT_PRICE * v['throughput_kwh']))
    seed = initial_guess(problem)
    result = None
    error = None
    try:
        from scipy.optimize import minimize
        within_budget()
        counts['optimizer_calls'] = 1
        result = minimize(objective, seed, method='SLSQP', jac=objective_jac, bounds=[(-problem.charge_limit_kw, problem.discharge_limit_kw)] * problem.free_minutes, constraints=[dict(type='eq', fun=eq, jac=eq_jac), dict(type='ineq', fun=ge, jac=ge_jac)], callback=callback, options=dict(maxiter=MAXITER, ftol=FTOL, disp=False))
        returned = np.asarray(result.x, dtype=float)
        if returned.shape != (58,) or not np.isfinite(returned).all():
            raise ValueError('Normally returned candidate is not a finite 58-vector')
        final = model(returned)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        final = cached
    elapsed = time.perf_counter() - started
    acceptance = candidate_acceptance(result, None if final is None else final['gates'], elapsed, error)
    accepted = acceptance['accepted']
    output = dict(method=METHOD, root_design_sha256=frozen_root_design_sha256, **acceptance, nominal_current_point_only=True, receding_horizon=False, objective=dict(grid_price=PRICE, throughput_price=THROUGHPUT_PRICE, added_penalties=0), options=dict(maxiter=MAXITER, ftol=FTOL, total_wall_seconds=WALL_SECONDS), initialization='UNIFORM_ENERGY_GAP_REMAINING_TIME_EFFICIENCY_ONLY', initial_controls_kw=seed.tolist(), counts=counts, iteration_history=history, wall_seconds=elapsed, error=error, scipy_success=None if result is None else bool(result.success), scipy_status=None if result is None else int(result.status), scipy_message=None if result is None else str(result.message), gates=None if final is None else final['gates'])
    if result is not None:
        if acceptance['normally_returned_finite_vector']:
            output['returned_controls_kw'] = np.asarray(result.x, dtype=float).tolist()
        else:
            output['invalid_returned_vector_repr'] = repr(result.x)
        raw_fun = float(result.fun)
        output['returned_objective_yuan'] = raw_fun if np.isfinite(raw_fun) else None
        if not np.isfinite(raw_fun):
            output['invalid_reported_objective_repr'] = repr(raw_fun)
    elif cached_x is not None:
        output['last_evaluated_controls_kw'] = cached_x.tolist()
    if accepted:
        output['plan'] = dict(minute_controls_kw=np.asarray(result.x).tolist(), powers_kw=final['power_kw'].tolist(), energies_kwh=final['energy_kwh'].tolist(), commands_kw=final['commands_kw'].tolist(), fixed_zero_tail_seconds=problem.zero_tail_seconds, immutable_prefix_seconds=len(problem.prefix_commands_kw))
    else:
        output['plan'] = None
    output['wall_seconds'] = time.perf_counter() - started
    if output['wall_seconds'] > WALL_SECONDS:
        output['accepted'] = False
        output['status'] = 'REFUSED_NUMERICAL_ATTEMPT'
        output['plan'] = None
        if output['error'] is None:
            output['error'] = dict(type='BudgetExpired', message='Final output preparation exceeded 60 seconds')
    return output
