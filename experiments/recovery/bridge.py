# EN: Analytical ramp-and-lag midpoint bridge synthesizing feasible two-hold terminal SOC recovery commands.
# 中文：合成满足斜率与一阶滞后约束的两段保持终端 SOC 恢复指令的解析中点桥接模块。

from __future__ import annotations
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
VERSION = 'terminal_recovery'

def _number(value, name, *, allow_inf=False):
    if isinstance(value, bool):
        raise ValueError(f'{name}: boolean is not a numeric input')
    value = float(value)
    if math.isnan(value) or (not allow_inf and (not math.isfinite(value))):
        raise ValueError(f'{name}: finite real number required')
    return value

def _series(value, n, name):
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        if len(value) != n:
            raise ValueError(f'{name}: expected {n} entries')
        return [_number(x, name, allow_inf=True) for x in value]
    return [_number(value, name, allow_inf=True)] * n

def _bounds(command_bounds):

    def pair(values, name):
        if not isinstance(values, Sequence) or len(values) != 2:
            raise ValueError(f'{name}: expected (lower, upper)')
        lower, upper = (_number(x, name, allow_inf=True) for x in values)
        if lower > upper or lower == math.inf or upper == -math.inf:
            raise ValueError(f'{name}: invalid interval')
        return (lower, upper)
    if command_bounds is None:
        return {'A': (-math.inf, math.inf), 'B': (-math.inf, math.inf)}
    if isinstance(command_bounds, Mapping):
        if set(command_bounds) != {'A', 'B'}:
            raise ValueError('command_bounds must contain exactly A and B')
        return {key: pair(command_bounds[key], key) for key in ('A', 'B')}
    interval = pair(command_bounds, 'command_bounds')
    return {'A': interval, 'B': interval}

def _limits(limits, n):
    defaults = {'e_min': -math.inf, 'e_max': math.inf, 'charge_max': math.inf, 'discharge_max': math.inf, 'load_min': math.inf, 'load_max': None, 'pv_min': None, 'import_max': None, 'reserve_min': None}
    if limits is None:
        limits = {}
    if not isinstance(limits, Mapping) or set(limits) - set(defaults):
        raise ValueError('unknown operating limit; see API contract')
    merged = defaults | dict(limits)
    for key in ('load_max', 'pv_min', 'import_max'):
        if merged[key] is not None and any((merged[k] is None for k in ('load_max', 'pv_min', 'import_max'))):
            raise ValueError('import certification needs load_max, pv_min, import_max together')
    result = {key: None if value is None else _series(value, n, key) for key, value in merged.items()}
    for key in ('charge_max', 'discharge_max', 'load_min', 'load_max', 'pv_min', 'import_max', 'reserve_min'):
        if result[key] is not None and any((x < 0 for x in result[key])):
            raise ValueError(f'{key}: nonnegative bound required')
    if any((lo > hi for lo, hi in zip(result['e_min'], result['e_max']))):
        raise ValueError('e_min exceeds e_max')
    if result['reserve_min'] is not None and any((not math.isfinite(x) for x in result['e_min'])):
        raise ValueError('reserve_min needs finite e_min')
    return result

def _psi(power, eta_c, eta_d):
    return power / eta_d if power >= 0 else eta_c * power

class _BridgeFamily:

    def __init__(self, N, a, r):
        self.N, self.a, self.r = (N, a, r)
        self.v, self.nv = (a * r, N * a * r)
        self.loglam = None if a == 1 else math.log1p(-a)
        self.lam = 1 - a
        self.powers = [1.0] + [0.0] * N if a == 1 else [math.exp(j * self.loglam) for j in range(N + 1)]
        self.one_minus = [0.0] + [1.0] * N if a == 1 else [-math.expm1(j * self.loglam) for j in range(N + 1)]
        self.thresholds = [k * self.v + r * self.one_minus[N - k] for k in range(N)]
        self.thresholds[-1] = self.nv

    def bridge(self, z):
        if z == 0:
            return ([0.0] * self.N, (0.0, 0.0))
        sign, mag = (1.0 if z > 0 else -1.0, abs(z))
        if mag > self.nv:
            if mag - self.nv > 8 * math.ulp(max(mag, self.nv)):
                raise ArithmeticError('midpoint arithmetic left bridge domain')
            mag = self.nv
        if mag == self.nv:
            path = [sign * self.v * (self.N - j) for j in range(1, self.N + 1)]
            edge = -sign * self.lam * self.r
            return (path, (-math.inf, edge) if sign > 0 else (edge, math.inf))
        k = bisect_left(self.thresholds, mag)
        m = self.N - k
        y = mag - k * self.v
        denom = self.one_minus[m]
        beta = -sign * self.powers[m] * y / denom
        path = []
        for j in range(1, self.N + 1):
            if j <= k:
                path.append(sign * (mag - self.v * j))
            else:
                t = j - k
                diff = self.powers[t] * self.one_minus[m - t]
                path.append(sign * y * diff / denom)
        return (path, (beta, beta))

    def path(self, P, X):
        first, pre_a = self.bridge(P - X)
        second, pre_b = self.bridge(X)
        return ([X + p for p in first] + second, (X + pre_a[0], X + pre_a[1]), pre_b)

def _choose(preimage, allowed):
    lower, upper = (max(preimage[0], allowed[0]), min(preimage[1], allowed[1]))
    if lower > upper:
        return None
    return max(lower, min(0.0, upper))

def _physical_checks(P, E_b, powers, energies, limits, eta_d):
    conflicts, min_slacks = ([], {})

    def check(name, slack, i):
        if math.isnan(slack):
            conflicts.append({'constraint': name, 'step': i, 'slack': 'nan'})
        elif slack < 0:
            conflicts.append({'constraint': name, 'step': i, 'slack': slack})
        min_slacks[name] = min(min_slacks.get(name, math.inf), slack)
    check('initial_energy_lower', E_b - limits['e_min'][0], 0)
    check('initial_energy_upper', limits['e_max'][0] - E_b, 0)
    check('initial_charge', P + limits['charge_max'][0], 0)
    check('initial_discharge', limits['discharge_max'][0] - P, 0)
    for j, p in enumerate(powers):
        i, E = (j + 1, energies[j + 1])
        check('charge', limits['charge_max'][j] + p, i)
        check('discharge', limits['discharge_max'][j] - p, i)
        check('energy_lower', E - limits['e_min'][j], i)
        check('energy_upper', limits['e_max'][j] - E, i)
        check('gross_load_absorption', limits['load_min'][j] - p, i)
        if limits['import_max'] is not None:
            worst_import = max(0.0, limits['load_max'][j] - limits['pv_min'][j] - p)
            check('import', limits['import_max'][j] - worst_import, i)
        if limits['reserve_min'] is not None:
            check('reserve', eta_d * (E - limits['e_min'][j]) - limits['reserve_min'][j], i)
    return (conflicts, min_slacks)

def solve_bridge(P, E_b, E_target, N, h, a, r, eta_c, eta_d, limits=None, command_bounds=None, *, energy_tol=1e-08, power_tol=1e-09, max_iterations=96):
    result = {'version': VERSION, 'algorithm': 'float64_midpoint_bisection', 'exact_arithmetic': False, 'status': 'invalid_input', 'reason': '', 'A': None, 'B': None, 'X': None, 'powers': [], 'energies': [], 'terminal_energy_residual': None, 'terminal_power_residual': None, 'iterations': 0, 'function_evaluations': 0}
    try:
        if isinstance(N, bool) or not isinstance(N, int) or N < 1:
            raise ValueError('N must be a positive integer')
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        P, E_b, E_target, h, a, r, eta_c, eta_d, energy_tol, power_tol = [_number(x, name) for x, name in zip((P, E_b, E_target, h, a, r, eta_c, eta_d, energy_tol, power_tol), ('P', 'E_b', 'E_target', 'h', 'a', 'r', 'eta_c', 'eta_d', 'energy_tol', 'power_tol'))]
        if not (h > 0 and r > 0 and (0 < a <= 1) and (0 < eta_c <= 1) and (0 < eta_d <= 1)):
            raise ValueError('need h,r>0; 0<a,eta_c,eta_d<=1')
        if energy_tol <= 0 or power_tol <= 0:
            raise ValueError('positive numerical tolerances required')
        operating = _limits(limits, 2 * N)
        permitted = _bounds(command_bounds)
        family = _BridgeFamily(N, a, r)
        if not math.isfinite(family.nv) or family.nv == 0:
            raise ValueError('N*a*r is outside usable float64 range')
    except (TypeError, ValueError, OverflowError) as exc:
        result['reason'] = str(exc)
        return result
    result['numerical_tolerances'] = {'energy_kwh': energy_tol, 'power_kw': power_tol, 'physical_constraint_slack': 0.0}
    result['inputs'] = dict(P=P, E_b=E_b, E_target=E_target, N=N, h=h, a=a, r=r, eta_c=eta_c, eta_d=eta_d, limits=deepcopy(dict(limits or {})), command_bounds=deepcopy(command_bounds))
    lo, hi = (max(P - family.nv, -family.nv), min(P + family.nv, family.nv))
    result['midpoint_interval'] = [lo, hi]
    if lo > hi:
        gap = lo - hi
        roundoff = 8 * math.ulp(max(abs(P), family.nv))
        if gap <= roundoff:
            X = P * 0.5
            result['domain_roundoff_adjustment'] = {'raw_interval': [lo, hi], 'gap_kw': gap, 'roundoff_bound_kw': roundoff, 'candidate_midpoint_kw': X, 'scope': 'numerical_witness_only_not_exact_reachability'}
            lo = hi = X
            result['midpoint_interval'] = [lo, hi]
        else:
            result.update(status='unreachable', reason='empty_midpoint_interval_abs_P_gt_2Nar')
            return result
    desired_loss = E_b - E_target
    root_goal = min(energy_tol * 0.1, h * min(eta_c, 1 / eta_d) * power_tol * 0.1)

    def evaluate(X):
        result['function_evaluations'] += 1
        path, pre_a, pre_b = family.path(P, X)
        loss = h * math.fsum((_psi(p, eta_c, eta_d) for p in path))
        return (loss, path, pre_a, pre_b)
    try:
        left, right = (evaluate(lo), evaluate(hi))
        result['reachable_energy_interval'] = [E_b - right[0], E_b - left[0]]
        if not (math.isfinite(left[0]) and math.isfinite(right[0])):
            raise ArithmeticError('nonfinite endpoint energy')
        if desired_loss < left[0] - energy_tol or desired_loss > right[0] + energy_tol:
            result.update(status='unreachable', reason='target_outside_two_hold_energy_interval')
            return result
        best_x, best = min(((lo, left), (hi, right)), key=lambda z: abs(z[1][0] - desired_loss))
        if abs(best[0] - desired_loss) > root_goal:
            for iteration in range(1, max_iterations + 1):
                X = lo + (hi - lo) * 0.5
                if X == lo or X == hi:
                    break
                current = evaluate(X)
                result['iterations'] = iteration
                if abs(current[0] - desired_loss) < abs(best[0] - desired_loss):
                    best_x, best = (X, current)
                if abs(current[0] - desired_loss) <= root_goal:
                    break
                if current[0] < desired_loss:
                    lo = X
                else:
                    hi = X
        if abs(best[0] - desired_loss) > energy_tol:
            raise ArithmeticError('bounded_midpoint_search_did_not_meet_energy_tolerance')
        ideal_path, pre_a, pre_b = best[1:]
        A, B = (_choose(pre_a, permitted['A']), _choose(pre_b, permitted['B']))
        result.update(X=best_x, target_preimages={'A': list(pre_a), 'B': list(pre_b)}, A=A, B=B)
        if A is None or B is None:
            result.update(status='constraint_conflict', reason='target_preimage_and_command_bounds_disjoint')
            return result
        p, E = (P, E_b)
        powers, energies = ([], [E_b])
        for i in range(2 * N):
            target = A if i < N else B
            p = p + a * max(-r, min(r, target - p))
            E = E - h * _psi(p, eta_c, eta_d)
            powers.append(p)
            energies.append(E)
        path_error = max((abs(x - y) for x, y in zip(ideal_path, powers)))
        result.update(powers=powers, energies=energies, terminal_energy_residual=E - E_target, terminal_power_residual=p, max_formula_replay_power_error=path_error, midpoint_replay_residual=powers[N - 1] - best_x, root_energy_residual=desired_loss - best[0])
        if abs(E - E_target) > energy_tol or abs(p) > power_tol or path_error > power_tol:
            raise ArithmeticError('encoded_command_replay_did_not_meet_numerical_tolerance')
        conflicts, slacks = _physical_checks(P, E_b, powers, energies, operating, eta_d)
        result.update(constraint_conflicts=conflicts, minimum_slacks=slacks)
        if conflicts:
            result.update(status='constraint_conflict', reason='unique_path_violates_supplied_operating_domain')
        else:
            result.update(status='reachable', reason='numerical_witness_within_declared_tolerances', certification_scope='unclipped_recurrence_and_supplied_local_domain_only')
        return result
    except (ArithmeticError, OverflowError, ValueError) as exc:
        result.update(status='numerical_failure', reason=str(exc))
        return result

def make_command_schedule(solution, onset_step, N, *, issue_period_steps=60, delay_steps=60, validity_steps=120, zero_until_step=None):
    if solution.get('status') != 'reachable':
        raise ValueError('reachable numerical witness required')
    if N != solution.get('inputs', {}).get('N'):
        raise ValueError('calendar N must match the synthesized witness')
    ints = (onset_step, N, issue_period_steps, delay_steps, validity_steps)
    if any((isinstance(x, bool) or not isinstance(x, int) for x in ints)):
        raise ValueError('calendar arguments must be integers')
    if N <= 0 or issue_period_steps <= 0 or delay_steps < 0:
        raise ValueError('invalid calendar parameters')
    if validity_steps < delay_steps + issue_period_steps:
        raise ValueError('renewed commands do not cover the next issue/effect interval')
    if onset_step % issue_period_steps or N % issue_period_steps or delay_steps % issue_period_steps:
        raise ValueError('both hold boundaries and delay must align with native issue calendar')
    end = onset_step + 2 * N
    if zero_until_step is None:
        zero_until_step = end + issue_period_steps
    if not isinstance(zero_until_step, int) or zero_until_step < end or zero_until_step % issue_period_steps:
        raise ValueError('zero hold endpoint must align and follow bridge end')
    rows = []
    for effective in range(onset_step, max(end + 1, zero_until_step), issue_period_steps):
        block = 'A' if effective < onset_step + N else 'B' if effective < end else 'zero'
        rows.append({'issue_step': effective - delay_steps, 'effective_step': effective, 'valid_until_step': effective - delay_steps + validity_steps, 'block': block, 'target_kw': 0.0 if block == 'zero' else solution[block]})
    return rows

def json_safe(value):
    if isinstance(value, float) and (not math.isfinite(value)):
        if math.isnan(value):
            return 'nan'
        return '+inf' if value > 0 else '-inf'
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

def check_execution_step(solution, step_index, *, previous_power, previous_energy, applied_target, lagged_power, actual_power, actual_energy, safety_conflicts=(), power_tol=1e-08, energy_tol=1e-08):
    reasons = [str(x) for x in safety_conflicts]
    if solution.get('status') != 'reachable':
        return {'ok': False, 'abort': True, 'reasons': ['no_reachable_bridge_witness']}
    if not isinstance(step_index, int) or not 0 <= step_index < len(solution['powers']):
        return {'ok': False, 'abort': True, 'reasons': ['invalid_bridge_step']}
    params = solution['inputs']
    expected_prev_p = params['P'] if step_index == 0 else solution['powers'][step_index - 1]
    expected_prev_E = solution['energies'][step_index]
    expected_p, expected_E = (solution['powers'][step_index], solution['energies'][step_index + 1])
    expected_u = solution['A'] if step_index < params['N'] else solution['B']
    residuals = {'entering_power': previous_power - expected_prev_p, 'entering_energy': previous_energy - expected_prev_E, 'applied_target': applied_target - expected_u, 'lagged_power': lagged_power - expected_p, 'physical_clipping': actual_power - lagged_power, 'actual_power': actual_power - expected_p, 'actual_energy': actual_energy - expected_E}
    for name, value in residuals.items():
        tol = energy_tol if 'energy' in name else power_tol
        if not math.isfinite(value) or abs(value) > tol:
            reasons.append(name + '_mismatch')
    return {'ok': not reasons, 'abort': bool(reasons), 'reasons': reasons, 'step_index': step_index, 'residuals': residuals}
