from collections.abc import Mapping, Sequence
import math
from bridge import solve_bridge as _core_solve, make_command_schedule, check_execution_step, json_safe, VERSION

def validate_operating_support(limits, steps):
    if limits is None:
        return []
    if not isinstance(limits, Mapping):
        return ['limits_must_be_a_mapping']
    values, errors = ({}, [])
    for key in ('e_min', 'e_max', 'charge_max', 'discharge_max', 'load_min'):
        if key in limits and limits[key] is None:
            errors.append(key + '_cannot_be_null')
    for key in ('load_min', 'load_max', 'pv_min'):
        if key not in limits or limits[key] is None:
            continue
        raw = limits[key]
        if isinstance(raw, Sequence) and (not isinstance(raw, (str, bytes))):
            if len(raw) != steps:
                errors.append(key + '_length_mismatch')
                continue
            raw = list(raw)
        else:
            raw = [raw] * steps
        try:
            row = [float(x) for x in raw]
            if any((isinstance(x, bool) for x in raw)) or any((not math.isfinite(x) or x < 0 for x in row)):
                errors.append(key + '_requires_finite_nonnegative_support')
            else:
                values[key] = row
        except (TypeError, ValueError, OverflowError):
            errors.append(key + '_invalid_support_value')
    if 'load_min' in values and 'load_max' in values:
        if any((lo > hi for lo, hi in zip(values['load_min'], values['load_max']))):
            errors.append('empty_load_support_load_min_exceeds_load_max')
    return errors

def solve_bridge(P, E_b, E_target, N, h, a, r, eta_c, eta_d, limits=None, command_bounds=None, *, energy_tol=1e-08, power_tol=1e-09, max_iterations=96):
    if isinstance(N, int) and (not isinstance(N, bool)) and (N > 0):
        errors = validate_operating_support(limits, 2 * N)
        if errors:
            return {'version': VERSION, 'algorithm': 'float64_midpoint_bisection', 'exact_arithmetic': False, 'status': 'invalid_input', 'reason': 'invalid_environmental_support', 'input_errors': errors, 'A': None, 'B': None, 'X': None, 'powers': [], 'energies': [], 'terminal_power_residual': None, 'terminal_energy_residual': None, 'iterations': 0, 'function_evaluations': 0}
    result = _core_solve(P, E_b, E_target, N, h, a, r, eta_c, eta_d, limits=limits, command_bounds=command_bounds, energy_tol=energy_tol, power_tol=power_tol, max_iterations=max_iterations)
    result['environmental_support_input_checked'] = isinstance(N, int) and (not isinstance(N, bool)) and (N > 0)
    return result
