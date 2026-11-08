import copy
import hashlib
import json
from bridge import _BridgeFamily, _limits, _physical_checks, _bounds, json_safe

def encode_charge_tail_zero_once(original):
    conflicts = original.get('constraint_conflicts', [])
    inputs = original.get('inputs', {})
    eligible = original.get('status') == 'constraint_conflict' and conflicts and all((x['constraint'] == 'gross_load_absorption' and -1e-09 <= x['slack'] < 0 for x in conflicts)) and (inputs.get('E_b', 0) < inputs.get('E_target', 0)) and (original.get('B') is not None) and (abs(original['B']) <= 1e-09)
    if not eligible:
        return original
    w = copy.deepcopy(original)
    N = inputs['N']
    a = inputs['a']
    r = inputs['r']
    h = inputs['h']
    p = inputs['P']
    e = inputs['E_b']
    powers = []
    energies = [e]
    A = original['A']
    B = 0.0
    bounds = _bounds(inputs['command_bounds'])
    for k in range(2 * N):
        u = A if k < N else B
        p += a * max(-r, min(r, u - p))
        e -= h * (p / inputs['eta_d'] if p >= 0 else p * inputs['eta_c'])
        powers.append(p)
        energies.append(e)
    violations, slacks = _physical_checks(inputs['P'], inputs['E_b'], powers, energies, _limits(inputs['limits'], 2 * N), inputs['eta_d'])
    ideal, _, _ = _BridgeFamily(N, a, r).path(inputs['P'], original['X'])
    path_error = max((abs(p - q) for p, q in zip(powers, ideal)))
    passed = not violations and bounds['A'][0] <= A <= bounds['A'][1] and (bounds['B'][0] <= B <= bounds['B'][1]) and (abs(e - inputs['E_target']) <= 1e-08) and (abs(p) <= 1e-09) and (path_error <= 1e-09)
    identity = hashlib.sha256(json.dumps(json_safe(original), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    adjustment = dict(version='CHARGE_TAIL_ZERO_ENCODING_ONCE', attempts=1, original_witness_sha256=identity, old_B_hex=original['B'].hex(), new_B_hex=B.hex(), physical_state_projected=False, thresholds_unchanged=True, passed=passed, new_constraint_conflicts=violations, new_minimum_slacks=slacks, new_terminal_energy_residual=e - inputs['E_target'], new_terminal_power_residual=p, new_max_formula_replay_power_error=path_error)
    if not passed:
        w['encoding_adjustment'] = adjustment
        return w
    w.update(status='reachable', reason='single_zero_B_encoding_satisfied_constraints', B=B, powers=powers, energies=energies, terminal_energy_residual=e - inputs['E_target'], terminal_power_residual=p, max_formula_replay_power_error=path_error, midpoint_replay_residual=powers[N - 1] - w['X'], constraint_conflicts=[], minimum_slacks=slacks, encoding_adjustment=repair, original_rejected_witness=original, encoded_target_scope='NUMERICAL_ORIGINAL_RECURRENCE_WITNESS_WITH_ORIGINAL_ENDPOINT_AND_PATH_TOLERANCES')
    return w
