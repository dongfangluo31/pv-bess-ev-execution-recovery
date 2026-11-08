import math
import numpy as np

def problem_from_request(request):
    entry = request['entry']
    now = request['now_s']
    packet = request['public_packet']
    m = request['measurement']
    queue = request['known_command_state']
    if now % 60 or m['sample_s'] != now or m['available_s'] > now or (packet['decision_s'] != now) or (queue['as_of_s'] != now):
        raise ValueError('Invalid current request clock')
    if any((r['available_s'] > now for r in packet['records'])):
        raise ValueError('Unreleased observation')
    if request['held_reserve']['issued_s'] > now:
        raise ValueError('Unreleased reserve')
    S = entry['start_s']
    Q = entry['handoff_s']
    T = entry['end_s']
    assert S <= now < T
    commands = []
    active = queue['last_delivered_command']
    pending = queue['pending_commands']
    highest = active['seq'] if active else 0
    if any((c['issue_s'] > now for c in pending)):
        raise ValueError('Future issued command')
    for t in range(now, now + 60):
        for c in sorted(pending, key=lambda c: (c['effective_s'], c['seq'])):
            if c['effective_s'] <= t and c['valid_until_s'] >= t and (c['seq'] > highest):
                active = c
                highest = c['seq']
        commands.append(0.0 if active is None or active['valid_until_s'] < t else active['signed_kw'])
    free = [t for t in range(now + 60, T, 60) if t < T - 60]
    return dict(entry=entry, now=now, seconds=T - now, Q=Q, T=T, free_effects=free, prefix=commands, E=m['soc'] * entry['battery']['energy_kwh'], P=m['tracked_power_kw'], L=packet['demand_point_kw'] if 'demand_point_kw' in packet else packet['ev_point_kw'], PV=packet['pv_point_kw'], R=request['held_reserve']['R'])

def initial_guess(b, warm=None):
    d = b['entry']['battery']
    n = len(b['free_effects'])
    u = np.zeros(n)
    warm = {} if warm is None else {int(k): v for k, v in warm.items()}
    for j, t in enumerate(b['free_effects']):
        target = b['entry']['mid_target_kwh'] if t < b['Q'] else b['entry']['final_target_kwh']
        initial = b['E'] if t < b['Q'] or b['now'] >= b['Q'] else b['entry']['mid_target_kwh']
        phase_end = b['Q'] if t < b['Q'] else b['T'] - 60
        phase_begin = max(b['now'] + 60, b['entry']['handoff_hold_end_s'] if t >= b['Q'] and b['now'] < b['Q'] else b['now'] + 60)
        hours = max(60, phase_end - phase_begin) / 3600
        gap = target - initial
        value = -gap / (d['eta_charge'] * hours) if gap >= 0 else -gap * d['eta_discharge'] / hours
        u[j] = min(d['discharge_limit_kw'], max(-d['charge_limit_kw'], float(warm.get(t, value))))
    return u

def evaluate(b, controls):
    entry = b['entry']
    battery = entry['battery']
    n = len(b['free_effects'])
    u = np.asarray(controls, dtype=float)
    if u.shape != (n,) or not np.isfinite(u).all():
        raise ValueError('Finite free command vector required')
    B = battery['energy_kwh']
    emin = B * battery['soc_min']
    emax = B * battery['soc_max']
    ec = battery['eta_charge']
    ed = battery['eta_discharge']
    Pc = battery['charge_limit_kw']
    Pd = battery['discharge_limit_kw']
    G = entry['feeder_safe_kw']
    net = b['L'] - b['PV']
    R = b['R']
    ramp = entry['dynamics']['ramp_kw_per_second']
    lag = entry['dynamics']['lag_seconds']
    a = 1.0 if lag == 0 else 1 - math.exp(-1 / lag)
    h = 1 / 3600
    N = b['seconds']
    E = np.empty(N + 1)
    P = np.empty(N + 1)
    JE = np.zeros((N + 1, n))
    JP = np.zeros((N + 1, n))
    E[0] = b['E']
    P[0] = b['P']
    zero = np.zeros(n)
    cmd = np.zeros(N)
    cmd[:60] = b['prefix']
    which = np.full(N, -1, dtype=int)
    for j, t in enumerate(b['free_effects']):
        idx = t - b['now']
        cmd[idx:idx + 60] = u[j]
        which[idx:idx + 60] = j
    grid = np.empty(N)
    thr = np.empty(N)
    projection = np.empty(N)
    jg = zero.copy()
    jt = zero.copy()
    margins = np.empty((N, 8))
    mj = np.empty((N, 8, n))
    for i in range(N):
        previous = P[i]
        energy = E[i]
        dp = JP[i]
        de = JE[i]
        du = zero.copy()
        if which[i] >= 0:
            du[which[i]] = 1
        delta = cmd[i] - previous
        limited = min(max(cmd[i], previous - ramp), previous + ramp)
        z = previous + a * (limited - previous)
        dz = (1 - a) * dp + a * du if -ramp <= delta <= ramp else dp.copy()
        if z < 0:
            rawcap = (emax - energy) / (ec * h)
            cap = max(rawcap, 0)
            vs = [-z, Pc, cap]
            ds = [-dz, zero, -de / (ec * h) if rawcap > 0 else zero]
            k = int(np.argmin(vs))
            c = vs[k]
            dc = ds[k]
            d = 0.0
            dd = zero
        else:
            rawcap = ed * (energy - emin) / h
            vs = [z, Pd, max(rawcap, 0.0), max(net, 0.0)]
            ds = [dz, zero, ed * de / h if rawcap > 0 else zero, zero]
            k = int(np.argmin(vs))
            d = vs[k]
            dd = ds[k]
            c = 0.0
            dc = zero
        power = d - c
        der = dd - dc
        P[i + 1] = power
        JP[i + 1] = der
        E[i + 1] = energy + ec * c * h - d * h / ed
        JE[i + 1] = de + ec * dc * h - dd * h / ed
        rawgrid = net + c - d
        grid[i] = max(rawgrid, 0)
        thr[i] = c + d
        projection[i] = abs(power - z)
        if rawgrid > 0:
            jg += (dc - dd) * h
        jt += (dc + dd) * h
        margins[i] = [E[i + 1] - emin, emax - E[i + 1], Pc - c, Pd - d, G - grid[i], ed * (E[i + 1] - emin) - R, max(net + c, 0) - d, ramp - abs(power - previous)]
        mj[i] = [JE[i + 1], -JE[i + 1], -dc, -dd, -(dc - dd) if rawgrid > 0 else zero, ed * JE[i + 1], dc - dd if net + c > 0 else -dd, -np.sign(power - previous) * (der - dp)]
    inequalities = []
    jac = []
    for start in range(0, N, 60):
        for k in range(8):
            idx = start + int(np.argmin(margins[start:min(start + 60, N), k]))
            inequalities.append(margins[idx, k])
            jac.append(mj[idx, k])
    for j, t in enumerate(b['free_effects']):
        idx = t - b['now'] - 60
        energy = E[idx]
        de = JE[idx]
        c = max(-u[j], 0)
        d = max(u[j], 0)
        dc = zero.copy()
        dd = zero.copy()
        if u[j] < 0:
            dc[j] = -1
        elif u[j] > 0:
            dd[j] = 1
        inequalities.extend([(emax - energy) / (ec / 60) - c, ed * (energy - emin) * 60 - d, max(net + c, 0) - d, G - (net + c - d)])
        jac.extend([-de / (ec / 60) - dc, ed * de * 60 - dd, (dc if net + c > 0 else zero) - dd, -dc + dd])
    eq = []
    ej = []
    targets = []
    if b['now'] < b['Q']:
        targets.append((b['Q'], entry['mid_target_kwh']))
    if b['now'] < b['Q'] + 60:
        targets.append((b['Q'] + 60, entry['mid_target_kwh']))
    targets.append((b['T'] - 60, entry['final_target_kwh']))
    for t, target in targets:
        if t > b['now']:
            idx = t - b['now']
            eq.append(E[idx] - target)
            ej.append(JE[idx])
            if t == b['T'] - 60:
                eq.append(P[idx])
                ej.append(JP[idx])
    gate_targets = {str(t): dict(energy_error_kwh=float(E[t - b['now']] - target), power_kw=float(P[t - b['now']])) for t, target in [(b['Q'], entry['mid_target_kwh']), (b['Q'] + 60, entry['mid_target_kwh']), (b['T'] - 60, entry['final_target_kwh']), (b['T'], entry['final_target_kwh'])] if t > b['now']}
    tail = max(0, b['T'] - 60 - b['now'])
    zero_hold_E = float(np.max(np.abs(E[tail:] - entry['final_target_kwh'])))
    zero_hold_P = float(np.max(np.abs(P[tail:])))
    command_violation = max(float(np.max(u - Pd)) if n else 0.0, float(np.max(-Pc - u)) if n else 0.0, 0.0)
    physical_min = float(np.min(margins))
    command_min = float(min(inequalities[-4 * n:])) if n else 0.0
    targets_ok = all((abs(x['energy_error_kwh']) <= 1e-08 and (int(t) < b['T'] - 60 or abs(x['power_kw']) <= 1e-09) for t, x in gate_targets.items()))
    gates = dict(accepted=physical_min >= 0 and command_min >= 0 and (command_violation == 0) and targets_ok and (zero_hold_E <= 1e-08) and (zero_hold_P <= 1e-09), minimum_physical_margin=physical_min, minimum_command_margin=command_min, command_bound_violation_kw=command_violation, targets=gate_targets, final_zero_energy_max=zero_hold_E, final_zero_power_max=zero_hold_P, physical_projection_allowed=True, maximum_projection_kw=float(np.max(projection)), reference_bridge_shape_required=False)
    return dict(energy_kwh=E, power_kw=P, commands_kw=cmd, grid_kw=grid, throughput_kw=thr, projection_kw=projection, ineq=np.asarray(inequalities), ineq_jac=np.stack(jac), equality=np.asarray(eq), equality_jac=np.stack(ej), grid_kwh=float(np.sum(grid) * h), throughput_kwh=float(np.sum(thr) * h), objective=0.6 * float(np.sum(grid) * h) + 0.08 * float(np.sum(thr) * h), objective_jac=0.6 * jg + 0.08 * jt, gates=gates, modeled_steps=N, free_commands=n)
