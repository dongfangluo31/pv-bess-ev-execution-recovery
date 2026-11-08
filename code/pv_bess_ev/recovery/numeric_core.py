# EN: Second-by-second forward sensitivity & constraint-margin evaluator for 58-variable numerical terminal recovery (SLSQP).
# 中文：面向 58 自由分钟变量数值终端恢复（SLSQP）的逐秒前向灵敏度（Jacobian）与物理约束裕度评估核。
# EN: Sign convention: signed BESS command/power u, p in [-charge_limit_kw, +discharge_limit_kw] (p > 0 discharge, p < 0 charge).
# 中文：符号约定：有符号功率指令与跟踪功率 u, p 取值于 [-充电上限, +放电上限]（p > 0 为放电，p < 0 为充电）。

from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Problem:
    energy_capacity_kwh: float
    charge_limit_kw: float
    discharge_limit_kw: float
    soc_min: float
    soc_max: float
    eta_charge: float
    eta_discharge: float
    ramp_kw_per_second: float
    lag_seconds: float
    initial_energy_kwh: float
    initial_power_kw: float
    target_energy_kwh: float
    load_kw: float
    pv_kw: float
    PCC_limit_kw: float
    held_reserve_kwh: float
    prefix_commands_kw: tuple
    free_minutes: int
    zero_tail_seconds: int = 60
    command_period_seconds: int = 60
    power_tolerance_kw: float = 1e-09
    energy_tolerance_kwh: float = 1e-08
    require_no_projection: bool = True

    def __post_init__(self):
        vals = [v for k, v in vars(self).items() if k not in ('prefix_commands_kw', 'require_no_projection')]
        if not all((math.isfinite(v) for v in vals)):
            raise ValueError('Finite explicitly supplied model required')
        if self.command_period_seconds != 60 or len(self.prefix_commands_kw) != 60:
            raise ValueError('Original 60 second immutable prefix and command period required')
        if type(self.free_minutes) is not int or self.free_minutes < 1:
            raise ValueError('At least one free minute command required')
        if self.zero_tail_seconds != 60:
            raise ValueError('Original 60 second final zero hold required')
        if not (self.energy_capacity_kwh > 0 and self.charge_limit_kw > 0 and (self.discharge_limit_kw > 0) and (self.ramp_kw_per_second > 0) and (self.lag_seconds >= 0) and (0 < self.eta_charge <= 1) and (0 < self.eta_discharge <= 1) and (0 <= self.soc_min < self.soc_max <= 1)):
            raise ValueError('Invalid device model')
        if min(self.load_kw, self.pv_kw, self.PCC_limit_kw, self.held_reserve_kwh) < 0:
            raise ValueError('Nonnegative released nominal point and held reserve required')
        if not (self.emin <= self.initial_energy_kwh <= self.emax and self.emin <= self.target_energy_kwh <= self.emax):
            raise ValueError('Actual initial state and unchanged target must be in SOC bounds')
        if any((not math.isfinite(u) or not -self.charge_limit_kw <= u <= self.discharge_limit_kw for u in self.prefix_commands_kw)):
            raise ValueError('Invalid issued-prefix command')
        if not -self.charge_limit_kw <= self.initial_power_kw <= self.discharge_limit_kw:
            raise ValueError('Invalid actual initial tracked power')
        if self.power_tolerance_kw != 1e-09 or self.energy_tolerance_kwh != 1e-08:
            raise ValueError('Original endpoint tolerances must be retained')

    @property
    def emin(self):
        return self.soc_min * self.energy_capacity_kwh

    @property
    def emax(self):
        return self.soc_max * self.energy_capacity_kwh

    @property
    def alpha(self):
        return 1.0 if self.lag_seconds == 0 else 1.0 - math.exp(-1.0 / self.lag_seconds)

    @property
    def bridge_index(self):
        return 60 + 60 * self.free_minutes

    @property
    def seconds(self):
        return self.bridge_index + self.zero_tail_seconds

def initial_guess(problem):
    gap = problem.target_energy_kwh - problem.initial_energy_kwh
    hours = problem.free_minutes / 60.0
    signed = -gap / (problem.eta_charge * hours) if gap >= 0 else -gap * problem.eta_discharge / hours
    signed = min(problem.discharge_limit_kw, max(-problem.charge_limit_kw, signed))
    return np.full(problem.free_minutes, signed, dtype=float)

def evaluate(problem, controls):
    b = problem
    u = np.asarray(controls, dtype=float)
    n = b.free_minutes
    if u.shape != (n,) or not np.isfinite(u).all():
        raise ValueError('One finite independent command per free minute required')
    seconds = b.seconds
    e = np.empty(seconds + 1)
    p = np.empty(seconds + 1)
    je = np.zeros((seconds + 1, n))
    jp = np.zeros((seconds + 1, n))
    e[0] = b.initial_energy_kwh
    p[0] = b.initial_power_kw
    # EN: Concatenate 60-s immutable already-issued prefix, 58x60-s optimizable minute holds, and 60-s mandatory zero-power terminal hold.
    # 中文：拼接首 60 秒已下发不可变前缀指令、58 个 60 秒待优化分钟保持指令，以及末 60 秒强制零功率保持段。
    commands = np.concatenate((np.asarray(b.prefix_commands_kw), np.repeat(u, 60), np.zeros(60)))
    grid = np.empty(seconds)
    throughput = np.empty(seconds)
    projection = np.empty(seconds)
    grid_jac = np.zeros(n)
    throughput_jac = np.zeros(n)
    kinds = ['SOC_lower', 'SOC_upper', 'charge_rating', 'discharge_rating', 'PCC', 'reserve', 'no_export', 'ramp']
    if b.require_no_projection:
        kinds.append('no_projection')
    margins = np.empty((seconds, len(kinds)))
    margin_jac = np.empty((seconds, len(kinds), n))
    zero = np.zeros(n)
    h = 1.0 / 3600.0
    net = b.load_kw - b.pv_kw
    for i in range(seconds):
        previous, energy = (p[i], e[i])
        dp, de = (jp[i], je[i])
        du = np.zeros(n)
        if 60 <= i < b.bridge_index:
            du[(i - 60) // 60] = 1.0
        difference = commands[i] - previous
        ramped = min(max(commands[i], previous - b.ramp_kw_per_second), previous + b.ramp_kw_per_second)
        z = previous + b.alpha * (ramped - previous)
        if -b.ramp_kw_per_second <= difference <= b.ramp_kw_per_second:
            dz = (1.0 - b.alpha) * dp + b.alpha * du
        else:
            dz = dp.copy()
        if z < 0.0:
            rawcap = (b.emax - energy) / (b.eta_charge * h)
            cap = max(rawcap, 0.0)
            candidates = [-z, b.charge_limit_kw, cap]
            derivs = [-dz, zero, -de / (b.eta_charge * h) if rawcap > 0 else zero]
            k = int(np.argmin(candidates))
            c = candidates[k]
            dc = derivs[k]
            discharge = 0.0
            dd = zero
            power = -c
            derivative = -dc
        else:
            rawcap = b.eta_discharge * (energy - b.emin) / h
            candidates = [z, b.discharge_limit_kw, max(rawcap, 0.0), max(net, 0.0)]
            derivs = [dz, zero, b.eta_discharge * de / h if rawcap > 0 else zero, zero]
            k = int(np.argmin(candidates))
            discharge = candidates[k]
            dd = derivs[k]
            c = 0.0
            dc = zero
            power = discharge
            derivative = dd
        p[i + 1] = power
        jp[i + 1] = derivative
        e[i + 1] = energy + b.eta_charge * c * h - discharge * h / b.eta_discharge
        je[i + 1] = de + b.eta_charge * dc * h - dd * h / b.eta_discharge
        raw_grid = net + c - discharge
        grid[i] = max(raw_grid, 0.0)
        if raw_grid > 0:
            grid_jac += (dc - dd) * h
        throughput[i] = c + discharge
        throughput_jac += (dc + dd) * h
        projection[i] = abs(power - z)
        actual_delta = power - previous
        noexport_raw = net + c
        values = [e[i + 1] - b.emin, b.emax - e[i + 1], b.charge_limit_kw - c, b.discharge_limit_kw - discharge, b.PCC_limit_kw - grid[i], b.eta_discharge * (e[i + 1] - b.emin) - b.held_reserve_kwh, max(noexport_raw, 0.0) - discharge, b.ramp_kw_per_second - abs(actual_delta)]
        derivs = [je[i + 1], -je[i + 1], -dc, -dd, -(dc - dd) if raw_grid > 0 else zero, b.eta_discharge * je[i + 1], (dc if noexport_raw > 0 else zero) - dd, -np.sign(actual_delta) * (derivative - dp)]
        if b.require_no_projection:
            values.append(b.power_tolerance_kw - projection[i])
            derivs.append(-np.sign(power - z) * (derivative - dz))
        margins[i] = values
        margin_jac[i] = np.stack(derivs)
    ineq, jac, labels = ([], [], [])
    # EN: Aggregate per-minute worst-case (minimum) constraint margins and exact subgradients across SOC, converter, PCC, reserve, no-export, and ramp.
    # 中文：按每 60 秒窗口提取 SOC、变流器、并网限额、备用、防逆流及斜率约束的最不利裕度及其解析梯度。
    for start in range(0, seconds, 60):
        stop = min(start + 60, seconds)
        for k, name in enumerate(kinds):
            idx = start + int(np.argmin(margins[start:stop, k]))
            ineq.append(margins[idx, k])
            jac.append(margin_jac[idx, k])
            labels.append(f'{name}@{idx}')
    for j, target in enumerate(u):
        idx = j * 60
        energy = e[idx]
        de = je[idx]
        dc = np.zeros(n)
        dd = np.zeros(n)
        c = max(-target, 0.0)
        discharge = max(target, 0.0)
        if target < 0:
            dc[j] = -1.0
        elif target > 0:
            dd[j] = 1.0
        upper_soc_charge = (b.emax - energy) / (b.eta_charge / 60.0)
        upper_soc_discharge = b.eta_discharge * (energy - b.emin) * 60.0
        raw_absorb = net + c
        values = [upper_soc_charge - c, upper_soc_discharge - discharge, max(raw_absorb, 0.0) - discharge, b.PCC_limit_kw - (net + c - discharge)]
        derivs = [-de / (b.eta_charge / 60.0) - dc, b.eta_discharge * de * 60.0 - dd, (dc if raw_absorb > 0 else zero) - dd, -dc + dd]
        for name, value, deriv in zip(['command_SOC_charge', 'command_SOC_discharge', 'command_no_export', 'command_REF_unchanged'], values, derivs):
            ineq.append(value)
            jac.append(deriv)
            labels.append(f'{name}@decision{idx}')
    equality = np.array([e[-1] - b.target_energy_kwh, p[b.bridge_index]])
    equality_jac = np.stack([je[-1], jp[b.bridge_index]])
    tail_p = p[b.bridge_index:]
    tail_e = e[b.bridge_index:]
    zero_power_error = float(np.max(np.abs(tail_p)))
    zero_energy_error = float(np.max(np.abs(tail_e - b.target_energy_kwh)))
    minimum_safety = float(np.min(margins[:, :8]))
    command_violation = max(float(np.max(u - b.discharge_limit_kw)), float(np.max(-b.charge_limit_kw - u)), 0.0)
    gates = dict(terminal_energy_error_kwh=float(e[-1] - b.target_energy_kwh), bridge_energy_error_kwh=float(e[b.bridge_index] - b.target_energy_kwh), bridge_power_kw=float(p[b.bridge_index]), terminal_power_kw=float(p[-1]), final_zero_max_power_kw=zero_power_error, final_zero_max_energy_error_kwh=zero_energy_error, minimum_physical_margin=minimum_safety, minimum_command_margin=float(min(ineq[-4 * n:])), max_physical_projection_kw=float(np.max(projection)), command_bound_violation_kw=command_violation)
    gates['accepted'] = bool(minimum_safety >= 0.0 and gates['minimum_command_margin'] >= 0.0 and (command_violation == 0.0) and (zero_power_error <= b.power_tolerance_kw) and (zero_energy_error <= b.energy_tolerance_kwh) and (not b.require_no_projection or gates['max_physical_projection_kw'] <= b.power_tolerance_kw))
    return dict(energy_kwh=e, power_kw=p, commands_kw=commands, grid_kw=grid, throughput_kw=throughput, physical_projection_kw=projection, ineq=np.asarray(ineq), ineq_jac=np.stack(jac), ineq_labels=labels, equality=equality, equality_jac=equality_jac, grid_energy_kwh=float(np.sum(grid) * h), grid_energy_jac=grid_jac, throughput_kwh=float(np.sum(throughput) * h), throughput_jac=throughput_jac, command_square_integral=float(np.sum(u * u) / 60.0), command_square_jac=2 * u / 60.0, gates=gates, modeled_steps=seconds, free_command_count=n, no_future_truth=True, no_bridge_initialization=True)
