# EN: Exact analytical midpoint-bridge solver for discrete ramp-before-lag BESS actuator trajectories.
# 中文：离散“先斜率限制、后一阶滞后”储能执行器轨迹的解析中点桥接（Midpoint Bridge）求解器。
# EN: Sign convention here: signed power p = p_discharge - p_charge (kW), so p > 0 discharges and p < 0 charges.
# 中文：本模块符号约定：有符号功率 p = p_discharge - p_charge（kW），即 p > 0 表示放电，p < 0 表示充电。

from __future__ import annotations
import math

# EN: One-second actuator transition: p_k = p_{k-1} + a * clip(u - p_{k-1}, -r, r), with a = 1 - exp(-1 / tau_lag) and ramp cap r (kW/s).
# 中文：1 秒执行器状态递推：p_k = p_{k-1} + a * clip(u - p_{k-1}, -r, r)，其中 a = 1 - exp(-1 / tau_lag)，r 为每秒斜率上限（kW/s）。
def held_target_bridge_step(p_prev: float, u: float, a: float, r: float) -> float:
    ramp_increment = max(-r, min(r, u - p_prev))
    return p_prev + a * ramp_increment

def simulate_two_hold(P: float, A: float, B: float, N: int, a: float, r: float) -> list[float]:
    powers = []
    p = P
    for _ in range(N):
        p = held_target_bridge_step(p, A, a, r)
        powers.append(p)
    for _ in range(N):
        p = held_target_bridge_step(p, B, a, r)
        powers.append(p)
    return powers

# EN: Piecewise-affine stored energy update with efficiency kink at p = 0: discharging costs h * p / eta_d, charging stores -h * eta_c * p.
# 中文：在 p = 0 处带充放电效率折点的分段仿射电量积分：放电消耗 h * p / eta_d，充电补充 -h * eta_c * p。
def compute_terminal_energy(powers: list[float], E_b: float, h: float, eta_d: float, eta_c: float) -> float:
    E = E_b
    for p in powers:
        if p >= 0:
            E -= h * (p / eta_d)
        else:
            E -= h * (eta_c * p)
    return E

# EN: Closed-form N-step bridge driving tracked power error z to 0 at midpoint while returning symmetric hold target beta.
# 中文：闭式求解 N 步过渡段，将跟踪功率偏差 z 精确消除至 0 并给出对称保持指令 beta。
def compute_bridge(z: float, N: int, a: float, r: float) -> tuple[list[float], float]:
    v = a * r
    lam = 1.0 - a
    if abs(z) < 1e-15:
        return ([0.0] * N, 0.0)
    eps = 1.0 if z > 0 else -1.0
    abs_z = abs(z)
    assert abs_z <= N * v + 1e-12, f'|z| > Nv: {abs_z} > {N * v}'
    if abs(abs_z - N * v) < 1e-12:
        return ([eps * v * (N - j) for j in range(1, N + 1)], -eps * lam * r)
    k = 0
    while k < N:
        tk = k * v + r * (1.0 - lam ** (N - k))
        if abs_z <= tk + 1e-12:
            break
        k += 1
    q = lam ** (N - k)
    Y = abs_z - k * v
    beta = -eps * q * Y / (1.0 - q)
    G = []
    for j in range(1, N + 1):
        if j <= k:
            G.append(z - eps * v * j)
        else:
            G.append(eps * Y * (lam ** (j - k) - q) / (1.0 - q))
    return (G, beta)
