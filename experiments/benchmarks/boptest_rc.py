# EN: BOPTEST single-zone RC building thermal MPC benchmark with comfort-band tightening (theta) under AR(1) ambient forecast errors.
# 中文：AR(1) 环境温度预测误差下带舒适带收缩裕度（theta）的 BOPTEST 单温区 RC 建筑热力 MPC 基准算例。

import hashlib
import math
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from solver_utils import BASE_SEED, lp, write_case_outputs
C_JK = 1000000.0
R_KW = 0.01
U_MIN = -10000.0
U_MAX = 10000.0
ETA = 0.99
T_AMB_OFFSET = 20.0
T_AMB_AMP = 10.0
T_LO = 20.0
T_HI = 22.0
DT_S = 3600.0
HORIZON = 24
N_STEPS = 720
AR1_PHI = 0.7

def plant_coeffs():
    a = math.exp(-DT_S / (R_KW * C_JK))
    b = R_KW * (1.0 - a)
    c = 1.0 - a
    return (a, b, c)

def true_ambient(h):
    return T_AMB_OFFSET + T_AMB_AMP * math.sin(2.0 * math.pi * h / 24.0)

def forecast_errors(sigma_fc, seed, n_steps, horizon):
    rng = np.random.default_rng(seed)
    e = np.zeros(horizon)
    z = rng.standard_normal(horizon) * math.sqrt(1.0 - AR1_PHI ** 2) * sigma_fc
    e[0] = rng.standard_normal() * sigma_fc
    for j in range(1, horizon):
        e[j] = AR1_PHI * e[j - 1] + z[j]
    return e

def run(bench_params, seed, label, output_root: Path):
    theta = float(bench_params['theta'])
    sigma_fc = float(bench_params['sigma_fc'])
    a, b, c = plant_coeffs()
    rng = np.random.default_rng(seed + 7)
    e_lead = forecast_errors(sigma_fc, seed, N_STEPS, HORIZON)
    T_real = T_LO
    solver_rows = []
    rows = []
    total_elec_w = 0.0
    total_viol = 0.0
    failed = 0
    for step in range(N_STEPS):
        h = step
        t_amb_true = true_ambient(h)
        tf = np.array([true_ambient(h + j) + e_lead[j - 1] for j in range(1, HORIZON + 1)])
        n = HORIZON
        cvec = np.concatenate([np.full(2 * n, 1.0 / ETA), np.zeros(n + 1)])
        A_eq = []
        b_eq = []
        for j in range(n):
            row = np.zeros(2 * n + n + 1)
            row[j] = -b
            row[n + j] = b
            row[2 * n + j] = -a
            row[2 * n + j + 1] = 1.0
            A_eq.append(row)
            b_eq.append(c * tf[j])
        A_ub = []
        b_ub = []
        for j in range(1, n + 1):
            row1 = np.zeros(2 * n + n + 1)
            row1[2 * n + j] = 1.0
            A_ub.append(row1)
            b_ub.append(T_HI - theta)
            row2 = np.zeros(2 * n + n + 1)
            row2[2 * n + j] = -1.0
            A_ub.append(row2)
            b_ub.append(-(T_LO + theta))
        A_eq.append(np.r_[np.zeros(2 * n), 1.0, np.zeros(n)])
        b_eq.append(T_real)
        bounds = [(0, U_MAX)] * n + [(0, -U_MIN)] * n + [(None, None)] * (n + 1)
        res, wall = lp(cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, A_eq=np.array(A_eq), b_eq=np.array(b_eq))
        if not res.success:
            failed += 1
            solver_rows.append({'status': res.status, 'nit': res.nit, 'wall_s': wall, 'message': str(res.message)[:120]})
            u = 0.0
        else:
            x = res.x
            u = x[0] - x[n]
            solver_rows.append({'status': res.status, 'nit': res.nit, 'wall_s': wall, 'num_vars': len(cvec), 'num_constraints': len(A_ub) + len(A_eq)})
        total_elec_w += abs(u) / ETA
        T_real = a * T_real + b * u + c * t_amb_true
        viol = max(0.0, T_LO - T_real) + max(0.0, T_real - T_HI)
        total_viol += viol
        rows.append({'step': step, 'T_amb_true': round(t_amb_true, 6), 'u': round(u, 4), 'T_real': round(T_real, 6), 'viol': round(viol, 6)})
    metrics = {'benchmark': 'BOPTEST_TC1_RC_BUILDING', 'theta': theta, 'sigma_fc': sigma_fc, 'forecast_stream_hash': hashlib.sha256(','.join((str(round(float(v), 12)) for v in e_lead)).encode()).hexdigest()[:16], 'electrical_energy_kwh': round(total_elec_w / 1000.0, 6), 'comfort_violation_kh': round(total_viol, 6), 'failed_solves': failed, 'n_steps': N_STEPS, 'rho_sigma_baseline_note': 'rho/sigma computed by aggregate vs theta=0 at equal sigma_fc'}
    traj = pd.DataFrame(rows)
    write_case_outputs(output_root, label, 'boptest1', bench_params, seed, solver_rows, traj, metrics)
    return (label, metrics)
