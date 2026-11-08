# EN: Roald et al. (2017) IEEE 14-bus analytical chance-constrained DC optimal power flow (CC-DC-OPF) with affine AGC participation.
# 中文：计及仿射自动发电控制（AGC）分摊的 IEEE 14 节点解析机会约束直流最优潮流（CC-DC-OPF）基准算例。

import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from solver_utils import lp, write_case_outputs
PD = np.array([0.0, 21.7, 94.2, 47.8, 7.6, 11.2, 0.0, 0.0, 29.5, 9.0, 3.5, 6.1, 13.5, 14.9])
BRANCH = [(1, 2, 0.05917), (1, 5, 0.22304), (2, 3, 0.19797), (2, 4, 0.17632), (2, 5, 0.17388), (3, 4, 0.17103), (4, 5, 0.04211), (4, 7, 0.20912), (4, 9, 0.55618), (5, 6, 0.25202), (6, 11, 0.1989), (6, 12, 0.25581), (6, 13, 0.13027), (7, 8, 0.17615), (7, 9, 0.11001), (9, 10, 0.0845), (9, 14, 0.27038), (10, 11, 0.19207), (12, 13, 0.19988), (13, 14, 0.34802)]
GENS = [(1, 332.4, 20.0, 0.0430292599), (2, 140.0, 20.0, 0.25), (6, 100.0, 40.0, 0.01), (8, 100.0, 40.0, 0.01)]
WIND_BUS = 3
WIND_CAP_MW = 100.0
WIND_MEAN_MW = 40.0
N_BUS = 14
LIMIT_FACTOR = 2.0
SIGMA_FLOOR_COEF = 1.25
NSEG = 4
N_MC = 2000

def segments():
    seg = []
    for bus, pmax, c1, c2 in GENS:
        w = pmax / NSEG
        for k in range(NSEG):
            xk = k * w
            slope = c1 + c2 * (2.0 * xk + w)
            seg.append((bus, slope, w))
    return seg

def build_ptdf():
    n = N_BUS
    B = np.zeros((n - 1, n - 1))
    for f, t, x in BRANCH:
        b = 1.0 / x
        if f > 1:
            B[f - 2, f - 2] += b
        if t > 1:
            B[t - 2, t - 2] += b
        if f > 1 and t > 1:
            B[f - 2, t - 2] -= b
            B[t - 2, f - 2] -= b
    Binv = np.linalg.inv(B)
    PTDF = np.zeros((len(BRANCH), n))
    for i, (f, t, x) in enumerate(BRANCH):
        for j in range(1, n):
            df = Binv[f - 2, j - 1] if f > 1 else 0.0
            dt = Binv[t - 2, j - 1] if t > 1 else 0.0
            PTDF[i, j] = (df - dt) / x
    return PTDF

def flow_sensitivity(PTDF):
    n_src = 15
    K = np.zeros((len(BRANCH), n_src))
    K[:, :14] = PTDF[:, :14]
    K[:, 14] = -PTDF[:, WIND_BUS - 1]
    for b in (1, 2, 6):
        alpha = 1.0 / 3.0
        K[:, :14] += (alpha * PTDF[:, b - 1])[:, None]
        K[:, 14] += -alpha * PTDF[:, b - 1]
    return K

def dispatch_lp(PTDF, fmax, eps, sigma_w_pct, margin, balance, sigma_load):
    seg = segments()
    nv = len(seg)
    cvec = np.array([s[1] for s in seg])
    phi = stats.norm.ppf(1.0 - eps)
    sigma_wind = sigma_w_pct / 100.0 * WIND_CAP_MW
    src_sigma = np.concatenate([sigma_load, [sigma_wind]])
    flow_sigma = np.sqrt(flow_sensitivity(PTDF) ** 2 @ src_sigma ** 2)
    inj_mean = np.zeros(N_BUS)
    inj_mean -= PD
    inj_mean[WIND_BUS - 1] += WIND_MEAN_MW
    mean_flow = PTDF @ inj_mean
    A_ub, b_ub = ([], [])
    A_eq = [np.ones(nv)]
    b_eq = [balance]
    for l in range(len(BRANCH)):
        row = np.zeros(nv)
        for i, (bus, _, _) in enumerate(seg):
            row[i] = PTDF[l, bus - 1]
        A_ub.append(row)
        b_ub.append(fmax[l] * (1 - margin) - phi * flow_sigma[l] - mean_flow[l])
        A_ub.append(-row)
        b_ub.append(fmax[l] * (1 - margin) - phi * flow_sigma[l] + mean_flow[l])
    sigma_omega = np.sqrt(np.sum(sigma_load ** 2) + sigma_wind ** 2)
    for i, (bus, pmax, _, _) in enumerate(GENS):
        alpha_g = 1.0 / 3.0 if bus in (1, 2, 6) else 0.0
        row = np.zeros(nv)
        for j, (b2, _, _) in enumerate(seg):
            if b2 == bus:
                row[j] = 1.0
        A_ub.append(row)
        b_ub.append(pmax - alpha_g * phi * sigma_omega)
        A_ub.append(-row)
        b_ub.append(0.0)
    bounds = [(0, s[2]) for s in seg]
    n_con = len(A_ub) + len(A_eq)
    res, wall = lp(cvec, np.array(A_ub), np.array(b_ub), bounds, A_eq=np.array(A_eq), b_eq=np.array(b_eq))
    return (res, wall, cvec, seg, n_con)

def run(bench_params, seed, label, output_root: Path):
    eps = float(bench_params['epsilon'])
    sigma_w_pct = float(bench_params['sigma_w_pct'])
    margin = float(bench_params.get('margin', 0.0))
    PTDF = build_ptdf()
    K = flow_sensitivity(PTDF)
    sigma_load = 0.1 * PD
    cvec_all = None
    balance = float(np.sum(PD)) - WIND_MEAN_MW
    res0, w0, cvec_all, seg, _ = dispatch_lp(PTDF, np.full(len(BRANCH), 1000000000.0), 0.5, 0.0, 0.0, balance, sigma_load)
    if not res0.success:
        raise RuntimeError(f'deterministic dispatch failed: {res0.message}')
    p_ref = np.zeros(len(GENS))
    for j, (bus, _, _) in enumerate(seg):
        p_ref[[b == bus for b in [g[0] for g in GENS]].index(True)] += res0.x[j]
    f0 = PTDF @ _inj(p_ref)
    sigma_flow_base = np.sqrt(K ** 2 @ np.concatenate([sigma_load, [0.1 * WIND_CAP_MW]]) ** 2)
    fmax = np.maximum(LIMIT_FACTOR * np.abs(f0), SIGMA_FLOOR_COEF * stats.norm.ppf(0.99) * sigma_flow_base)
    fmax = np.maximum(fmax, 1.0)
    res, wall, cvec_all, seg, n_con = dispatch_lp(PTDF, fmax, eps, sigma_w_pct, margin, balance, sigma_load)
    if not res.success:
        metrics = {'benchmark': 'ROALD17_CC_DC_OPF_IEEE14', 'epsilon': eps, 'sigma_w_pct': sigma_w_pct, 'margin': margin, 'status': 'LP_' + str(res.status), 'cost': None, 'realized_violation_mw': None, 'message': str(res.message)[:200]}
        write_case_outputs(output_root, label, 'roald14', bench_params, seed, [{'status': res.status, 'nit': res.nit, 'wall_s': wall}], pd.DataFrame(), metrics)
        return (label, metrics)
    p_cc = np.zeros(len(GENS))
    for j, (bus, _, _) in enumerate(seg):
        p_cc[[b == bus for b in [g[0] for g in GENS]].index(True)] += res.x[j]
    cost = float(cvec_all @ res.x)
    rng = np.random.default_rng(seed + 11)
    rows = []
    viol_sum = 0.0
    stream_tokens = []
    for s in range(N_MC):
        load_dev = rng.standard_normal(14) * sigma_load
        wind_dev = rng.standard_normal() * (sigma_w_pct / 100.0 * WIND_CAP_MW)
        if s < 100:
            stream_tokens.append(','.join((str(round(float(v), 6)) for v in load_dev)) + '|' + str(round(float(wind_dev), 6)))
        omega = float(np.sum(load_dev) - wind_dev)
        inj = np.zeros(N_BUS)
        inj -= PD + load_dev
        inj[WIND_BUS - 1] += WIND_MEAN_MW + wind_dev
        for i, (b, _, _, _) in enumerate(GENS):
            alpha_g = 1.0 / 3.0 if b in (1, 2, 6) else 0.0
            inj[b - 1] += p_cc[i] + alpha_g * omega
        flows = PTDF @ inj
        exceed = np.maximum(np.abs(flows) - fmax, 0.0)
        viol_sum += float(np.sum(exceed))
        rows.append({'scenario': s, 'max_exceed_mw': round(float(np.max(exceed)), 6), 'sum_exceed_mw': round(float(np.sum(exceed)), 6), 'cost_scenario': round(float(cvec_all @ res.x), 6)})
    metrics = {'benchmark': 'ROALD17_CC_DC_OPF_IEEE14', 'epsilon': eps, 'sigma_w_pct': sigma_w_pct, 'margin': margin, 'status': 'optimal', 'scenario_stream_hash': hashlib.sha256(';'.join(stream_tokens).encode()).hexdigest()[:16], 'cost': round(cost, 6), 'p_cc': [round(float(v), 4) for v in p_cc], 'realized_violation_mw': round(viol_sum / N_MC, 6), 'max_line_exceedance_mw': round(float(np.max(pd.DataFrame(rows)['max_exceed_mw'])), 6), 'line_limits_mw': [round(v, 4) for v in fmax], 'rho_sigma_baseline_note': 'rho/sigma computed by aggregate vs (eps=0.01, sigma_w=10%, margin=0)'}
    solver_rows = [{'status': res.status, 'nit': res.nit, 'wall_s': wall, 'num_vars': len(cvec_all), 'num_constraints': n_con}]
    write_case_outputs(output_root, label, 'roald14', bench_params, seed, solver_rows, pd.DataFrame(rows), metrics)
    return (label, metrics)

def _inj(p_g):
    inj = np.zeros(N_BUS)
    inj -= PD
    inj[WIND_BUS - 1] += WIND_MEAN_MW
    for i, (b, _, _, _) in enumerate(GENS):
        inj[b - 1] += p_g[i]
    return inj
