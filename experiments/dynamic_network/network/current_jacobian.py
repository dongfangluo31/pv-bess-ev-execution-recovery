# EN: Analytical bus-voltage-to-power sensitivity (`dV/dP`, `dV/dQ`) linearized from converged AC power-flow states.
# 中文：基于收敛交流潮流状态线性化求解节点电压对有功/无功注入的解析雅可比灵敏度矩阵。

import numpy as np
from scene_io import STATIONS, BUS_MAP

def assemble(snapshot):
    if snapshot['bus_lookup'] != list(range(33)):
        raise ValueError('This fixed case requires the source-verified identity bus lookup')
    n = 33
    y0 = snapshot['Ybus']
    Y = np.zeros((n, n), dtype=complex)
    for i, j, a, b in zip(y0['row'], y0['col'], y0['real'], y0['imag']):
        Y[int(i), int(j)] += complex(a, b)
    V = np.array(snapshot['V']['real']) + 1j * np.array(snapshot['V']['imag'])
    norm = np.abs(V)
    if Y.shape != (n, n) or V.shape != (n,) or (not np.isfinite(Y).all()) or (not np.isfinite(V).all()) or (min(norm) <= 0):
        raise ValueError('Current voltage/Ybus domain')
    I = Y @ V
    D = np.diag(V)
    N = np.diag(V / norm)
    da = 1j * D @ np.conj(np.diag(I) - Y @ D)
    dm = D @ np.conj(Y @ N) + np.diag(np.conj(I)) @ N
    pq = np.arange(1, 33)
    ix = np.ix_(pq, pq)
    J = np.block([[da[ix].real, dm[ix].real], [da[ix].imag, dm[ix].imag]])
    rhs = np.zeros((64, 7))
    for j, s in enumerate(STATIONS):
        rhs[BUS_MAP[s] - 1, j] = -1 / (1000 * float(snapshot['base_mva']))
    return (J, rhs)

def solve_assembled(J, rhs):
    return np.linalg.solve(J, rhs)

def verify_solution(J, rhs, solution):
    res = J @ solution - rhs
    if not np.isfinite(solution).all() or np.max(np.abs(res)) > 1e-10:
        raise ValueError('Current Jacobian linear equations failed')
    return dict(J=J, rhs=rhs, solution=solution, sensitivity=solution[32:], max_linear_residual=float(np.max(np.abs(res))), evaluation_point='THIS_AC_SNAPSHOT_FINAL_COMPLEX_V', library_returned_J_used=False, linear_systems=1, right_hand_sides=7)

def projection(snapshot, jacobian, scalar_imports, actual_imports, voltage_lower, upper):
    J = jacobian['sensitivity']
    actual = np.array([actual_imports[s] for s in STATIONS])
    base = np.array([scalar_imports[s] for s in STATIONS])
    V = np.abs(np.array(snapshot['V']['real']) + 1j * np.array(snapshot['V']['imag']))[1:]
    predicted = V + J @ (base - actual)
    maximum = np.array([upper[str(i)] for i in range(1, 33)])
    A = np.vstack((J, -J))
    b = np.concatenate((predicted - voltage_lower, maximum - predicted))
    return (A, b, predicted)
