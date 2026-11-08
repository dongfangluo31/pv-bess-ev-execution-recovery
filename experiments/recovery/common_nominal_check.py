# EN: Independent physical-balance, causal-timing, and numerical consistency verification module.
# 中文：独立物理功率能量平衡、因果时序与数值一致性核验模块。

import time
from public_adapter import from_public_bytes
from numeric_core import evaluate

def check_minute_controls(public_blob, *, expected_snapshot_sha256, minute_controls_kw):
    started = time.perf_counter()
    problem, metadata = from_public_bytes(public_blob, expected_sha256=expected_snapshot_sha256)
    if len(minute_controls_kw) != 58:
        raise ValueError('Both branches must supply exactly 58 minute controls')
    v = evaluate(problem, minute_controls_kw)
    q = metadata['issue_s']
    calendar = [dict(control_decision_s=q + 60 * j, effective_s=q + 60 * (j + 1), valid_until_s=q + 60 * j + 120, signed_kw=float(x)) for j, x in enumerate(minute_controls_kw)]
    calendar.extend((dict(control_decision_s=q + 60 * j, effective_s=q + 60 * (j + 1), valid_until_s=q + 60 * j + 120, signed_kw=0.0) for j in (58, 59)))
    return dict(status='NOMINAL_GATE_PASS' if v['gates']['accepted'] else 'NOMINAL_GATE_REFUSED', accepted=bool(v['gates']['accepted']), same_checker_for_both_methods=True, metadata=metadata, gates=v['gates'], common_cost_yuan=0.6 * v['grid_energy_kwh'] + 0.08 * v['throughput_kwh'], nominal_grid_kwh=v['grid_energy_kwh'], nominal_throughput_kwh=v['throughput_kwh'], minute_controls_kw=[float(x) for x in minute_controls_kw], calendar=calendar, powers_kw=v['power_kw'].tolist(), energies_kwh=v['energy_kwh'].tolist(), commands_kw=v['commands_kw'].tolist(), model_evaluations=1, model_seconds=problem.seconds, optimizer_calls=0, actual_plant_calls=0, hidden_truth_reads=0, measured_scope='nominal model at q current point and held reserve only', wall_seconds=time.perf_counter() - started)
