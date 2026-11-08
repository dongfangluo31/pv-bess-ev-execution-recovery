import hashlib, json, math
from numeric_core import Problem

def from_public_bytes(blob, *, expected_sha256):
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Public snapshot bytes differ from frozen identity')
    d = json.loads(blob)
    if d['schema'] != 'MATCHED_TERMINAL_ENTRY' or d['origin_run_id'] != 'EQUAL_BRIDGE_MAY18_NEG':
        raise ValueError('Only the preselected public entry is supported')
    c = d['clock']
    q = c['issue_s']
    b = d['battery']
    m = d['telemetry']
    r = d['held_nominal_reserve']
    s = d['terminal_service']
    wanted = dict(issue_s=27900, onset_s=27960, bridge_end_s=31440, episode_end_s=31500, prefix_seconds=60, controlled_bridge_seconds=3480, zero_tail_seconds=60, total_suffix_seconds=3600, fixed_command_period_s=60, dispatch_delay_s=60, validity_end_offset_s=120, inclusive_expiry=True, ready_deadline_seconds=60)
    if any((c[k] != v for k, v in wanted.items())):
        raise ValueError('Original common command calendar differs')
    if m['sample_s'] != q or m['available_s'] > q or d['known_command_state']['as_of_s'] != q:
        raise ValueError('Actual issue-time own telemetry and queue required')
    if r['issued_s'] > q or r['issued_s'] % 900:
        raise ValueError('Reserve must be a prior/current public outer issue')
    point = d['nominal_current_point']
    if point['reserve_kwh'] != r['value_kwh']:
        raise ValueError('Nominal held reserve may not be replaced by empirical forecast R')
    if d['physical_policy'] != 'PV_PRIORITY_NOEXPORT' or d['dynamics']['plant_step_s'] != 1:
        raise ValueError('Original one-second PV-priority physical model required')
    if s['target_energy_kwh'] != b['energy_kwh'] * b['soc_initial'] or s['target_power_kw'] != 0:
        raise ValueError('Original shared initial inventory and zero-power target required')
    if not s['energy_and_power_required_at_bridge_end'] or not s['all_final60_poststep_states_required']:
        raise ValueError('Bridge and complete zero-hold service required')
    if s['zero_command_from_s'] != c['bridge_end_s'] or s['zero_hold_until_s'] != c['episode_end_s']:
        raise ValueError('Unchanged final zero interval required')
    economic = d['original_reporting_economics']
    if economic['import_price_yuan_per_kwh'] != 0.6 or economic['throughput_price_yuan_per_kwh'] != 0.08 or economic['curtailment_in_common_cost']:
        raise ValueError('Only the original common cost is the planning objective')
    queue = d['known_command_state']
    known = ([queue['last_delivered_command']] if queue.get('last_delivered_command') else []) + list(queue['pending_commands'])
    validated = []
    seqs = set()
    for cmd in known:
        seq = cmd['seq']
        decision = cmd['control_decision_s']
        effect = cmd.get('effective_s', cmd.get('dispatch_slot_s'))
        if type(seq) is not int or seq in seqs or seq <= 0:
            raise ValueError('Unique issued command sequence required')
        seqs.add(seq)
        if decision % 60 or decision > q or (not decision <= cmd['issue_s'] <= q) or (effect != decision + 60) or (cmd['valid_until_s'] != decision + 120) or (cmd['issue_s'] > effect) or (not math.isfinite(cmd['signed_kw'])):
            raise ValueError('Unknown future command or changed issue/effect/expiry semantics')
        validated.append((seq, effect, cmd['valid_until_s'], float(cmd['signed_kw'])))
    prefix = []
    for t in range(q, q + 60):
        active = [x for x in validated if x[1] <= t]
        latest = max(active, key=lambda x: x[0]) if active else None
        prefix.append(0.0 if latest is None or latest[2] < t else latest[3])
    if prefix != d['immutable_prefix_commands_kw']:
        raise ValueError('Declared prefix differs from original queue recurrence')
    if abs(d['stored_energy_kwh'] - m['soc'] * b['energy_kwh']) > 1e-12:
        raise ValueError('Stored energy differs from own actual telemetry')
    problem = Problem(energy_capacity_kwh=b['energy_kwh'], charge_limit_kw=b['charge_limit_kw'], discharge_limit_kw=b['discharge_limit_kw'], soc_min=b['soc_min'], soc_max=b['soc_max'], eta_charge=b['eta_charge'], eta_discharge=b['eta_discharge'], ramp_kw_per_second=d['dynamics']['ramp_kw_per_second'], lag_seconds=d['dynamics']['lag_seconds'], initial_energy_kwh=d['stored_energy_kwh'], initial_power_kw=m['tracked_power_kw'], target_energy_kwh=s['target_energy_kwh'], load_kw=point['load_kw'], pv_kw=point['pv_kw'], PCC_limit_kw=d['feeder_safe_kw'], held_reserve_kwh=r['value_kwh'], prefix_commands_kw=tuple(prefix), free_minutes=58, power_tolerance_kw=s['power_tolerance_kw'], energy_tolerance_kwh=s['energy_tolerance_kwh'], require_no_projection=True)
    return (problem, dict(public_snapshot_sha256=actual, origin_run_id=d['origin_run_id'], issue_s=q, onset_s=c['onset_s'], bridge_end_s=c['bridge_end_s'], episode_end_s=c['episode_end_s'], held_reserve_forecast_sha256=r['forecast_sha256'], held_reserve_issued_s=r['issued_s'], empirical_forecast_not_used_as_extra_constraint=True, actual_future_values_consumed=False, reference_bridge_values_consumed=False))
