# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib, json, math
FUTURE = datetime.max.replace(tzinfo=timezone.utc)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def stable(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def dt(value):
    if value is None or value == '':
        return (None, 'MISSING')
    if not isinstance(value, str):
        return (None, 'INVALID_TYPE')
    try:
        result = parsedate_to_datetime(value)
        if result.tzinfo is None or result.utcoffset() is None:
            return (None, 'MISSING_TIMEZONE')
        return (result.astimezone(timezone.utc), 'VALID')
    except (ValueError, TypeError, OverflowError):
        return (None, 'INVALID_FORMAT')

def work(value):
    if value is None:
        return (None, 'MISSING')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return (None, 'INVALID_TYPE')
    if not math.isfinite(value):
        return (None, 'NONFINITE')
    if value <= 0:
        return (float(value), 'NONPOSITIVE')
    return (float(value), 'VALID')

def iso(value):
    return value.isoformat() if value is not None else None

def build_session(session, origin):
    c, cs = dt(session.get('connectionTime'))
    unplug, us = dt(session.get('disconnectTime'))
    raw_inputs = session.get('userInputs')
    inputs = raw_inputs if isinstance(raw_inputs, list) else []
    arrival = dict(origin, stationID=session.get('stationID'), connectionTime_raw=session.get('connectionTime'), connection_time_utc=iso(c), connection_time_status=cs, dataset_role='ALL_ARRIVAL_DENOMINATOR_NO_FUTURE_DEMAND', _time=c)
    unique, mappings = ({}, [])
    for index, raw in enumerate(inputs):
        signature = stable(raw)
        event_id = sha((origin['site'] + '\x00' + origin['sessionID'] + '\x00' + signature).encode('utf-8'))
        duplicate = event_id in unique
        if not duplicate:
            record = raw if isinstance(raw, dict) else {}
            m, ms = dt(record.get('modifiedAt'))
            d, ds = dt(record.get('requestedDeparture'))
            w, ws = work(record.get('kWhRequested'))
            valid = isinstance(raw, dict) and ms == ds == ws == 'VALID' and (d > m)
            relation = 'UNORDERABLE' if m is None or c is None else 'PREARRIVAL_QUARANTINE' if m < c else 'SAME_RECORDED_SECOND' if m == c else 'POSTARRIVAL'
            unique[event_id] = dict(origin, canonical_event_id=event_id, canonical_raw_input_json=signature, raw_input_value_sha256=sha(signature.encode('utf-8')), first_original_input_index=index, original_input_indices_json=[], duplicate_records_merged=0, modifiedAt_raw=record.get('modifiedAt'), event_time_utc=iso(m), modified_time_status=ms, requestedDeparture_raw=record.get('requestedDeparture'), declared_deadline_utc=iso(d), deadline_status=ds, kWhRequested_raw=record.get('kWhRequested'), work_status=ws, individually_valid_declaration=int(valid), event_relation_to_arrival=relation, modified_minus_connection_seconds=(m - c).total_seconds() if m is not None and c is not None else None, group_id=None, same_timestamp_conflict=0, canonical_events_in_group=0, original_records_in_group=0, effective_declaration_state_after_group='UNORDERABLE_NO_FALLBACK', event_is_hard_admission_ack=0, _m=m, _d=d, _w=w, _valid=valid)
        unique[event_id]['original_input_indices_json'].append(index)
        unique[event_id]['duplicate_records_merged'] += int(duplicate)
        mappings.append(dict(origin, original_input_index=index, canonical_event_id=event_id, raw_input_value_sha256=sha(signature.encode('utf-8')), is_merged_exact_duplicate=int(duplicate)))
    canonical = list(unique.values())
    groups = defaultdict(list)
    for event in canonical:
        if event['_m'] is not None:
            groups[event['_m']].append(event)
    transitions, valid_groups = ([], [])
    sync_group = None
    for m in sorted(groups):
        group = groups[m]
        contracts = {(event['modified_time_status'], event['deadline_status'], event['work_status'], event['_d'], event['_w'], event['_valid']) for event in group}
        conflict = len(contracts) > 1
        good = not conflict and all((event['_valid'] for event in group))
        group_id = sha((origin['site'] + '\x00' + origin['sessionID'] + '\x00' + iso(m)).encode('utf-8'))
        original_n = sum((len(event['original_input_indices_json']) for event in group))
        if c is None or m < c:
            state = 'PREARRIVAL_OR_UNBOUND_QUARANTINE'
        elif conflict:
            state = 'AMBIGUOUS_CURRENT_DECLARATION_NO_FALLBACK'
        elif not good:
            state = 'INVALID_CURRENT_DECLARATION_NO_FALLBACK'
        else:
            state = 'VALID_CURRENT_DECLARED_TOTAL_REQUEST'
        for event in group:
            event.update(group_id=group_id, same_timestamp_conflict=int(conflict), canonical_events_in_group=len(group), original_records_in_group=original_n, effective_declaration_state_after_group=state)
        first = min(group, key=lambda event: event['first_original_input_index'])
        trans = dict(origin, declaration_group_id=group_id, event_time_utc=iso(m), original_modifiedAt_values_json=stable([event['modifiedAt_raw'] for event in group]), canonical_event_ids_json=stable([event['canonical_event_id'] for event in group]), current_declaration_state=state, same_timestamp_conflict=int(conflict), current_total_request_kWh=first['_w'] if state == 'VALID_CURRENT_DECLARED_TOTAL_REQUEST' else None, current_declared_deadline_utc=iso(first['_d']) if state == 'VALID_CURRENT_DECLARED_TOTAL_REQUEST' else None, update_is_additional_work=0, implicit_hard_admission=0, accepted_commitment_state='NOT_OBSERVED_POLICY_LEDGER_REQUIRED', existing_commitment_effect='NO_AUTOMATIC_CANCEL_OR_REACCEPT_RULE_ASSUMED', _m=m)
        transitions.append(trans)
        if c is not None and m == c:
            sync_group = dict(events=group, first=first, good=good, conflict=conflict, group_id=group_id)
        if c is not None and m >= c and good:
            valid_groups.append(dict(events=group, first=first, good=good, conflict=conflict, group_id=group_id))
    for event in canonical:
        event['original_input_indices_json'] = stable(event['original_input_indices_json'])
    first_group = valid_groups[0] if valid_groups else None
    sync = sync_group if sync_group and sync_group['good'] else None

    def candidate(group, model):
        if group is None:
            return None
        event = group['first']
        return dict(origin, candidate_id=model + ':' + origin['site'] + ':' + origin['sessionID'], contract_model=model, stationID=session.get('stationID'), declaration_group_id=group['group_id'], canonical_event_ids_json=stable([e['canonical_event_id'] for e in group['events']]), release_time_utc=iso(event['_m']), modifiedAt_raw=event['modifiedAt_raw'], declared_deadline_utc=iso(event['_d']), requestedDeparture_raw=event['requestedDeparture_raw'], declared_work_kWh=event['_w'], work_semantics='COUNTERFACTUAL_TOTAL_DECLARED_SESSION_WORK_NOT_ORIGINAL_REMAINING', pre_release_metered_energy_kWh=None, pre_release_metering_status='UNKNOWN_IN_SESSION_ENDPOINT_DO_NOT_FILL', service_before_release_rule='NO_HISTORICAL_PRE_RELEASE_CHARGING_REPLAYED', rate_cap_kW=None, rate_cap_status='REQUIRES_DECLARED_HARDWARE_INPUT_NOT_OBSERVED_PEAK', same_second_event_rule='OBSERVE_ALL_EVENTS_STAMPED_T_BEFORE_DECISION_T_NO_SUBSECOND_ORDER_CLAIM', admission_decision=None, admission_status='NOT_DECIDED_CONTROLLER_MUST_DECIDE', later_update_policy='UNRESOLVED_SEPARATE_EVENTS_NOT_ADDED_TO_WORK', early_departure_policy='UNRESOLVED_SEPARATE_PHYSICAL_EVENTS_NOT_DEADLINE_CLIPPING', candidate_scope='DECLARATION_SCHEMA_ONLY_NOT_HARDWARE_FEASIBILITY', _time=event['_m'])
    if not inputs:
        sync_reason = first_reason = 'NO_RECORDED_INPUT'
    elif c is None:
        sync_reason = first_reason = 'INVALID_CONNECTION_TIME'
    else:
        sync_reason = 'INCLUDED' if sync else 'NO_SAME_SECOND_DECLARATION' if sync_group is None else 'SAME_SECOND_CONFLICT' if sync_group['conflict'] else 'SAME_SECOND_INVALID'
        first_reason = 'INCLUDED' if first_group else 'NO_ORDERABLE_DECLARATION' if not groups else 'ONLY_PREARRIVAL_DECLARATIONS' if all((m < c for m in groups)) else 'NO_VALID_AT_OR_AFTER_ARRIVAL'
    ledger = dict(origin, dataset_role='EXCLUDED_SESSIONS_METADATA', original_input_records=len(inputs), canonical_input_records=len(canonical), exact_duplicate_records_merged=len(inputs) - len(canonical), prearrival_canonical_events=sum((e['event_relation_to_arrival'] == 'PREARRIVAL_QUARANTINE' for e in canonical)), sync_subset_status=sync_reason, first_declaration_replay_status=first_reason, same_timestamp_conflict_groups=sum((t['same_timestamp_conflict'] for t in transitions)))
    selected = first_group['first'] if first_group else None
    outcomes = dict(origin, dataset_role='RETROSPECTIVE_OUTCOMES_NEVER_ADMISSION_FEATURES', actual_disconnectTime_raw=session.get('disconnectTime'), actual_disconnect_time_utc=iso(unplug), actual_disconnect_status=us, actual_doneChargingTime_raw=session.get('doneChargingTime'), actual_final_kWhDelivered=session.get('kWhDelivered'), candidate_release_during_historical_connection=int(c is not None and unplug is not None and (selected is not None) and (c <= selected['_m'] < unplug)), first_candidate_deadline_later_than_actual_unplug=int(selected is not None and unplug is not None and (selected['_d'] > unplug)), first_candidate_release_at_or_after_actual_unplug=int(selected is not None and unplug is not None and (selected['_m'] >= unplug)), early_departure_is_future_outcome_not_exclusion=1)
    physical = []
    for name, event_time, raw in (('CONNECT', c, session.get('connectionTime')), ('UNPLUG', unplug, session.get('disconnectTime'))):
        physical.append(dict(origin, physical_event_type=name, event_time_utc=iso(event_time), event_time_raw=raw, available_at_utc=iso(event_time), visibility_rule='REVEAL_ONLY_WHEN_EVENT_OCCURS', _time=event_time))
    return dict(arrival=arrival, canonical=canonical, mapping=mappings, transitions=transitions, sync=candidate(sync, 'ARRIVAL_SAME_RECORDED_SECOND_v1'), first=candidate(first_group, 'FIRST_VALID_POSTARRIVAL_TOTAL_WORK_REPLAY_v1'), ledger=ledger, outcomes=outcomes, physical=physical)
