import csv, gzip, hashlib, json, math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
PUBLIC_FIELDS = ('public_event_id', 'event_time_utc', 'event_type', 'site', 'session_id', 'station_id', 'candidate_id', 'declaration_group_id', 'declaration_state', 'declared_work_kWh', 'declared_deadline_utc')
CSV_FIELDS = ('run_id',) + PUBLIC_FIELDS
EVENT_TYPES = {'CONNECT', 'UNPLUG', 'FIRST_REQUEST', 'DECLARATION_STATE'}
VALID = 'VALID_CURRENT_DECLARED_TOTAL_REQUEST'
INVALID = {'INVALID_CURRENT_DECLARATION_NO_FALLBACK', 'AMBIGUOUS_CURRENT_DECLARATION_NO_FALLBACK'}

def dt(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.utcoffset() is None:
        raise ValueError('timezone required')
    return value

def public_event_id(event):
    recorded = dt(event['event_time_utc']).astimezone(timezone.utc).isoformat()
    key = ['ACN_PUBLIC_EVENT_V2', event['site'], event['session_id'], recorded, event['event_type'], event['declaration_group_id']]
    return hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()

def validate_public_event(event, now):
    if set(event) != set(PUBLIC_FIELDS):
        raise ValueError('non-whitelisted public fields')
    if any((not isinstance(value, str) for value in event.values())):
        raise ValueError('text representation required')
    if event['event_type'] not in EVENT_TYPES:
        raise ValueError('unknown event type')
    if dt(event['event_time_utc']) != now:
        raise ValueError('mixed-time or future event')
    if not event['site'] or not event['session_id']:
        raise ValueError('missing visible identity')
    if event['public_event_id'] != public_event_id(event):
        raise ValueError('public ID disagrees with visible identity')
    declaration = event['event_type'] in ('FIRST_REQUEST', 'DECLARATION_STATE')
    if not declaration:
        if any((event[k] for k in ('candidate_id', 'declaration_group_id', 'declaration_state', 'declared_work_kWh', 'declared_deadline_utc'))):
            raise ValueError('declaration fields on physical event')
        return
    if not event['declaration_group_id']:
        raise ValueError('missing current group identity')
    state = event['declaration_state']
    if state != VALID and state not in INVALID:
        raise ValueError('unknown declaration state')
    if state == VALID:
        work = float(event['declared_work_kWh'])
        if not math.isfinite(work) or work <= 0:
            raise ValueError('nonfinite or nonpositive valid work')
        if dt(event['declared_deadline_utc']) <= now:
            raise ValueError('expired valid declaration')
    elif event['declared_work_kWh'] or event['declared_deadline_utc']:
        raise ValueError('invalid declaration must retain unknown q/deadline')
    if event['event_type'] == 'FIRST_REQUEST':
        if state != VALID or not event['candidate_id']:
            raise ValueError('invalid first request')
    elif event['candidate_id']:
        raise ValueError('later declaration cannot create a candidate')
