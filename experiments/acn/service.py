from fractions import Fraction as F
from datetime import datetime, timezone
from collections import Counter
from ids import digest
from event_interface import PUBLIC_FIELDS, validate_public_event, dt, VALID
ORIGIN = datetime(2020, 2, 1, tzinfo=timezone.utc)
END = 29 * 86400
RATE = F(36, 5)
PER_SECOND = RATE / 3600

def sec(text):
    delta = dt(text) - ORIGIN
    value = F(delta.days * 86400 + delta.seconds) + F(delta.microseconds, 1000000)
    if value.denominator != 1:
        raise ValueError('Recorded integer second required')
    return int(value)

class ServiceLedger:

    def __init__(self, site):
        self.site = site
        self.now = F(0)
        self.sessions = {}
        self.active = {}
        self.counts = Counter()
        self.released = []
        self.outcomes = []

    def observe(self, t, events):
        if F(t) != self.now:
            raise ValueError('Event must reach current service time')
        for e in events:
            validate_public_event(e, dt(e['event_time_utc']))
            if e['site'] != self.site or sec(e['event_time_utc']) != t:
                raise ValueError('Mixed site/time batch')
        for e in events:
            if e['event_type'] != 'CONNECT':
                continue
            sid = e['session_id']
            if sid in self.sessions:
                raise ValueError('Repeated month CONNECT')
            self.sessions[sid] = dict(site=self.site, session_id=sid, station_id=e['station_id'], connect_s=t, connect_public_id=e['public_event_id'], connected=True, first_request=None, requested=F(0), remaining=F(0), delivered=F(0), on_time_delivered=F(0), deadline_s=None, complete_s=None, unplug_s=None, request_status='NO_RELEASED_FIRST_REQUEST', later_declarations=0, invalid_later_declarations=0)
            self.counts['month_connect'] += 1
        for e in events:
            if e['event_type'] != 'UNPLUG':
                continue
            sid = e['session_id']
            s = self.sessions.get(sid)
            if s is None:
                self.counts['left_censored_UNPLUG'] += 1
                continue
            s['connected'] = False
            s['unplug_s'] = t
            self.active.pop(sid, None)
            self.counts['month_cohort_unplug'] += 1
        for e in events:
            if e['event_type'] not in ('FIRST_REQUEST', 'DECLARATION_STATE'):
                continue
            sid = e['session_id']
            s = self.sessions.get(sid)
            if s is None:
                self.counts['left_censored_' + e['event_type']] += 1
                self.outcomes.append(dict(public_event_id=e['public_event_id'], session_id=sid, t=t, status='LEFT_CENSORED_NO_MODEL_SERVICE'))
                continue
            if e['event_type'] == 'DECLARATION_STATE':
                s['later_declarations'] += 1
                s['invalid_later_declarations'] += int(e['declaration_state'] != VALID)
                self.counts['later_declaration_not_added'] += 1
                continue
            if s['first_request'] is not None:
                raise ValueError('Repeated FIRST_REQUEST')
            s['first_request'] = dict(e)
            self.counts['month_cohort_first_request'] += 1
            if not s['connected']:
                s['request_status'] = 'NOT_CONNECTED_AT_RELEASE'
                self.counts['not_connected_at_release'] += 1
                continue
            q = F(e['declared_work_kWh'])
            deadline = sec(e['declared_deadline_utc'])
            if q <= 0 or deadline <= t or e['declaration_state'] != VALID:
                raise ValueError('Invalid FIRST_REQUEST reached service')
            s.update(requested=q, remaining=q, deadline_s=deadline, request_status='FIXED_COUNTERFACTUAL_SERVICE_NO_ADMISSION_GUARANTEE')
            self.active[sid] = s
            self.counts['serviced_candidates'] += 1
            self.outcomes.append(dict(public_event_id=e['public_event_id'], session_id=sid, t=t, status=s['request_status'], requested_kwh_exact=str(q), deadline_s=deadline))
        self.released.extend((dict(e) for e in events))

    def advance(self, end_s):
        end = F(end_s)
        segments = []
        if end < self.now:
            raise ValueError('Service time reversal')
        while self.now < end:
            stop = min([end] + [self.now + s['remaining'] / PER_SECOND for s in self.active.values()])
            delta = stop - self.now
            power = RATE * len(self.active)
            segments.append(dict(start_s_exact=str(self.now), end_s_exact=str(stop), demand_kw_exact=str(power), energy_kwh_exact=str(power * delta / 3600), active_sessions=len(self.active)))
            for sid, s in list(self.active.items()):
                delivered = PER_SECOND * delta
                s['remaining'] -= delivered
                s['delivered'] += delivered
                ontime = max(F(0), min(stop, F(s['deadline_s'])) - self.now)
                s['on_time_delivered'] += PER_SECOND * ontime
                if s['remaining'] < 0:
                    raise ValueError('Source service overshoot')
                if s['remaining'] == 0:
                    s['complete_s'] = str(stop)
                    self.active.pop(sid)
            self.now = stop
        return segments

    def visible(self):
        return [dict(session_id=sid, station_id=s['station_id'], connect_s=s['connect_s'], first_release_s=sec(s['first_request']['event_time_utc']), declared_deadline_s=s['deadline_s'], requested_kwh_exact=str(s['requested']), delivered_kwh_exact=str(s['delivered']), remaining_kwh_exact=str(s['remaining'])) for sid, s in sorted(self.active.items())]

    def forecast_known(self, hours=24):
        result = []
        for h in range(hours):
            energy = sum((max(F(0), min(s['remaining'] - RATE * h, RATE)) for s in self.active.values()), F(0))
            result.append(float(energy))
        return result

    def snapshot(self):
        states = []
        for sid, s in sorted(self.sessions.items()):
            states.append({k: str(v) if isinstance(v, F) else v for k, v in s.items()})
        return dict(site=self.site, now_s_exact=str(self.now), counts=dict(self.counts), sessions=states)

    def final_rows(self):
        for sid, s in sorted(self.sessions.items()):
            r = {k: str(v) if isinstance(v, F) else v for k, v in s.items() if k != 'first_request'}
            first = s['first_request']
            r['first_request_public_id'] = None if first is None else first['public_event_id']
            r['first_release_s'] = None if first is None else sec(first['event_time_utc'])
            r['right_censored'] = bool(s['connected'] and (s['remaining'] > 0 or s['request_status'] == 'NO_RELEASED_FIRST_REQUEST'))
            r['declared_total_kwh_exact'] = None if first is None else first['declared_work_kWh']
            r['unplugged_unmet'] = bool(s['remaining'] > 0 and (not s['connected']))
            r['completed'] = s['requested'] > 0 and s['remaining'] == 0
            r['on_time_completed'] = r['completed'] and F(s['complete_s']) <= s['deadline_s']
            r['late_delivered_kwh_exact'] = str(s['delivered'] - s['on_time_delivered'])
            r['unknown_request_not_zero'] = s['request_status'] == 'NO_RELEASED_FIRST_REQUEST'
            yield r
