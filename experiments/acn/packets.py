# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

from collections import defaultdict
import csv
from datetime import timedelta
from fractions import Fraction as F
import gzip
import json

from event_interface import PUBLIC_FIELDS, public_event_id, validate_public_event, dt
from raw_projection import build_session, dt as raw_time
from service import ORIGIN, END, RATE, PER_SECOND, sec


def event_from(session, site, kind, stamp, **extra):
    event = dict.fromkeys(PUBLIC_FIELDS, "")
    event.update(event_time_utc=stamp, site=site, session_id=str(session["sessionID"]),
                 station_id=str(session["stationID"]), event_type=kind, **extra)
    event["public_event_id"] = public_event_id(event)
    validate_public_event(event, dt(stamp))
    return event


def from_raw_pages(directory, site):
    pages = sorted(directory.glob("page_*.json"))
    pages = [p for p in pages if not p.name.endswith(".receipt.json")]
    if not pages:
        raise ValueError("Expected ACN API pages named page_*.json in the supplied site's directory")
    bytime = defaultdict(list)
    seen = set()
    end_date = ORIGIN + timedelta(seconds=END)
    for page_path in pages:
        page = json.loads(page_path.read_text(encoding="utf-8-sig"))
        for session in page["_items"]:
            connected, _ = raw_time(session["connectionTime"])
            unplugged, _ = raw_time(session["disconnectTime"])
            if connected is None or unplugged is None:
                raise ValueError("Recorded connection and disconnection timestamps are required")
            if connected >= end_date:
                continue
            if unplugged < ORIGIN and not any(
                (raw_time(item.get("modifiedAt"))[0] or connected) >= ORIGIN
                for item in session.get("userInputs") or [] if isinstance(item, dict)
            ):
                continue
            sid = str(session["sessionID"])
            if sid in seen:
                raise ValueError("Source contains a repeated session")
            seen.add(sid)
            data = build_session(session, {"site": site, "sessionID":sid})
            events = [event_from(session,site,r["physical_event_type"],r["event_time_utc"])
                      for r in data["physical"]]
            first = data["first"]
            for r in data["transitions"]:
                state = r["current_declaration_state"]
                if state == "PREARRIVAL_OR_UNBOUND_QUARANTINE":
                    continue
                is_first = first is not None and r["declaration_group_id"] == first["declaration_group_id"]
                kw = r["current_total_request_kWh"]
                events.append(event_from(session, site, "FIRST_REQUEST" if is_first else "DECLARATION_STATE",
                                         r["event_time_utc"], candidate_id=first["candidate_id"] if is_first else "",
                                         declaration_group_id=r["declaration_group_id"], declaration_state=state,
                                         declared_work_kWh=str(kw) if kw is not None else "",
                                         declared_deadline_utc=r["current_declared_deadline_utc"] or ""))
            for event in events:
                t = sec(event["event_time_utc"])
                if 0 <= t < END:
                    bytime[t].append(event)
    for batch in bytime.values():
        batch.sort(key=lambda e:e["public_event_id"])
    return bytime


def from_recorded_events(path, site):
    bytime = defaultdict(list)
    with gzip.open(path,"rt",encoding="utf-8",newline="") as fh:
        for row in csv.DictReader(fh):
            if row["site"] != site:
                continue
            t=sec(row["event_time_utc"])
            if 0 <= t < END:
                event={k:row[k] for k in PUBLIC_FIELDS}
                validate_public_event(event,dt(event["event_time_utc"]))
                bytime[t].append(event)
    return bytime


def service_intervals(bytime, ledger, event_clock, stop_s, continue_at=None):
    times = sorted(bytime)
    i = 0
    history = []
    hour_energy = F(0)
    while ledger.now < stop_s:
        t = ledger.now
        if continue_at is not None and not continue_at(t):
            return
        if i < len(times) and times[i] == t:
            ledger.observe(int(t),bytime[times[i]])
            i += 1
        now_load = float(RATE*len(ledger.active))
        point = None
        if t.denominator == 1 and t % 3600 == 0:
            known = ledger.forecast_known()
            point = known if len(history)<24 else [max(a,b) for a,b in zip(known,history[-24:])]
        ledger.released.clear()
        next_tick = min(stop_s, F((int(t)//60+1)*60))
        if event_clock:
            completion = min((t+s["remaining"]/PER_SECOND for s in ledger.active.values()),default=next_tick)
            next_external = F(times[i]) if i<len(times) else next_tick
            end = min(next_tick,completion,next_external)
            segments = ledger.advance(end)
        else:
            end = next_tick
            segments = []
            while i<len(times) and times[i]<end:
                segments.extend(ledger.advance(times[i]))
                ledger.observe(times[i],bytime[times[i]])
                i += 1
            segments.extend(ledger.advance(end))
        energy = sum((F(s["energy_kwh_exact"]) for s in segments),F(0))
        hour_energy += energy
        plant = {"start_s":str(t), "end_s":str(end), "segments":segments, "energy_kwh_exact":str(energy)}
        yield t, now_load, point, plant
        if end % 3600 == 0:
            history.append(float(hour_energy))
            hour_energy=F(0)
