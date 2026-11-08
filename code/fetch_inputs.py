# EN: Fetch, verify (SHA-256), and unpack upstream public datasets registered in inputs/sources.json.
# 中文：根据 inputs/sources.json 目录下载、SHA-256 校验并解包上游公开原始数据集。
# EN: Subcommands: list | download | unpack | prepare-storenet | prepare-pvgis | fetch-acn.
# 中文：支持子命令：list | download | unpack | prepare-storenet | prepare-pvgis | fetch-acn。

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "inputs" / "sources.json"
DATES = ("2020-01-23", "2020-04-04", "2020-07-07", "2020-10-25")
ACN_HOST = "ev.caltech.edu"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA-256 differs for {path.name}: {actual} != {expected}")
    return actual


def safe_name(value):
    if not value or value in (".", "..") or any(c in value for c in '/\\:\x00'):
        raise ValueError(f"Invalid asset filename: {value!r}")
    return value


@contextmanager
# EN: Stage writes in a sibling temporary file and atomically link only after optional SHA-256 verification.
# 中文：在同目录临时文件中写入并完成可选 SHA-256 校验后再原子链接至目标路径，防止半写入文件。
def staged(destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".fetch-", delete=False) as temp:
        path = Path(temp.name)
    try:
        yield path
        if expected:
            check(path, expected)
        if destination.exists():
            if digest(path) != digest(destination):
                raise FileExistsError(f"Existing file differs: {destination}")
        else:
            os.link(path, destination)
    finally:
        path.unlink(missing_ok=True)


def fetch(asset, out, supplied):
    if asset["access"] == "authenticated_api":
        raise ValueError("Use the fetch-acn command with your own token or authorized snapshot")
    name = safe_name(asset["filename"])
    target = out / safe_name(asset["family"]) / name
    sha = asset.get("sha256")
    if not sha:
        raise ValueError(f"{asset['id']}: {asset['reason']}")
    if target.is_file():
        check(target, sha)
        return target
    if supplied:
        candidates = [supplied / name, supplied / asset["family"] / name]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Provide {name} in {supplied}")
        with staged(target, sha) as temp:
            shutil.copyfile(source, temp)
        return target
    url = asset.get("download_url")
    if asset["access"] != "download" or not url:
        raise ValueError(f"{asset['id']}: {asset['reason']} Source: {asset['landing_url']}")
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"Unsupported transport: {asset['id']}")
    with staged(target, sha) as temp, urllib.request.urlopen(url, timeout=90) as remote, temp.open("wb") as dst:
        shutil.copyfileobj(remote, dst, length=1024 * 1024)
    return target


def unpack(asset, file, member, out):
    check(file, asset["sha256"])
    parts = PurePosixPath(member).parts
    if not parts or PurePosixPath(member).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError("Select an exact relative ZIP member")
    for p in parts:
        safe_name(p)
    with zipfile.ZipFile(file) as archive:
        matches = [item for item in archive.infolist() if item.filename == member and not item.is_dir()]
        if len(matches) != 1:
            raise ValueError(f"ZIP member must exist exactly once: {member}")
        with archive.open(matches[0]) as src, staged(out) as temp, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


# EN: Extract the four seasonal 24-h StoreNet Irish household minute windows (1 seed minute + 1440 active minutes, W -> kW).
# 中文：从 StoreNet 爱尔兰家庭分钟级原始表提取四个季节代表日（含首分钟初值共 1441 行，W 折算为 kW）。
def storenet_input(file, out, catalog):
    source = next(x for x in catalog["sources"] if x["id"] == "storenet-h4-w")
    check(file, source["sha256"])
    labels = {}
    days = [[] for _ in DATES]
    for quarter, day in enumerate(DATES):
        start = datetime.fromisoformat(day)
        for index in range(1441):
            label = (start + timedelta(minutes=index - 1)).strftime("%Y-%m-%d %H:%M:%S")
            labels[label] = (quarter, index)
    with file.open(encoding="utf-8-sig", newline="") as raw:
        for line, row in enumerate(csv.DictReader(raw), 2):
            tick = labels.get(row["date"])
            if tick is None:
                continue
            quarter, index = tick
            if len(days[quarter]) != index:
                raise ValueError(f"Duplicate, missing or reordered source minute at line {line}")
            demand = Fraction(row[" Consumption(W)"]) / 1000
            pv = Fraction(row[" Production(W)"]) / 1000
            if min(demand, pv) < 0:
                raise ValueError(f"Negative input at line {line}")
            days[quarter].append((index, row["date"], line, (index - 1) * 60, index * 60,
                                  index * 60, float(demand), float(pv), str(demand), str(pv)))
    header = ("sample", "published_label", "source_line", "support_start_s", "support_end_s",
              "available_s", "load_kw", "pv_kw", "load_kw_exact", "pv_kw_exact")
    for number, rows in enumerate(days, 1):
        if len(rows) != 1441:
            raise ValueError(f"Quarter {number} does not have the required seed plus 1440 rows")
        target = out / f"storenet_h4_q{number}.csv"
        with staged(target) as temp, temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        print(target)


class SameOrigin(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        check_acn_url(new_url)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def check_acn_url(url):
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme != "https" or parts.hostname != ACN_HOST or parts.port not in (None, 443)
            or parts.username or parts.password or not parts.path.startswith("/api/v1/sessions/")):
        raise ValueError("ACN response points outside the authorized API")


def iso_time(value):
    try:
        t = parsedate_to_datetime(value)
    except (ValueError, TypeError):
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"Missing UTC offset in ACN event time: {value}")
    return t.astimezone(timezone.utc).isoformat()


def pv_hourly(asset, source, out):
    if asset["family"] != "PVGIS":
        raise ValueError("Select a PVGIS asset")
    check(source, asset["sha256"])
    data = json.loads(source.read_text(encoding="utf-8-sig"))["outputs"]["hourly"]
    record = asset["derived_hourly_csv"]
    if len(data) != record["record_count"]:
        raise ValueError("PVGIS hourly row count differs")
    target = out / safe_name(record["filename"])
    with staged(target, record["sha256"]) as temp, temp.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["timestamp", "pv_kw_per_kwp"])
        for row in data:
            v = row["P"] / 1000
            writer.writerow([row["time"], str(v) if asset["id"] == "pvgis-dundee-2023" else f"{v:.6f}"])
    print(target)


def snapshot_pages(snapshot, site, asset):
    for expected in asset["pages"]:
        number = expected["page"]
        raw = (snapshot / site / f"page_{number:06d}.json").read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"ACN page {number} differs from the pinned source snapshot for {site}")
        page = json.loads(raw)
        if len(page["_items"]) != expected["rows"]:
            raise ValueError(f"Unexpected row count on ACN page {number}")
        yield page


def live_acn_pages(site, start, end):
    token = os.environ.get("ACN_API_TOKEN")
    if not token:
        raise ValueError("ACN_API_TOKEN is required. Register at https://ev.caltech.edu/dataset")
    endpoint = f"https://{ACN_HOST}/api/v1/sessions/{site}"
    condition = (f'connectionTime >= "{format_datetime(start, usegmt=True)}" and '
                 f'connectionTime < "{format_datetime(end, usegmt=True)}"')
    url = endpoint + "?" + urllib.parse.urlencode({"where": condition, "sort": "connectionTime", "max_results": 100})
    headers = {"Authorization": "Basic " + base64.b64encode((token + ":").encode()).decode(), "Accept": "application/json"}
    opener = urllib.request.build_opener(SameOrigin())
    seen = set()
    while url:
        check_acn_url(url)
        if url in seen:
            raise ValueError("ACN pagination loop")
        seen.add(url)
        with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
            page = json.load(response)
        yield page
        href = page.get("_links", {}).get("next", {}).get("href")
        base = f"https://{ACN_HOST}/api/v1/" if href and href.startswith("sessions/") else url
        url = urllib.parse.urljoin(base, href) if href else None


# EN: Reconstruct February 2020 ACN charging sessions from local raw API pages or live ACN_API_TOKEN requests.
# 中文：从本地原始分页文件或实时 ACN_API_TOKEN 接口提取 2020 年 2 月充电会话并按连接时刻排序。
def acn_fetch(site, out, asset, snapshot):
    start = datetime(2020, 2, 1, tzinfo=timezone.utc)
    end = datetime(2020, 3, 1, tzinfo=timezone.utc)
    pages = snapshot_pages(snapshot, site, asset) if snapshot else live_acn_pages(site, start, end)
    sessions = []
    for page in pages:
        for row in page["_items"]:
            arrival = iso_time(row["connectionTime"])
            if not start <= datetime.fromisoformat(arrival) < end:
                continue
            inputs = []
            for ui in row.get("userInputs") or []:
                supplied = ui.get("modifiedAt")
                if supplied:
                    inputs.append({"declared_utc": iso_time(supplied), "requested_kwh": ui.get("kWhRequested"),
                                   "requested_departure_utc": iso_time(ui["requestedDeparture"]) if ui.get("requestedDeparture") else None,
                                   "minutes_available": ui.get("minutesAvailable")})
            sessions.append({"arrival_utc": arrival, "departure_utc": iso_time(row["disconnectTime"]),
                             "delivered_kwh": row.get("kWhDelivered"), "declarations": inputs})
    sessions.sort(key=lambda row: (row["arrival_utc"], row["departure_utc"], json.dumps(row, sort_keys=True)))
    target = out / f"acn_{site}_2020_02_sessions.jsonl"
    with staged(target) as temp, temp.open("w", encoding="utf-8", newline="\n") as dst:
        for i, row in enumerate(sessions, 1):
            dst.write(json.dumps({"record": i, **row}, separators=(",", ":")) + "\n")
    print(json.dumps({"file": str(target), "sessions": len(sessions), "sha256": digest(target),
                      "source": "frozen_snapshot" if snapshot else "live_monthly_api"}))


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("list")
    get = subs.add_parser("fetch")
    get.add_argument("asset", nargs="+")
    get.add_argument("--out", type=Path, required=True)
    get.add_argument("--from-dir", type=Path)
    extract = subs.add_parser("unpack-zip")
    extract.add_argument("asset")
    extract.add_argument("file", type=Path)
    extract.add_argument("member")
    extract.add_argument("--out", type=Path, required=True)
    prep = subs.add_parser("prepare-storenet")
    prep.add_argument("--csv", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    pv = subs.add_parser("prepare-pvgis")
    pv.add_argument("asset")
    pv.add_argument("--json", type=Path, required=True)
    pv.add_argument("--out", type=Path, required=True)
    acn = subs.add_parser("fetch-acn")
    acn.add_argument("site", choices=["caltech", "jpl", "office001"])
    acn.add_argument("--out", type=Path, required=True)
    acn.add_argument("--from-dir", type=Path, help="existing snapshot directory containing site/page_NNNNNN.json")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assets = {a["id"]: a for a in catalog["sources"]}
    if args.action == "list":
        for a in assets.values():
            print(a["id"], a["access"], a["version"], sep="\t")
    elif args.action == "fetch":
        for key in args.asset:
            print(fetch(assets[key], args.out, args.from_dir))
    elif args.action == "unpack-zip":
        unpack(assets[args.asset], args.file, args.member, args.out)
    elif args.action == "prepare-storenet":
        storenet_input(args.csv, args.out, catalog)
    elif args.action == "prepare-pvgis":
        pv_hourly(assets[args.asset], args.json, args.out)
    elif args.action == "fetch-acn":
        acn_fetch(args.site, args.out, assets["acn-"+args.site], args.from_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error))
