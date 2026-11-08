# EN: Reconstruct Bus Station hourly EV demand (kW), TOU tariff, and LONG_194D split inputs from raw upstream archives.
# 中文：从上游原始压缩包重建公交场站（Bus Station）逐时充电负荷（kW）、分时电价与 LONG_194D 长序列模型输入。
# EN: Usage: python code/prepare_station_inputs.py --ev-zip Dataset.zip [--pv-station-csv station01.csv --pv-metadata-csv metadata.csv] --output <dir>
# 中文：用法：python code/prepare_station_inputs.py --ev-zip Dataset.zip [--pv-station-csv station01.csv --pv-metadata-csv metadata.csv] --output <输出目录>

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATION = ROOT / "experiments" / "station"
DATASET_HASH = "a97a1e13c0ab0b1dc5533266290b7fe557a563c28db3b53afc9f240f4d369181"
PV_HASH = "a3229d0d3b92e10e8db7563617ac219bda6c7359c613cc8998d778ca291bc12a"
PV_METADATA_HASH = "d148eaeab2000223684449dc83a1d77843668a81e3ad483e4a7c9cda06023544"
TRANSACTIONS = "Dataset/Charging_Data.csv"
TARIFF = "Dataset/Time-of-use_Price.csv"
RAW_FIELDS = [
    "Location Information", "Start Time", "End Time", "Transaction power/kwh",
    "Electricity cost/Yuan", "Transaction Amount/Yuan", "is_abnormal",
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# EN: Filter abnormal records, select Bus Station (53,602 clean sessions), and enforce duration >= 1 min & mean power <= 250 kW (53,536 sessions).
# 中文：剔除异常交易并筛选最大场站 Bus Station（53,602 条有效会话），再按时长 >= 1 分钟且均功率 <= 250 kW 过滤得 53,536 条会话。
def station_sessions(archive):
    with archive.open(TRANSACTIONS) as stream:
        raw = pd.read_csv(stream, encoding="gbk", usecols=RAW_FIELDS)
    start = pd.to_datetime(raw["Start Time"], errors="coerce")
    end = pd.to_datetime(raw["End Time"], errors="coerce")
    hours = (end - start).dt.total_seconds() / 3600
    energy = raw["Transaction power/kwh"]
    clean = (
        hours.gt(0) & hours.le(24) & energy.gt(0) & energy.le(120)
        & raw["is_abnormal"].eq(0)
        & np.isfinite(raw["Electricity cost/Yuan"] / energy)
        & np.isfinite(raw["Transaction Amount/Yuan"] / energy)
    )
    station = raw["Location Information"].astype(str).str.strip()
    largest = station[clean].value_counts().idxmax()
    selected = pd.DataFrame(dict(start=start, end=end, hours=hours, energy=energy))[clean & station.eq(largest)]
    if largest != "Bus Station" or len(selected) != 53602:
        raise ValueError("Bus Station selection does not match the frozen session count")
    for _ in range(2):
        selected = pd.read_csv(io.StringIO(selected.to_csv(index=False)), parse_dates=["start", "end"])
    power = selected.energy / selected.hours
    filtered = selected[selected.hours.ge(1 / 60) & power.gt(0) & power.le(250)]
    if len(filtered) != 53536:
        raise ValueError("Duration and average-power filters do not match the frozen count")
    return filtered


# EN: Allocate each session's constant average power (energy / hours) proportionally across overlapped clock hours (1 h bin => kWh numerically equals hourly mean kW).
# 中文：将会话恒定平均功率按与自然小时重叠时长分摊至 17,546 个整点小时（1 小时步长下 kWh 与小时平均 kW 数值相等）。
def aggregate_hours(sessions):
    first = sessions.start.min().floor("h")
    ticks = pd.date_range(first, sessions.end.max().ceil("h"), freq="h")
    hourly = np.zeros(len(ticks))
    hour = pd.Timedelta(hours=1)
    for start, end, hours, energy in sessions.itertuples(index=False, name=None):
        left = start.floor("h")
        index = int((left - first) / hour)
        power = energy / hours
        while left < end:
            right = left + hour
            overlap = (min(end, right) - max(start, left)).total_seconds() / 3600
            hourly[index] += power * overlap
            left, index = right, index + 1
    if len(ticks) != 17546 or not np.isclose(hourly.sum(), 1115555.31, rtol=0, atol=1e-6):
        raise ValueError("Hourly coverage or energy balance differs from the published input")
    return pd.DataFrame(dict(time=ticks, ev_energy_kwh=hourly, ev_load_kw=hourly))


def check_hourly_reference(rebuilt):
    path = ROOT / "inputs" / "model_inputs" / "bus_station" / "station_hourly_ev_load.csv"
    reference = pd.read_csv(path, parse_dates=["time"])
    if not np.array_equal(rebuilt["time"].to_numpy(), reference["time"].to_numpy()):
        raise ValueError("Frozen and reconstructed hourly timestamps differ")
    columns = ["ev_energy_kwh", "ev_load_kw"]
    error = float(np.abs(rebuilt[columns].to_numpy() - reference[columns].to_numpy()).max())
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError(f"Hourly reconstruction differs from frozen inputs by {error}")
    return error


def portable_metadata(value):
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


# EN: Align PVOD station01 UTC 15-min PV records to UTC+8 hourly means scaled to 300 kW rating (168 h history + 720 h calibration + 4656 h test).
# 中文：将 PVOD station01 的 UTC 15 分钟光伏序列对齐至 UTC+8 整点均值并按 300 kW 额定容量折算（168h 历史 + 720h 校准 + 4656h 测试）。
def build_long_stream(ev_path, price_path, pv_path, metadata_path, stage):
    sys.path.insert(0, str(STATION))
    from feeder_safe.data_pipeline import PreparedDataset, build_split_dataset, write_prepared_dataset

    for path, expected in ((pv_path, PV_HASH), (metadata_path, PV_METADATA_HASH)):
        if sha256(path) != expected:
            raise ValueError(f"Input SHA-256 differs from the published {path.name} file")
    dataset = build_split_dataset(
        "LONG_194D", ev_path=ev_path, price_path=price_path, pv_path=pv_path,
        pv_metadata_path=metadata_path, pv_rated_kw=300.0,
    )
    provenance = dataset.pv_provenance.copy()
    for column in ("source_file", "source_metadata_file"):
        if column in provenance:
            provenance[column] = provenance[column].map(lambda s: Path(s).name)
    wrapped = PreparedDataset(dataset.inputs, provenance, portable_metadata(dataset.manifest))
    write_prepared_dataset(wrapped, stage / "LONG_194D")
    counts = dataset.inputs["split_role"].value_counts().to_dict()
    if counts != {"test": 4656, "calibration": 720, "forecast_history": 168}:
        raise ValueError(f"Unexpected input role counts {counts}")
    test = dataset.inputs[dataset.inputs["split_role"].eq("test")]
    return {
        "split": "LONG_194D", "role_rows": counts,
        "first_test_hour": str(test["timestamp"].iloc[0]), "last_test_hour": str(test["timestamp"].iloc[-1]),
        "test_ev_kwh": float(test["ev_energy_kwh"].sum()), "test_pv_kwh": float(test["pv_energy_kwh"].sum()),
        "test_ev_peak_kw": float(test["ev_load_kw"].max()), "test_pv_peak_kw": float(test["pv_power_kw"].max()),
        "pv_source_sha256": sha256(pv_path), "pv_metadata_sha256": sha256(metadata_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild the published station demand and long-stream inputs.")
    parser.add_argument("--ev-zip", type=Path, required=True)
    parser.add_argument("--pv-station-csv", type=Path)
    parser.add_argument("--pv-metadata-csv", type=Path)
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.pv_station_csv) != bool(args.pv_metadata_csv):
        parser.error("The station CSV and its metadata CSV must be supplied together")
    if sha256(args.ev_zip) != DATASET_HASH:
        parser.error("The charging ZIP must match Figshare v3 file 52886138")
    destination = args.output.resolve()
    if destination.exists():
        parser.error(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="station-rebuild-", dir=destination.parent) as temp:
        stage = Path(temp) / "inputs"
        stage.mkdir()
        with ZipFile(args.ev_zip) as archive:
            sessions = station_sessions(archive)
            tariff = archive.read(TARIFF)
        ev = aggregate_hours(sessions)
        maximum_error = check_hourly_reference(ev)
        ev_path = stage / "station_hourly_ev_load.csv"
        price_path = stage / "Time-of-use_Price.csv"
        ev.to_csv(ev_path, index=False, encoding="utf-8-sig")
        price_path.write_bytes(tariff)
        receipt = {
            "status": "BUS_STATION_HOURLY_INPUT_REBUILT",
            "dataset_sha256": DATASET_HASH, "station": "Bus Station", "clean_session_count": 53602,
            "energy_allocation_session_count": len(sessions), "hours": len(ev),
            "energy_kwh": float(ev.ev_energy_kwh.sum()), "max_abs_reference_difference_kw": maximum_error,
            "output_sha256": sha256(ev_path),
        }
        if args.pv_station_csv:
            receipt["long_stream"] = build_long_stream(
                ev_path, price_path, args.pv_station_csv, args.pv_metadata_csv, stage
            )
            receipt["status"] = "BUS_STATION_AND_LONG_194D_INPUTS_REBUILT"
        os.rename(stage, destination)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
