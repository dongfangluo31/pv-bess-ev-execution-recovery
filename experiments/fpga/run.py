# EN: Q16.16 fixed-point reserve-bounded execution filter (REF) reference verification and UART HIL CLI.
# 中文：Q16.16 定点备用有界执行滤波器（REF）参考模型核验与串口硬件在环（HIL）命令行入口。

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python"))

from ref_hil.fixed import FilterInput, apply_ref_filter, q_to_float
from ref_hil.generate_vectors import DEFAULT_SEED, directed_vectors, random_vectors, write_vectors
from ref_hil.protocol import decode_request, decode_response, encode_request, encode_response


def run_self_test(random_count: int, seed: int, vector_out: Path | None, summary_out: Path | None) -> dict[str, object]:
    # EN: Verify Q16.16 ordered charge cancellation before reserve-bounded discharge and UART CRC-16 framing round-trip.
    # 中文：校验 Q16.16 定点“先撤回充电、归零后按备用余量补放电”顺序法则及 UART CRC-16 帧编解码一致性。
    vectors = directed_vectors() + random_vectors(random_count, seed)
    exclusivity_violations = 0
    for idx, vec in enumerate(vectors, start=1):
        res = apply_ref_filter(vec)
        if vec.nominal_charge >= 0 and vec.nominal_discharge >= 0 and (vec.nominal_charge == 0 or vec.nominal_discharge == 0):
            if res.executed_charge > 0 and res.add_discharge > 0:
                exclusivity_violations += 1
        req_frame = encode_request(idx, vec)
        seq_req, decoded_req = decode_request(req_frame)
        if seq_req != idx or decoded_req != vec:
            raise RuntimeError(f"Request frame round-trip mismatch at vector {idx}")
        resp_frame = encode_response(idx, res)
        seq_resp, decoded_res = decode_response(resp_frame)
        if seq_resp != idx or decoded_res != res:
            raise RuntimeError(f"Response frame round-trip mismatch at vector {idx}")
    if exclusivity_violations != 0:
        raise RuntimeError(f"Detected {exclusivity_violations} charge/add-discharge exclusivity violations")

    sha256 = None
    if vector_out is not None:
        sha256 = write_vectors(vector_out, vectors)
    report: dict[str, object] = {
        "directed_vectors": len(directed_vectors()),
        "random_vectors": random_count,
        "total_vectors": len(vectors),
        "seed": seed,
        "exclusivity_violations": exclusivity_violations,
        "protocol_crc16_roundtrip_passed": True,
        "vectors_sha256": sha256,
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Q16.16 fixed-point REF verification and vector generator.")
    parser.add_argument("--count", type=int, default=5000, help="Number of pseudo-random Q16.16 test vectors.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic RNG seed.")
    parser.add_argument("--vectors-out", type=Path, help="Optional ASCII hex test-vector output path.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--sample-kw", nargs=6, type=float, metavar=("P_IMP", "P_CH", "P_DIS", "MAX_RED_CH", "MAX_ADD_DIS", "CAP"),
                        help="Evaluate a single 6-tuple in kW with Q16.16 fixed-point arithmetic.")
    args = parser.parse_args()

    if args.sample_kw is not None:
        req = FilterInput.from_kw(args.sample_kw)
        res = apply_ref_filter(req)
        payload = {
            "request_q16_16": asdict(req),
            "result_q16_16": asdict(res),
            "result_kw": {k: q_to_float(v) for k, v in asdict(res).items() if k not in {"status", "latency_cycles"}},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    report = run_self_test(args.count, args.seed, args.vectors_out, args.summary_out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
