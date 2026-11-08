# EN: Deterministic directed and pseudo-random Q16.16 test-vector generator (seed 20260706) for FPGA verification.
# 中文：面向 FPGA 等价性核验的确定性定向边界与伪随机 Q16.16 测试向量生成器（随机种子 20260706）。

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from .fixed import FilterInput, S32_MAX, S32_MIN, apply_ref_filter, q_from_decimal
DEFAULT_SEED = 20260706

def _hex32(value: int) -> str:
    return f'{value & 4294967295:08x}'

def directed_vectors() -> list[FilterInput]:
    q = q_from_decimal
    return [FilterInput.from_kw([150, 20, 0, 20, 40, 200]), FilterInput.from_kw([243.28, 115.8, 0, 115.8, 80, 200]), FilterInput.from_kw([350, 50, 0, 50, 40, 200]), FilterInput.from_kw([350, 100, 0, 40, 200, 200]), FilterInput.from_kw([250, 0, 10, 0, 100, 200]), FilterInput.from_kw([-25, 0, 0, 0, 0, 200]), FilterInput(q(250), -1, q(10), q(20), q(20), q(200)), FilterInput(S32_MAX, S32_MAX, S32_MAX, S32_MAX, S32_MAX, 0), FilterInput(S32_MIN, 0, 0, 0, 0, S32_MAX)]

def random_vectors(count: int, seed: int) -> list[FilterInput]:
    rng = random.Random(seed)
    vectors: list[FilterInput] = []
    for _ in range(count):
        nominal_import = rng.randint(q_from_decimal(-200), q_from_decimal(800))
        nominal_charge = rng.randint(0, q_from_decimal(250))
        nominal_discharge = rng.randint(0, q_from_decimal(250))
        max_reduce = rng.randint(0, q_from_decimal(250))
        max_add = rng.randint(0, q_from_decimal(250))
        feeder_limit = rng.randint(q_from_decimal(50), q_from_decimal(400))
        vectors.append(FilterInput(nominal_import, nominal_charge, nominal_discharge, max_reduce, max_add, feeder_limit))
    return vectors

def write_vectors(path: Path, vectors: list[FilterInput]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open('w', encoding='ascii', newline='\n') as handle:
        for request in vectors:
            result = apply_ref_filter(request)
            fields = (*request.__dict__.values(), result.reduce_charge, result.add_discharge, result.executed_charge, result.executed_discharge, result.executed_import, result.remaining_overlimit, result.status)
            line = ' '.join((_hex32(value) for value in fields)) + '\n'
            handle.write(line)
            digest.update(line.encode('ascii'))
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=5000, help='number of random vectors')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    vectors = directed_vectors() + random_vectors(args.count, args.seed)
    digest = write_vectors(args.output, vectors)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({'seed': args.seed, 'directed_vectors': len(directed_vectors()), 'random_vectors': args.count, 'total_vectors': len(vectors), 'sha256': digest}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
