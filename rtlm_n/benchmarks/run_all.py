"""Run the full RTLM-N experimental program and emit the pass-gate report.

    python -m rtlm_n.benchmarks.run_all [--seed N] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from rtlm_n.benchmarks import (
    exp0_exact_head,
    exp1_residual,
    exp2_canonical_adapter,
    exp3_jacobian,
    exp4_thousand_models,
    exp5_cascade,
    exp6_heterogeneous,
    exp7_migration,
)

EXPERIMENTS = [
    exp0_exact_head,
    exp1_residual,
    exp2_canonical_adapter,
    exp3_jacobian,
    exp4_thousand_models,
    exp5_cascade,
    exp6_heterogeneous,
    exp7_migration,
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=str, default="rtlmn_report.json")
    args = parser.parse_args()

    packages = []
    for mod in EXPERIMENTS:
        t0 = time.time()
        print("=" * 78)
        proof = mod.run(seed=args.seed, verbose=True)
        print(f"  ({time.time() - t0:.1f}s)")
        packages.append(proof)

    print("=" * 78)
    n_pass = sum(p.passed for p in packages)
    print(f"RTLM-N experimental program: {n_pass}/{len(packages)} experiments passed")
    for p in packages:
        print(f"  [{'PASS' if p.passed else 'FAIL'}] {p.experiment}")

    with open(args.report, "w") as f:
        json.dump([asdict(p) for p in packages], f, indent=2, default=str)
    print(f"proof packages written to {args.report}")
    return 0 if n_pass == len(packages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
