#!/usr/bin/env python3
"""
Measure the release gate on simulated model behaviour (no LLM, no database server).

    python scripts/measure_gate.py --out gate-report.json
    python scripts/measure_gate.py --seeds 3 --suites pr

Run from the repository root. See app/core/gate_measurement.py for what the numbers do
and do not mean: they describe the gate on hand-written answers, not a real model.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("DATABASE_URL", "sqlite://")   # never touch a real database

from app.core.gate_measurement import format_markdown, measure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="benchmarks/core-v2.yaml")
    ap.add_argument("--simulation", default="benchmarks/simulations/core-v2.sim.yaml")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--suites", nargs="+", default=["pr", "full"])
    ap.add_argument("--out", help="write the full JSON report here")
    args = ap.parse_args()

    report = measure(args.benchmark, args.simulation, seeds=args.seeds, suites=tuple(args.suites))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(format_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
