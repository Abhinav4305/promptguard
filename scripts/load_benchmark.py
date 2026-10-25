#!/usr/bin/env python3
"""
Load a versioned benchmark file into a running PromptGuard API and print the dataset
ids of one suite. Idempotent: re-running with an unchanged file creates nothing.

    python scripts/load_benchmark.py benchmarks/core-v1.yaml --suite pr
    -> prints e.g. 1,2,3,...   (comma-separated, ready for a CI step output)

Exit codes: 0 ok, 1 invalid file, 2 API refused it (e.g. 409: a published version
changed), 3 API unreachable.
"""
import argparse
import json
import sys

import httpx
import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--suite", default="full", help="suite whose dataset ids are printed (default: full)")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    try:
        with open(args.file, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        print(f"cannot read {args.file}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict) or "cases" not in payload:
        print(f"{args.file} is not a benchmark file (expected keys: benchmark, version, cases)", file=sys.stderr)
        return 1

    try:
        resp = httpx.post(f"{args.api}/datasets/import", json=payload, timeout=args.timeout)
    except httpx.HTTPError as exc:
        print(f"cannot reach {args.api}: {exc}", file=sys.stderr)
        return 3
    if resp.status_code != 200:
        print(f"API refused the benchmark (HTTP {resp.status_code}): {resp.text[:2000]}", file=sys.stderr)
        return 2

    body = resp.json()
    ids = [c["id"] for c in body["cases"] if args.suite in (c.get("suites") or [])]
    if not ids:
        print(f"no cases in suite {args.suite!r}", file=sys.stderr)
        return 1
    print(
        f"benchmark {body['benchmark']} v{body['version']}: {body['created']} created, "
        f"{body['unchanged']} unchanged, suite {args.suite!r} = {len(ids)} cases",
        file=sys.stderr,
    )
    print(",".join(str(i) for i in ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
