"""
Seed script — run once after docker compose up to populate the database
and trigger a comparison so the dashboard is immediately viewable.

The baseline prompt is read from prompts/production.yaml — the same file
GitHub Actions reads on every push and pull request — so this local demo
always reflects whatever prompt is actually considered "live," instead of
a second hardcoded copy that could quietly drift out of sync with it.

The candidate prompt below is intentionally NOT read from anywhere real —
it's a deliberately vaguer instruction, hardcoded here on purpose, so the
demo always has a visible regression to show on the dashboard.

Usage:
    python seed.py

Then open the URL this script prints at the end.
"""

import time
import httpx
import yaml

BASE_URL = "http://localhost:8000"
PRODUCTION_PROMPT_PATH = "prompts/production.yaml"


def post(path, payload):
    r = httpx.post(f"{BASE_URL}{path}", json=payload)
    r.raise_for_status()
    return r.json()


def get(path):
    r = httpx.get(f"{BASE_URL}{path}")
    r.raise_for_status()
    return r.json()


def patch(path):
    r = httpx.patch(f"{BASE_URL}{path}")
    r.raise_for_status()
    return r.json()


def load_production_prompt():
    with open(PRODUCTION_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def wait_for_api():
    print("Waiting for API to be ready...")
    for _ in range(30):
        try:
            httpx.get(f"{BASE_URL}/health").raise_for_status()
            print("API is ready.\n")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("API did not become healthy in time. Is docker compose up running?")


def poll_until_completed(run_id: int, label: str) -> dict:
    print(f"Waiting for {label} (run {run_id}) to complete", end="", flush=True)
    for _ in range(60):
        run = get(f"/evaluations/{run_id}")
        if run["status"] == "COMPLETED":
            print(f" done. ({len(run['results'])} results saved)")
            return run
        if run["status"] == "FAILED":
            raise RuntimeError(f"{label} run {run_id} failed: {run.get('error_message')}")
        if run["status"] == "PARTIAL":
            failed = [f"case {r['dataset_id']} [{r['error_category']}] {r['error_message']}"
                      for r in run["results"] if r["status"] == "FAILED"]
            raise RuntimeError(
                f"{label} run {run_id} is PARTIAL ({run.get('error_message')}):\n  " + "\n  ".join(failed)
            )
        print(".", end="", flush=True)
        time.sleep(3)
    raise RuntimeError(f"{label} run {run_id} timed out.")


def main():
    wait_for_api()

    # ── Prompts ────────────────────────────────────────────────────────────
    print("Creating prompts...")
    production_prompt = load_production_prompt()
    baseline_prompt = post("/prompts/", {
        "version_tag": production_prompt["version_tag"],
        "prompt_text": production_prompt["prompt_text"],
        "model_name": production_prompt["model_name"],
    })
    candidate_prompt = post("/prompts/", {
        "version_tag": "v2-vague",
        "prompt_text": "Answer questions.",
        "model_name": "llama3.2",
    })
    print(f"  Baseline prompt id={baseline_prompt['id']} "
          f"({production_prompt['version_tag']}, from {PRODUCTION_PROMPT_PATH})")
    print(f"  Candidate prompt id={candidate_prompt['id']} (v2-vague, hardcoded demo regression)\n")

    # ── Datasets ───────────────────────────────────────────────────────────
    print("Creating datasets...")
    datasets = post("/datasets/bulk", [
        {"input_query": "What is 2 + 2?",
         "expected_output": "2 + 2 equals 4."},
        {"input_query": "What is the capital of France?",
         "expected_output": "The capital of France is Paris."},
        {"input_query": "What colour is the sky?",
         "expected_output": "The sky is blue."},
        {"input_query": "Who wrote Romeo and Juliet?",
         "expected_output": "Romeo and Juliet was written by William Shakespeare."},
        {"input_query": "How many days are in a leap year?",
         "expected_output": "A leap year has 366 days."},
    ])
    dataset_ids = [d["id"] for d in datasets]
    print(f"  Created {len(dataset_ids)} datasets: {dataset_ids}\n")

    # ── Baseline evaluation ────────────────────────────────────────────────
    print("Triggering baseline evaluation...")
    baseline_run = post("/evaluations/run", {
        "prompt_id": baseline_prompt["id"],
        "dataset_ids": dataset_ids,
        "is_baseline": True,
    })
    baseline_run = poll_until_completed(baseline_run["id"], "baseline")
    patch(f"/evaluations/{baseline_run['id']}/set-baseline")
    print(f"  Marked run {baseline_run['id']} as official baseline.\n")

    # ── Candidate evaluation ───────────────────────────────────────────────
    print("Triggering candidate evaluation...")
    candidate_run = post("/evaluations/run", {
        "prompt_id": candidate_prompt["id"],
        "dataset_ids": dataset_ids,
    })
    candidate_run = poll_until_completed(candidate_run["id"], "candidate")

    # ── Done ───────────────────────────────────────────────────────────────
    dashboard_url = (
        f"{BASE_URL}/dashboard/compare"
        f"?baseline_run_id={baseline_run['id']}"
        f"&candidate_run_id={candidate_run['id']}"
    )
    print("\n" + "=" * 60)
    print("Setup complete. Open this URL in your browser:")
    print(f"\n  {dashboard_url}\n")
    print("=" * 60)


if __name__ == "__main__":
    main()