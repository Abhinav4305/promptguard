"""
Seed script — run once after docker compose up to populate the database
and trigger a comparison so the dashboard is immediately viewable.

Usage:
    python seed.py

Then open: http://localhost:8000/dashboard/compare?baseline_run_id=1&candidate_run_id=2
"""

import time
import httpx

BASE_URL = "http://localhost:8000"


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
            raise RuntimeError(f"{label} run {run_id} failed.")
        print(".", end="", flush=True)
        time.sleep(3)
    raise RuntimeError(f"{label} run {run_id} timed out.")


def main():
    wait_for_api()

    # ── Prompts ────────────────────────────────────────────────────────────
    print("Creating prompts...")
    baseline_prompt = post("/prompts/", {
        "version_tag": "v1-concise",
        "prompt_text": "You are a concise assistant. Answer in one short sentence.",
        "model_name": "llama3.2",
    })
    candidate_prompt = post("/prompts/", {
        "version_tag": "v2-vague",
        "prompt_text": "Answer questions.",
        "model_name": "llama3.2",
    })
    print(f"  Baseline prompt id={baseline_prompt['id']}")
    print(f"  Candidate prompt id={candidate_prompt['id']}\n")

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