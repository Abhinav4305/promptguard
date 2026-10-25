"""
API behaviour: run creation, comparison statuses, baseline rules, dashboard.
The Celery task is replaced by a recorder; runs are executed synchronously through
the real executor with a fake LLM.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import evaluations as evaluation_routes
from app.core.run_executor import execute_run
from app.db.models import Dataset, EvaluationResult, EvaluationRun, ResultStatus, RunStatus
from app.db.session import get_db
from app.main import app
from tests.conftest import FakeLLM, fake_resolver, make_resolver


@pytest.fixture()
def enqueued():
    return []


@pytest.fixture()
def client(session_factory, monkeypatch, enqueued):
    def override():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(evaluation_routes.run_evaluation, "delay", lambda rid: enqueued.append(rid))
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def world(db, seed):
    prompt = seed.prompt()
    datasets = seed.datasets(3)
    return prompt, datasets


def run_now(db, seed, prompt, datasets, llm=None):
    """Create a run and execute it to a terminal state with the real executor."""
    r = seed.run(prompt, datasets)
    execute_run(db, r.id, llm_fn=llm or FakeLLM({"q1": "a1", "q2": "a2", "q3": "a3"}), resolver=fake_resolver)
    db.expire_all()
    return r.id


# ── POST /evaluations/run ───────────────────────────────────────────────────────────

def test_trigger_creates_pending_run_with_pending_cases(client, world, enqueued):
    prompt, datasets = world
    resp = client.post("/evaluations/run", json={"prompt_id": prompt.id, "dataset_ids": [d.id for d in datasets]})

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "PENDING" and body["expected_cases"] == 3
    assert body["completed_cases"] == body["succeeded_cases"] == body["failed_cases"] == 0
    assert [r["status"] for r in body["results"]] == ["PENDING"] * 3
    assert enqueued == [body["id"]]


def test_trigger_with_empty_dataset_ids_uses_every_dataset(client, world):
    prompt, datasets = world
    body = client.post("/evaluations/run", json={"prompt_id": prompt.id, "dataset_ids": []}).json()
    assert body["expected_cases"] == 3
    assert sorted(r["dataset_id"] for r in body["results"]) == sorted(d.id for d in datasets)


def test_trigger_rejects_bad_input_without_creating_a_run(client, db, world, enqueued):
    prompt, datasets = world
    ids = [d.id for d in datasets]
    assert client.post("/evaluations/run", json={"prompt_id": 999, "dataset_ids": ids}).status_code == 404
    assert client.post("/evaluations/run", json={"prompt_id": prompt.id, "dataset_ids": [ids[0], ids[0]]}).status_code == 422
    unknown = client.post("/evaluations/run", json={"prompt_id": prompt.id, "dataset_ids": [ids[0], 4242]})
    assert unknown.status_code == 422 and "4242" in unknown.json()["detail"]
    assert db.query(EvaluationRun).count() == 0 and db.query(EvaluationResult).count() == 0
    assert enqueued == []


def test_trigger_with_no_datasets_at_all_is_rejected(client, seed):
    prompt = seed.prompt()
    assert client.post("/evaluations/run", json={"prompt_id": prompt.id, "dataset_ids": []}).status_code == 422


def test_enqueue_failure_marks_run_failed_instead_of_leaving_it_pending(client, db, world, monkeypatch):
    prompt, datasets = world

    def boom(_run_id):
        raise ConnectionError("redis down")

    monkeypatch.setattr(evaluation_routes.run_evaluation, "delay", boom)
    resp = client.post("/evaluations/run", json={"prompt_id": prompt.id, "dataset_ids": [d.id for d in datasets]})

    assert resp.status_code == 503
    run = db.query(EvaluationRun).one()
    assert run.status == RunStatus.FAILED and "could not enqueue" in run.error_message


# ── GET /evaluations/{id} ──────────────────────────────────────────────────────────

def test_run_response_exposes_counters_and_case_errors(client, db, seed, world):
    prompt, datasets = world
    rid = run_now(db, seed, prompt, datasets, FakeLLM({"q2": httpx.ConnectError("refused")}))
    # (one infrastructure failure on the default final attempt -> terminal PARTIAL)

    body = client.get(f"/evaluations/{rid}").json()
    assert body["status"] == "PARTIAL"
    assert (body["expected_cases"], body["completed_cases"], body["succeeded_cases"], body["failed_cases"]) == (3, 3, 2, 1)
    assert "1 of 3 cases failed" in body["error_message"]
    bad = next(r for r in body["results"] if r["status"] == "FAILED")
    assert bad["error_category"] == "INFRASTRUCTURE" and "ConnectError" in bad["error_message"]
    assert bad["actual_output"] is None and bad["similarity_score"] is None


# ── GET /evaluations/compare ───────────────────────────────────────────────────────

def test_compare_valid_runs_returns_ok_with_metrics(client, db, seed, world):
    prompt, datasets = world
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, datasets)
    body = client.get("/evaluations/compare", params={"baseline_run_id": b, "candidate_run_id": c}).json()

    assert body["comparison_status"] == "OK" and body["passed"] is True
    assert body["issues"] == [] and body["policy_version"]
    assert body["similarity"]["baseline_avg"] == 1.0
    assert len(body["per_dataset"]) == 3


def test_compare_detects_genuine_regression(client, db, seed, world):
    prompt, datasets = world
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, datasets, FakeLLM(default="completely different"))
    body = client.get("/evaluations/compare", params={"baseline_run_id": b, "candidate_run_id": c}).json()

    assert body["comparison_status"] == "OK"        # valid comparison...
    assert body["passed"] is False                   # ...genuinely worse
    assert body["similarity"]["regression_detected"] is True


def test_compare_with_a_partial_candidate_is_incomplete_and_blocks(client, db, seed, world):
    prompt, datasets = world
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, datasets, FakeLLM({"q1": httpx.ConnectError("x"), "q2": httpx.ConnectError("x"),
                                                     "q3": "a3"}))
    resp = client.get("/evaluations/compare", params={"baseline_run_id": b, "candidate_run_id": c})

    assert resp.status_code == 200
    body = resp.json()
    assert body["comparison_status"] == "INCOMPLETE" and body["passed"] is False
    assert body["similarity"] is None and body["per_dataset"] == []
    assert {"RUN_NOT_COMPLETED", "CASE_FAILED"} <= {i["code"] for i in body["issues"]}
    assert "No regressions" not in body["summary"]


def test_compare_runs_over_different_cases_is_incomparable(client, db, seed, world):
    prompt, datasets = world
    more = seed.datasets(2)  # two further cases with the same text, new ids
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, more)
    body = client.get("/evaluations/compare", params={"baseline_run_id": b, "candidate_run_id": c}).json()
    assert body["comparison_status"] == "INCOMPARABLE" and body["passed"] is False
    assert "COVERAGE_MISMATCH" in {i["code"] for i in body["issues"]}


def test_compare_status_codes_for_missing_and_unfinished_runs(client, db, seed, world):
    prompt, datasets = world
    done = run_now(db, seed, prompt, datasets)
    pending = seed.run(prompt, datasets).id
    assert client.get("/evaluations/compare", params={"baseline_run_id": done, "candidate_run_id": 999}).status_code == 404
    assert client.get("/evaluations/compare", params={"baseline_run_id": 999, "candidate_run_id": done}).status_code == 404
    unfinished = client.get("/evaluations/compare", params={"baseline_run_id": done, "candidate_run_id": pending})
    assert unfinished.status_code == 422 and "not finished" in unfinished.json()["detail"]


def test_legacy_style_completed_run_with_failed_rows_is_still_caught(client, db, seed, world):
    """A run wrongly marked COMPLETED but holding FAILED rows must not pass a comparison."""
    prompt, datasets = world
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, datasets)
    row = db.query(EvaluationResult).filter_by(run_id=c).first()
    row.status, row.similarity_score = ResultStatus.FAILED, None
    db.commit()

    body = client.get("/evaluations/compare", params={"baseline_run_id": b, "candidate_run_id": c}).json()
    assert body["comparison_status"] == "INCOMPLETE" and body["passed"] is False


# ── PATCH /evaluations/{id}/set-baseline ────────────────────────────────────────────

def test_only_completed_runs_can_become_baseline(client, db, seed, world):
    prompt, datasets = world
    partial = run_now(db, seed, prompt, datasets, FakeLLM({"q1": httpx.ConnectError("x")}))
    good = run_now(db, seed, prompt, datasets)

    rejected = client.patch(f"/evaluations/{partial}/set-baseline")
    assert rejected.status_code == 422 and "PARTIAL" in rejected.json()["detail"]
    accepted = client.patch(f"/evaluations/{good}/set-baseline")
    assert accepted.status_code == 200 and accepted.json()["is_baseline"] is True


# ── Dashboard ───────────────────────────────────────────────────────────────────────

def test_dashboard_shows_blocked_page_for_incomplete_comparison(client, db, seed, world):
    prompt, datasets = world
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, datasets, FakeLLM({"q2": httpx.ConnectError("refused")}))
    resp = client.get("/dashboard/compare", params={"baseline_run_id": b, "candidate_run_id": c})

    assert resp.status_code == 200
    html = resp.text
    assert "INCOMPLETE" in html and "BLOCKED" in html
    assert "ConnectError" in html and "INFRASTRUCTURE" in html      # failed case is listed
    assert "No regressions detected" not in html and "Chart(" not in html


def test_dashboard_valid_comparison_still_renders_the_verdict(client, db, seed, world):
    prompt, datasets = world
    b = run_now(db, seed, prompt, datasets)
    c = run_now(db, seed, prompt, datasets)
    html = client.get("/dashboard/compare", params={"baseline_run_id": b, "candidate_run_id": c}).text
    assert "No regressions detected" in html and "Chart(" in html


def test_dashboard_escapes_questions_and_model_output(client, db, seed):
    prompt = seed.prompt()
    evil = Dataset(input_query='<script>alert("q")</script></script>', expected_output="x")
    db.add(evil)
    db.commit()
    llm = FakeLLM({evil.input_query: '<img src=x onerror=alert(1)>'})
    b = run_now(db, seed, prompt, [evil], llm)
    c = run_now(db, seed, prompt, [evil], llm)

    html = client.get("/dashboard/compare", params={"baseline_run_id": b, "candidate_run_id": c}).text
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html and "&lt;img src=x" in html
    # The chart label sits inside a <script> block: it must not be able to close it.
    assert html.count("</script>") == 2          # Chart.js loader + the page's own block, nothing injected


def test_dashboard_status_codes(client, db, seed, world):
    prompt, datasets = world
    done = run_now(db, seed, prompt, datasets)
    pending = seed.run(prompt, datasets).id
    assert client.get("/dashboard/compare", params={"baseline_run_id": done, "candidate_run_id": 999}).status_code == 404
    assert client.get("/dashboard/compare", params={"baseline_run_id": done, "candidate_run_id": pending}).status_code == 422
