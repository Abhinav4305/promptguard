"""
Failure-aware run lifecycle (audit findings 1 and 2).

The point of these tests: a run may only be COMPLETED if every case succeeded, a
failed case is always visible with a category, and retries neither double count nor
redo finished work.
"""
import json

import httpx
import pytest

from app.core.run_executor import classify_llm_error, execute_run, mark_run_failed
from app.db.models import ErrorCategory, EvaluationResult, EvaluationRun, ResultStatus, RunStatus
from tests.conftest import FakeLLM, fake_resolver, make_resolver


def _http_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://ollama/api/chat")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, request=req))


def _rows(db, run_id):
    db.expire_all()
    return {
        r.dataset.input_query: r
        for r in db.query(EvaluationResult).filter_by(run_id=run_id).all()
    }


@pytest.fixture()
def setup(db, seed):
    prompt = seed.prompt()
    datasets = seed.datasets(3)
    run = seed.run(prompt, datasets)
    return run


# ── Happy path ────────────────────────────────────────────────────────────────

def test_all_cases_succeed_marks_run_completed(db, setup):
    llm = FakeLLM({"q1": "a1", "q2": "a2", "q3": "a3"})
    outcome = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver)

    assert outcome.status == RunStatus.COMPLETED
    assert (outcome.expected_cases, outcome.succeeded_cases, outcome.failed_cases) == (3, 3, 0)
    run = db.get(EvaluationRun, setup.id)
    db.refresh(run)
    assert run.status == RunStatus.COMPLETED
    assert run.completed_cases == 3
    assert run.started_at is not None and run.completed_at is not None
    assert run.error_message is None
    for row in _rows(db, run.id).values():
        assert row.status == ResultStatus.SUCCEEDED
        assert row.similarity_score == 1.0
        assert row.error_category is None and row.error_message is None
        assert row.attempts == 1


# ── The original bug: failures hidden behind COMPLETED ──────────────────────────

def test_one_failed_case_makes_run_partial_not_completed(db, setup):
    llm = FakeLLM({"q2": _http_error(404)})  # gateway rejects: model not found
    outcome = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver)

    assert outcome.status == RunStatus.PARTIAL
    assert (outcome.succeeded_cases, outcome.failed_cases) == (2, 1)
    run = db.get(EvaluationRun, setup.id)
    assert run.status == RunStatus.PARTIAL
    assert "1 of 3 cases failed" in run.error_message and "MODEL_ERROR=1" in run.error_message

    rows = _rows(db, run.id)
    failed = rows["q2"]
    assert failed.status == ResultStatus.FAILED
    assert failed.error_category == ErrorCategory.MODEL_ERROR
    assert "HTTPStatusError" in failed.error_message
    assert failed.actual_output is None          # an error is NOT stored as model output
    assert failed.similarity_score is None
    # Partial results are preserved for debugging.
    assert rows["q1"].status == rows["q3"].status == ResultStatus.SUCCEEDED
    assert rows["q1"].similarity_score is not None


def test_all_cases_failing_marks_run_failed(db, setup):
    llm = FakeLLM(default=httpx.ConnectError("refused"))
    outcome = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=True)

    assert outcome.status == RunStatus.FAILED
    assert (outcome.succeeded_cases, outcome.failed_cases) == (0, 3)
    assert not outcome.needs_retry
    assert db.get(EvaluationRun, setup.id).completed_at is not None


# ── Retry behaviour ─────────────────────────────────────────────────────────────

def test_infrastructure_failure_keeps_run_running_and_requests_retry(db, setup):
    llm = FakeLLM({"q2": httpx.ReadTimeout("slow")})
    outcome = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=False)

    assert outcome.needs_retry
    assert outcome.status == RunStatus.RUNNING        # never FAILED/COMPLETED mid-retry
    run = db.get(EvaluationRun, setup.id)
    assert run.status == RunStatus.RUNNING and run.completed_at is None
    assert run.error_message.startswith("retrying:")
    assert _rows(db, run.id)["q2"].error_category == ErrorCategory.INFRASTRUCTURE


def test_retry_reruns_only_unfinished_cases_and_does_not_double_count(db, setup):
    llm = FakeLLM({"q1": "a1", "q2": [httpx.ConnectError("refused"), "a2"], "q3": "a3"})

    first = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=False)
    started_at = db.get(EvaluationRun, setup.id).started_at
    assert first.needs_retry and (first.succeeded_cases, first.failed_cases) == (2, 1)

    second = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=True)
    assert second.status == RunStatus.COMPLETED
    assert (second.expected_cases, second.succeeded_cases, second.failed_cases) == (3, 3, 0)

    # Cases that already succeeded were never called again.
    assert llm.count("q1") == 1 and llm.count("q3") == 1 and llm.count("q2") == 2
    rows = _rows(db, setup.id)
    assert (rows["q1"].attempts, rows["q2"].attempts, rows["q3"].attempts) == (1, 2, 1)
    # The failure from attempt 1 left no trace on the recovered case.
    assert rows["q2"].error_category is None and rows["q2"].error_message is None
    # Still exactly one row per case, and the start time is that of the first attempt.
    assert db.query(EvaluationResult).filter_by(run_id=setup.id).count() == 3
    run = db.get(EvaluationRun, setup.id)
    assert run.started_at == started_at and run.error_message is None


def test_persistent_infrastructure_failure_ends_partial_after_final_attempt(db, setup):
    llm = FakeLLM({"q3": httpx.ConnectError("refused")})
    execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=False)
    outcome = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=True)

    assert outcome.status == RunStatus.PARTIAL and not outcome.needs_retry
    run = db.get(EvaluationRun, setup.id)
    assert "INFRASTRUCTURE=1" in run.error_message
    assert _rows(db, run.id)["q3"].attempts == 2


def test_model_errors_are_not_retried(db, setup):
    llm = FakeLLM({"q1": _http_error(400)})
    outcome = execute_run(db, setup.id, llm_fn=llm, resolver=fake_resolver, final_attempt=False)
    assert not outcome.needs_retry
    assert outcome.status == RunStatus.PARTIAL
    assert llm.count("q1") == 1


def test_stale_data_from_a_crashed_attempt_is_cleared_on_rerun(db, setup):
    # Simulate a worker that died after writing a score but before finishing the case.
    row = db.query(EvaluationResult).filter_by(run_id=setup.id).first()
    row.status = ResultStatus.RUNNING
    row.similarity_score = 0.99
    row.actual_output = "stale"
    db.commit()
    query = row.dataset.input_query

    execute_run(db, setup.id, llm_fn=FakeLLM({query: _http_error(404)}), resolver=fake_resolver)

    row = _rows(db, setup.id)[query]
    assert row.status == ResultStatus.FAILED
    assert row.similarity_score is None and row.actual_output is None


# ── Evaluator failures are distinct from model failures ─────────────────────────────

def test_scoring_error_is_an_evaluator_error_and_keeps_model_output(db, setup):
    def broken_score(prediction, reference):
        raise RuntimeError("metric blew up")

    outcome = execute_run(db, setup.id, llm_fn=FakeLLM(), resolver=make_resolver(broken_score))
    assert outcome.status == RunStatus.FAILED
    row = _rows(db, setup.id)["q1"]
    assert row.status == ResultStatus.FAILED
    assert row.error_category == ErrorCategory.EVALUATOR_ERROR
    assert row.actual_output == "answer"          # what the model said is kept for debugging
    assert row.latency_seconds == 0.5
    assert row.similarity_score is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.1, None, "0.9"])
def test_invalid_scores_are_rejected_not_stored(db, setup, bad):
    outcome = execute_run(db, setup.id, llm_fn=FakeLLM(), resolver=make_resolver(lambda prediction, reference: bad))
    assert outcome.status == RunStatus.FAILED
    row = _rows(db, setup.id)["q1"]
    assert row.error_category == ErrorCategory.EVALUATOR_ERROR
    assert row.similarity_score is None


# ── A run that cannot start never looks successful ───────────────────────────────────

def test_missing_prompt_fails_the_run(db, seed):
    prompt = seed.prompt()
    run = seed.run(prompt, seed.datasets(2))
    # SQLite does not enforce the FK here, so pointing the run at a missing prompt id is enough.
    db.query(EvaluationRun).filter_by(id=run.id).update({"prompt_id": 9999})
    db.commit()

    outcome = execute_run(db, run.id, llm_fn=FakeLLM(), resolver=fake_resolver)
    assert outcome.status == RunStatus.FAILED
    assert "prompt 9999 not found" in db.get(EvaluationRun, run.id).error_message


def test_run_with_no_cases_is_failed_not_completed(db, seed):
    run = seed.run(seed.prompt(), [])
    outcome = execute_run(db, run.id, llm_fn=FakeLLM(), resolver=fake_resolver)
    assert outcome.status == RunStatus.FAILED
    assert "no test cases" in db.get(EvaluationRun, run.id).error_message


def test_unknown_run_returns_none(db):
    assert execute_run(db, 12345, llm_fn=FakeLLM(), resolver=fake_resolver) is None


def test_mark_run_failed_after_unexpected_error_keeps_partial_results(db, setup):
    execute_run(db, setup.id, llm_fn=FakeLLM({"q3": httpx.ConnectError("x")}), resolver=fake_resolver,
                final_attempt=False)
    mark_run_failed(db, setup.id, "unexpected error: OperationalError")

    run = db.get(EvaluationRun, setup.id)
    db.refresh(run)
    assert run.status == RunStatus.PARTIAL            # 2 succeeded rows are kept, run is not COMPLETED
    assert (run.succeeded_cases, run.failed_cases) == (2, 1)
    assert run.error_message.startswith("unexpected error")


# ── Error classification ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "exc, expected",
    [
        (httpx.ConnectError("refused"), ErrorCategory.INFRASTRUCTURE),
        (httpx.ReadTimeout("slow"), ErrorCategory.INFRASTRUCTURE),
        (_http_error(503), ErrorCategory.INFRASTRUCTURE),
        (_http_error(429), ErrorCategory.INFRASTRUCTURE),
        (_http_error(404), ErrorCategory.MODEL_ERROR),
        (_http_error(400), ErrorCategory.MODEL_ERROR),
        (json.JSONDecodeError("bad", "{", 0), ErrorCategory.MODEL_ERROR),
        (KeyError("message"), ErrorCategory.MODEL_ERROR),
        (RuntimeError("???"), ErrorCategory.UNKNOWN),
    ],
)
def test_classify_llm_error(exc, expected):
    assert classify_llm_error(exc) == expected
