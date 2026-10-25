"""
The Celery wrapper: retry decisions around the executor. Uses Celery's eager `apply()`,
which re-applies on `self.retry` with an incremented retry counter and needs no broker.
"""
import httpx
import pytest

from app.core import run_executor
from app.db.models import EvaluationRun, ResultStatus, RunStatus
from app.tasks import evaluation_tasks
from tests.conftest import FakeLLM, fake_resolver, make_resolver


@pytest.fixture()
def wire(monkeypatch, session_factory):
    """Point the task at the test database and a fake LLM."""

    def install(llm):
        monkeypatch.setattr(evaluation_tasks, "SessionLocal", session_factory)
        real = run_executor.execute_run
        monkeypatch.setattr(
            evaluation_tasks, "execute_run",
            lambda db, run_id, final_attempt: real(
                db, run_id, llm_fn=llm, resolver=fake_resolver, final_attempt=final_attempt
            ),
        )

    return install


@pytest.fixture()
def run_row(seed):
    return seed.run(seed.prompt(), seed.datasets(3))


def test_transient_infrastructure_failure_is_retried_to_completion(db, run_row, wire):
    llm = FakeLLM({"q1": "a1", "q2": [httpx.ConnectError("refused"), "a2"], "q3": "a3"})
    wire(llm)

    result = evaluation_tasks.run_evaluation.apply(args=[run_row.id])

    payload = result.get()
    assert payload["status"] == "COMPLETED"
    assert (payload["expected_cases"], payload["succeeded_cases"], payload["failed_cases"]) == (3, 3, 0)
    assert payload["results_count"] == 3                                   # legacy key preserved
    assert llm.count("q1") == 1 and llm.count("q2") == 2 and llm.count("q3") == 1
    db.expire_all()
    assert db.get(EvaluationRun, run_row.id).status == RunStatus.COMPLETED


def test_persistent_infrastructure_failure_ends_partial_after_all_retries(db, run_row, wire):
    llm = FakeLLM({"q3": httpx.ConnectError("refused")})
    wire(llm)

    result = evaluation_tasks.run_evaluation.apply(args=[run_row.id])

    assert result.get()["status"] == "PARTIAL"
    assert llm.count("q3") == 3                    # initial attempt + max_retries=2
    assert llm.count("q1") == 1 and llm.count("q2") == 1
    db.expire_all()
    run = db.get(EvaluationRun, run_row.id)
    assert run.status == RunStatus.PARTIAL and (run.succeeded_cases, run.failed_cases) == (2, 1)


def test_run_never_shows_failed_while_retries_remain(db, run_row, wire, monkeypatch):
    seen = []
    llm = FakeLLM({"q2": [httpx.ConnectError("refused"), "a2"]})
    wire(llm)
    real = evaluation_tasks.execute_run

    def spy(db_, run_id, final_attempt):
        out = real(db_, run_id, final_attempt=final_attempt)
        seen.append(out.status)
        return out

    monkeypatch.setattr(evaluation_tasks, "execute_run", spy)
    evaluation_tasks.run_evaluation.apply(args=[run_row.id])
    assert seen == [RunStatus.RUNNING, RunStatus.COMPLETED]


def test_unexpected_errors_are_retried_then_the_run_is_closed_not_left_running(db, run_row, wire, monkeypatch):
    wire(FakeLLM())
    attempts = []

    def explode(db_, run_id, final_attempt):
        attempts.append(final_attempt)
        raise RuntimeError("database went away")

    monkeypatch.setattr(evaluation_tasks, "execute_run", explode)
    result = evaluation_tasks.run_evaluation.apply(args=[run_row.id])

    assert attempts == [False, False, True]
    assert result.failed()
    db.expire_all()
    run = db.get(EvaluationRun, run_row.id)
    assert run.status == RunStatus.FAILED
    assert "unexpected error: RuntimeError: database went away" in run.error_message


def test_missing_run_returns_error_dict(wire):
    wire(FakeLLM())
    assert evaluation_tasks.run_evaluation.apply(args=[4242]).get() == {"error": "run not found"}
