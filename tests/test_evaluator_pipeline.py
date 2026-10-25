"""
Evaluators wired through the executor, the comparator and the API, using the REAL
evaluator registry (the Phase 1 tests use a fake resolver instead).
"""
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from app.api.routes import evaluations as evaluation_routes
from app.core.comparator import ComparisonStatus, compare_runs, evaluate_comparison
from app.core.run_executor import execute_run
from app.db.models import (
    Dataset, ErrorCategory, EvaluationResult, EvaluationRun, EvaluationScore, ResultStatus, RunStatus,
    ScoreStatus,
)
from app.db.session import get_db
from app.main import app
from tests.conftest import FakeLLM
from tests.test_comparator import failed, ok, run, score


# ── helpers ──────────────────────────────────────────────────────────────────────────

def make_cases(db, specs):
    """specs: list of (query, expected, eval_spec)."""
    rows = [Dataset(input_query=q, expected_output=e, eval_spec=spec) for q, e, spec in specs]
    db.add_all(rows)
    db.commit()
    return rows


def execute(db, seed, prompt, cases, llm, **kw):
    r = seed.run(prompt, cases)
    outcome = execute_run(db, r.id, llm_fn=llm, **kw)
    db.expire_all()
    return r, outcome


def results_by_query(db, run_id):
    rows = db.query(EvaluationResult).filter_by(run_id=run_id).all()
    return {db.get(Dataset, r.dataset_id).input_query: r for r in rows}


CONTAINS_PARIS = [{"name": "contains", "config": {"all": ["paris"]}}]


# ── executor ─────────────────────────────────────────────────────────────────────────

def test_case_without_eval_spec_keeps_legacy_text_similarity(db, seed):
    cases = make_cases(db, [("q", "the cat sat", None)])
    r, outcome = execute(db, seed, seed.prompt(), cases, FakeLLM({"q": "the cat sat"}))
    row = results_by_query(db, r.id)["q"]

    assert outcome.status == RunStatus.COMPLETED
    (sc,) = row.scores
    assert (sc.evaluator_name, sc.evaluator_version, sc.status) == ("text_similarity", "1", ScoreStatus.OK)
    assert sc.gating is True and sc.config_hash
    assert row.similarity_score == sc.score == 1.0       # compatibility copy


def test_correctness_gates_and_similarity_becomes_advisory(db, seed):
    spec = CONTAINS_PARIS + [{"name": "text_similarity"}]
    cases = make_cases(db, [("capital?", "Paris", spec)])
    r, _ = execute(db, seed, seed.prompt(), cases, FakeLLM({"capital?": "It is Paris, of course."}))
    row = results_by_query(db, r.id)["capital?"]

    by = {s.evaluator_name: s for s in row.scores}
    assert by["contains"].gating is True and by["contains"].score == 1.0
    assert by["text_similarity"].gating is False
    assert row.similarity_score == by["text_similarity"].score


def test_correctness_only_case_has_no_similarity_copy_but_succeeds(db, seed):
    cases = make_cases(db, [("capital?", "Paris", CONTAINS_PARIS)])
    r, outcome = execute(db, seed, seed.prompt(), cases, FakeLLM({"capital?": "Lyon"}))
    row = results_by_query(db, r.id)["capital?"]

    assert outcome.status == RunStatus.COMPLETED          # a wrong answer is a score of 0, not a failure
    assert row.similarity_score is None
    assert row.scores[0].score == 0.0 and row.scores[0].status == ScoreStatus.OK


def test_invalid_eval_spec_fails_the_case_as_evaluator_error_and_is_not_retried(db, seed):
    cases = make_cases(db, [("q", "a", [{"name": "no_such_evaluator"}])])
    r, outcome = execute(db, seed, seed.prompt(), cases, FakeLLM(), final_attempt=False)
    row = results_by_query(db, r.id)["q"]

    assert outcome.status == RunStatus.FAILED and not outcome.needs_retry
    assert row.error_category == ErrorCategory.EVALUATOR_ERROR
    assert "no_such_evaluator" in row.error_message
    assert row.actual_output == "answer"                  # model output kept for debugging


def test_evaluator_error_is_recorded_explicitly_never_as_a_silent_gap(db, seed):
    # json_equal cannot score when the REFERENCE is not JSON: that is an evaluator error.
    spec = [{"name": "json_equal"}]
    cases = make_cases(db, [("q", "not json at all", spec)])
    r, outcome = execute(db, seed, seed.prompt(), cases, FakeLLM({"q": '{"a": 1}'}))
    row = results_by_query(db, r.id)["q"]

    assert outcome.status == RunStatus.FAILED
    assert row.status == ResultStatus.FAILED and row.error_category == ErrorCategory.EVALUATOR_ERROR
    (sc,) = row.scores
    assert sc.status == ScoreStatus.ERROR and sc.score is None and sc.error_message


def test_rerun_replaces_scores_of_a_failed_attempt(db, seed):
    cases = make_cases(db, [("q", "not json", [{"name": "json_equal"}])])
    r, first = execute(db, seed, seed.prompt(), cases, FakeLLM({"q": '{"a": 1}'}))
    assert first.status == RunStatus.FAILED

    cases[0].expected_output = '{"a": 1}'                  # fix the benchmark and resume the same run
    db.commit()
    second = execute_run(db, r.id, llm_fn=FakeLLM({"q": '{"a": 1}'}))
    db.expire_all()

    assert second.status == RunStatus.COMPLETED
    assert db.query(EvaluationScore).count() == 1          # replaced, not duplicated
    row = results_by_query(db, r.id)["q"]
    assert row.scores[0].status == ScoreStatus.OK and row.scores[0].score == 1.0


# ── comparator ───────────────────────────────────────────────────────────────────────

def corr(value, **kw):
    return score("contains", value, version="1", kind="correctness", config_hash="c0", **kw)


def case(ds, correctness, sim=0.9, *, sim_gating=False):
    return ok(ds, sim, scores=[corr(correctness), score(value=sim, gating=sim_gating)])


def compare(b, c, b_run=None, c_run=None):
    return evaluate_comparison(b_run or run(1, expected=len(b)), b, c_run or run(2, expected=len(c)), c)


def codes(out):
    return {i.code for i in out.issues}


def test_correctness_drop_blocks_even_when_similarity_is_unchanged():
    base = [case(i, 1.0) for i in range(1, 5)]
    cand = [case(1, 1.0), case(2, 1.0), case(3, 0.0), case(4, 0.0)]      # similarity identical (0.9)
    out = compare(base, cand)
    assert out.status == ComparisonStatus.OK and out.passed is False
    assert "contains dropped" in out.summary
    corr_delta = next(e for e in out.report.evaluators if e.name == "contains")
    assert corr_delta.gating and corr_delta.regression_detected
    assert {c.dataset_id for c in out.report.regressed_cases} == {3, 4}


def test_similarity_drop_next_to_a_correct_answer_is_advisory_only():
    base = [case(i, 1.0, sim=0.9) for i in range(1, 5)]
    cand = [case(i, 1.0, sim=0.3) for i in range(1, 5)]                   # reworded but still correct
    out = compare(base, cand)
    assert out.passed is True
    sim = next(e for e in out.report.evaluators if e.name == "text_similarity")
    assert sim.gating is False and sim.advisory_regression and not sim.regression_detected
    assert "advisory" in out.summary


def test_single_flip_within_threshold_passes_but_is_listed():
    base = [case(i, 1.0) for i in range(1, 21)]
    cand = [case(i, 0.0 if i == 7 else 1.0) for i in range(1, 21)]        # 1 of 20 = 5% < 10%
    out = compare(base, cand)
    assert out.passed is True
    assert [c.dataset_id for c in out.report.regressed_cases] == [7]
    assert "1 case(s) scored lower" in out.summary


def test_changed_evaluator_version_or_config_makes_runs_incomparable():
    base = [case(1, 1.0), case(2, 1.0)]
    for field, value in (("evaluator_version", "2"), ("config_hash", "different"), ("gating", False)):
        changed = NS(**{**vars(corr(1.0)), field: value})
        cand = [ok(1, 0.9, scores=[changed, score(gating=False)]), case(2, 1.0)]
        out = compare(base, cand)
        assert out.status == ComparisonStatus.INCOMPARABLE and out.passed is False
        assert "EVALUATOR_MISMATCH" in codes(out) and out.report is None


def test_different_evaluator_sets_are_incomparable():
    base = [ok(1, scores=[score(value=0.9)])]
    cand = [ok(1, scores=[corr(1.0), score(value=0.9, gating=False)])]
    out = compare(base, cand)
    assert out.status == ComparisonStatus.INCOMPARABLE and "EVALUATOR_MISMATCH" in codes(out)


@pytest.mark.parametrize("bad", [
    score(status="ERROR", value=None),
    score(status="ERROR", value=0.9),          # an ERROR row is never trusted, whatever number it carries
    score(value=float("nan")),
    score(value=1.2),
    score(value=None),
])
def test_missing_error_or_invalid_scores_are_never_averaged_over(bad):
    base = [case(1, 1.0), case(2, 1.0)]
    cand = [case(1, 1.0), ok(2, 0.9, scores=[bad])]
    out = compare(base, cand)
    assert out.status == ComparisonStatus.INCOMPLETE and out.passed is False
    assert "MISSING_SCORES" in codes(out) and out.report is None


def test_succeeded_case_without_any_score_row_is_incomplete():
    out = compare([case(1, 1.0)], [ok(1, 0.9, scores=[])])
    assert out.status == ComparisonStatus.INCOMPLETE and "MISSING_SCORES" in codes(out)


def test_case_with_no_gating_evaluator_blocks_instead_of_never_failing():
    nogate = lambda: ok(1, 0.9, scores=[score(gating=False)])   # noqa: E731
    out = compare([nogate()], [nogate()])
    assert out.passed is False and "NO_GATING_SCORE" in codes(out)


def test_compare_runs_refuses_mismatched_evaluators_directly():
    b = [case(1, 1.0)]
    c = [ok(1, 0.9, scores=[score(gating=False)])]
    with pytest.raises(ValueError):
        compare_runs(b, c, 1, 2)


def test_failed_cases_still_reported_before_any_score_logic():
    out = compare([case(1, 1.0), case(2, 1.0)], [case(1, 1.0), failed(2)])
    assert out.status == ComparisonStatus.INCOMPLETE and "CASE_FAILED" in codes(out)


# ── API / dashboard with real evaluators ──────────────────────────────────────────────

@pytest.fixture()
def client(session_factory, monkeypatch):
    def override():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(evaluation_routes.run_evaluation, "delay", lambda rid: None)
    yield TestClient(app)
    app.dependency_overrides.clear()


def two_runs(db, seed, good_answer, bad_answer):
    prompt = seed.prompt()
    cases = make_cases(db, [
        (f"capital {i}?", "Paris", CONTAINS_PARIS + [{"name": "text_similarity"}]) for i in range(4)
    ])
    base, _ = execute(db, seed, prompt, cases, FakeLLM(default=good_answer))
    cand, _ = execute(db, seed, prompt, cases, FakeLLM(default=bad_answer))
    return base, cand


def test_api_compare_blocks_a_wrong_but_similar_answer(client, db, seed):
    base, cand = two_runs(db, seed, "The capital is Paris.", "The capital is Lyon.")
    body = client.get("/evaluations/compare", params={"baseline_run_id": base.id, "candidate_run_id": cand.id}).json()

    assert body["comparison_status"] == "OK" and body["passed"] is False
    by = {e["name"]: e for e in body["evaluators"]}
    assert by["contains"]["gating"] and by["contains"]["regression_detected"]
    assert by["text_similarity"]["gating"] is False
    assert body["similarity"] is not None
    assert len(body["regressed_cases"]) == 4
    assert set(body["per_dataset"][0]["scores"]) == {"contains", "text_similarity"}


def test_api_compare_allows_a_correct_paraphrase(client, db, seed):
    base, cand = two_runs(db, seed, "Paris.", "Well, the answer to that question is definitely Paris!")
    body = client.get("/evaluations/compare", params={"baseline_run_id": base.id, "candidate_run_id": cand.id}).json()
    assert body["comparison_status"] == "OK" and body["passed"] is True


def test_api_run_response_exposes_scores(client, db, seed):
    base, _ = two_runs(db, seed, "Paris", "Paris")
    body = client.get(f"/evaluations/{base.id}").json()
    names = {s["evaluator_name"] for s in body["results"][0]["scores"]}
    assert names == {"contains", "text_similarity"}
    sc = next(s for s in body["results"][0]["scores"] if s["evaluator_name"] == "contains")
    assert sc["evaluator_version"] and sc["config_hash"] and sc["kind"] == "correctness" and sc["status"] == "OK"


def test_dashboard_renders_per_evaluator_cards_and_scores(client, db, seed):
    base, cand = two_runs(db, seed, "The capital is Paris.", "The capital is Lyon.")
    html = client.get("/dashboard/compare", params={"baseline_run_id": base.id, "candidate_run_id": cand.id}).text
    assert "contains" in html and "advisory" in html and "Regressions" in html


def test_correctness_and_similarity_use_their_own_thresholds(monkeypatch):
    from app.core.config import settings
    base = [case(i, 1.0, sim=0.8, sim_gating=True) for i in range(1, 5)]
    # similarity gates here (no correctness evaluator): 0.8 -> 0.6 is a 25% drop
    sim_only = lambda v: ok(1, v, scores=[score(value=v)])   # noqa: E731
    monkeypatch.setattr(settings, "regression_similarity_drop_threshold", 0.10)
    monkeypatch.setattr(settings, "regression_correctness_drop_threshold", 0.50)
    assert compare([sim_only(0.8)], [sim_only(0.6)]).passed is False

    # correctness: 4 -> 3 correct is a 25% drop, within the 50% correctness threshold
    cand = [case(1, 1.0), case(2, 1.0), case(3, 1.0), case(4, 0.0)]
    assert compare([case(i, 1.0) for i in range(1, 5)], cand).passed is True
    monkeypatch.setattr(settings, "regression_correctness_drop_threshold", 0.10)
    assert compare([case(i, 1.0) for i in range(1, 5)], cand).passed is False
