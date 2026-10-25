"""Versioned benchmark: file integrity, import API semantics, case identity."""
import copy
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.core.benchmark import CATEGORIES, case_content_hash
from app.core.evaluators import EvalInput, EvaluatorKind, resolve_evaluators, run_evaluator
from app.db.models import Dataset
from app.db.session import get_db
from app.main import app
from app.schemas.dataset import BenchmarkImport

BENCH_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "core-v1.yaml"


@pytest.fixture(scope="module")
def bench():
    return yaml.safe_load(BENCH_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def client(session_factory):
    def override():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── the shipped benchmark file ─────────────────────────────────────────────────────────

def test_core_benchmark_file_is_valid(bench):
    parsed = BenchmarkImport.model_validate(bench)       # keys unique, enums valid, evaluators instantiate
    assert (parsed.benchmark, parsed.version) == ("core", "1")
    assert 30 <= len(parsed.cases) <= 60


def test_core_benchmark_has_a_small_pr_suite_and_covers_the_planned_categories(bench):
    cases = bench["cases"]
    pr = [c for c in cases if "pr" in c["suites"]]
    assert 8 <= len(pr) <= 15 and len(pr) < len(cases) / 2
    assert all("full" in c["suites"] for c in cases)
    assert {c["category"] for c in cases} == CATEGORIES
    assert {"structured_output", "classification", "qa"} <= {c["task_type"] for c in cases}
    # the fast suite must still exercise every kind of grading
    pr_evaluators = {e["name"] for c in pr for e in c["eval_spec"]}
    assert {"contains", "exact_match", "json_schema", "json_equal"} <= pr_evaluators


def test_every_case_has_an_objective_gating_check(bench):
    for c in bench["cases"]:
        resolved = resolve_evaluators(c["eval_spec"])
        gating = [r for r in resolved if r.gating]
        assert gating and all(r.evaluator.kind == EvaluatorKind.CORRECTNESS for r in gating), c["case_key"]


def test_each_reference_answer_passes_its_own_checks_and_an_empty_answer_fails(bench):
    """Guards against typos in terms/schemas: the benchmark must accept its reference and reject nothing-at-all."""
    for c in bench["cases"]:
        for item in resolve_evaluators(c["eval_spec"]):
            if item.evaluator.kind != EvaluatorKind.CORRECTNESS:
                continue
            ref = run_evaluator(item.evaluator, EvalInput(c["expected_output"], c["expected_output"], c))
            assert ref.score == 1.0, (c["case_key"], item.evaluator.name, ref)
            empty = run_evaluator(item.evaluator, EvalInput("", c["expected_output"], c))
            assert empty.score == 0.0, (c["case_key"], item.evaluator.name, "empty answer must not pass")


def test_wrong_but_similar_answers_fail_the_checks_where_similarity_would_pass(bench):
    wrong = {
        "sim-capital-city-001": "The capital of France is Lyon.",
        "sim-minutes-001": "There are 50 minutes in an hour.",
        "sim-whale-001": "A whale is a fish.",
        "sim-negation-001": "Yes, the Earth is flat; it is not roughly spherical.",
    }
    by_key = {c["case_key"]: c for c in bench["cases"]}
    for key, answer in wrong.items():
        c = by_key[key]
        scores = {r.evaluator.name: run_evaluator(r.evaluator, EvalInput(answer, c["expected_output"], c)).score
                  for r in resolve_evaluators(c["eval_spec"])}
        assert scores["contains"] == 0.0, key
        assert scores["text_similarity"] > 0.6, (key, scores)       # similarity alone would NOT have blocked it


# ── import API ────────────────────────────────────────────────────────────────────

def small(version="1", **override):
    case = {
        "case_key": "t-001", "input_query": "What is the capital of France?", "expected_output": "Paris",
        "task_type": "qa", "category": "normal", "difficulty": "easy", "suites": ["pr", "full"],
        "eval_spec": [{"name": "contains", "config": {"all": ["paris"]}}],
    }
    case.update(override)
    return {"benchmark": "t", "version": version, "cases": [case]}


def test_import_creates_then_is_idempotent(client, db):
    first = client.post("/datasets/import", json=small()).json()
    again = client.post("/datasets/import", json=small()).json()

    assert (first["created"], first["unchanged"]) == (1, 0)
    assert (again["created"], again["unchanged"]) == (0, 1)
    assert first["cases"][0]["id"] == again["cases"][0]["id"]
    assert db.query(Dataset).count() == 1
    row = first["cases"][0]
    assert row["case_key"] == "t-001" and row["suites"] == ["pr", "full"] and len(row["content_hash"]) == 64


def test_changed_case_under_same_version_is_refused_and_nothing_is_written(client, db):
    client.post("/datasets/import", json=small())
    payload = small(expected_output="Lutetia")
    payload["cases"].append({**payload["cases"][0], "case_key": "t-002"})      # a new case alongside the change
    resp = client.post("/datasets/import", json=payload)

    assert resp.status_code == 409 and "t-001" in resp.json()["detail"] and "new version" in resp.json()["detail"]
    assert db.query(Dataset).count() == 1                                      # t-002 was not written either
    assert db.query(Dataset).one().expected_output == "Paris"


def test_changing_the_evaluator_or_suite_membership_also_counts_as_a_change(client):
    client.post("/datasets/import", json=small())
    assert client.post("/datasets/import", json=small(eval_spec=[{"name": "contains", "config": {"all": ["lyon"]}}])).status_code == 409
    assert client.post("/datasets/import", json=small(suites=["full"])).status_code == 409


def test_new_version_is_a_separate_set_of_cases(client, db):
    a = client.post("/datasets/import", json=small("1")).json()["cases"][0]
    b = client.post("/datasets/import", json=small("2", expected_output="Lutetia")).json()["cases"][0]
    assert a["id"] != b["id"] and a["content_hash"] != b["content_hash"]
    assert db.query(Dataset).count() == 2


@pytest.mark.parametrize("mutate", [
    lambda p: p["cases"][0].update(case_key="Bad Key"),
    lambda p: p["cases"][0].update(category="nonsense"),
    lambda p: p["cases"][0].update(suites=["pr"]),                       # must be in 'full'
    lambda p: p["cases"][0].update(eval_spec=[]),                         # no implicit grading
    lambda p: p["cases"][0].update(eval_spec=[{"name": "no_such_evaluator"}]),
    lambda p: p["cases"][0].update(eval_spec=[{"name": "contains", "config": {}}]),
    lambda p: p["cases"].append(copy.deepcopy(p["cases"][0])),            # duplicate case_key
    lambda p: p.update(cases=[]),
])
def test_invalid_benchmarks_are_rejected_with_422_and_write_nothing(client, db, mutate):
    payload = small()
    mutate(payload)
    assert client.post("/datasets/import", json=payload).status_code == 422
    assert db.query(Dataset).count() == 0


def test_list_filters_by_benchmark_suite_and_category(client):
    payload = small()
    payload["cases"].append({**payload["cases"][0], "case_key": "t-002", "suites": ["full"], "category": "edge"})
    client.post("/datasets/import", json=payload)
    client.post("/datasets/", json={"input_query": "legacy", "expected_output": "x"})

    def keys(**params):
        return sorted(d["case_key"] or "legacy" for d in client.get("/datasets/", params=params).json())

    assert keys() == ["legacy", "t-001", "t-002"]
    assert keys(benchmark="t") == ["t-001", "t-002"]
    assert keys(benchmark="t", suite="pr") == ["t-001"]
    assert keys(category="edge") == ["t-002"]


def test_legacy_endpoints_still_work_and_validate_optional_eval_spec(client):
    ok = client.post("/datasets/", json={"input_query": "q", "expected_output": "a"})
    assert ok.status_code == 201 and ok.json()["eval_spec"] is None
    graded = client.post("/datasets/bulk", json=[
        {"input_query": "q", "expected_output": "a", "eval_spec": [{"name": "exact_match"}]}])
    assert graded.status_code == 201 and graded.json()[0]["eval_spec"] == [{"name": "exact_match"}]
    assert client.post("/datasets/", json={"input_query": "q", "expected_output": "a",
                                           "eval_spec": [{"name": "bogus"}]}).status_code == 422


def test_content_hash_depends_on_every_defining_field():
    base = dict(input_query="q", expected_output="a", eval_spec=[{"name": "exact_match"}],
                task_type="qa", category="normal", difficulty="easy", suites=["pr", "full"])
    h = case_content_hash(**base)
    assert h == case_content_hash(**{**base, "suites": ["full", "pr"]})          # order-insensitive
    for field, value in [("input_query", "q2"), ("expected_output", "a2"), ("eval_spec", [{"name": "contains"}]),
                         ("task_type", "extraction"), ("category", "edge"), ("difficulty", "hard"), ("suites", ["full"])]:
        assert case_content_hash(**{**base, field: value}) != h, field
