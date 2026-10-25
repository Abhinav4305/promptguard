"""
Gate effectiveness on the controlled, SIMULATED model answers (see app/core/gate_measurement.py).
These are regression tests for the gate's behaviour, not measurements of a real model.
"""
from pathlib import Path

import pytest
import yaml

from app.core.evaluators import EvalInput, EvaluatorKind, resolve_evaluators, run_evaluator
from app.core.gate_measurement import format_markdown, measure

ROOT = Path(__file__).resolve().parent.parent
PAIRS = {
    "v1": (ROOT / "benchmarks" / "core-v1.yaml", ROOT / "benchmarks" / "simulations" / "core-v1.sim.yaml"),
    "v2": (ROOT / "benchmarks" / "core-v2.yaml", ROOT / "benchmarks" / "simulations" / "core-v2.sim.yaml"),
}
BENCH, SIM = PAIRS["v2"]                       # the benchmark CI uses; the measurement runs on it


@pytest.fixture(scope="module", params=sorted(PAIRS))
def cases(request):
    return {c["case_key"]: c for c in yaml.safe_load(PAIRS[request.param][0].read_text(encoding="utf-8"))["cases"]}


@pytest.fixture(scope="module")
def sim(request, cases):
    # the simulation file that goes with whichever benchmark version `cases` loaded
    for name, (bench_path, sim_path) in PAIRS.items():
        keys = {c["case_key"] for c in yaml.safe_load(bench_path.read_text(encoding="utf-8"))["cases"]}
        if keys == set(cases):
            return yaml.safe_load(sim_path.read_text(encoding="utf-8"))
    raise AssertionError("no simulation file matches the benchmark")


def correctness_scores(case, answer):
    out = {}
    for item in resolve_evaluators(case["eval_spec"]):
        if item.evaluator.kind == EvaluatorKind.CORRECTNESS:
            out[item.evaluator.name] = run_evaluator(
                item.evaluator, EvalInput(answer, case["expected_output"], case)).score
    return out


def test_simulation_covers_every_benchmark_case_and_nothing_else(cases, sim):
    assert set(sim) == set(cases)


def test_simulated_labels_agree_with_the_benchmarks_own_checks(cases, sim):
    """good/paraphrase must be fully correct; wrong_* must fail at least one gating check."""
    for key, case in cases.items():
        for kind in ("good", "paraphrase"):
            scores = correctness_scores(case, sim[key][kind])
            assert scores and all(v == 1.0 for v in scores.values()), (key, kind, scores)
        for kind in ("wrong_close", "wrong_far"):
            scores = correctness_scores(case, sim[key][kind])
            assert any(v < 1.0 for v in scores.values()), (key, kind, "wrong answer passes every check", scores)


def test_wrong_close_answers_really_are_close_in_wording_for_most_cases(cases, sim):
    """If wrong_close were far from the reference the old-vs-new comparison would be meaningless."""
    from app.core.evaluators import create_evaluator
    ts = create_evaluator("text_similarity")
    close = sum(
        ts.evaluate(EvalInput(sim[k]["wrong_close"], c["expected_output"], c)).score >= 0.6
        for k, c in cases.items() if c["task_type"] in ("qa", "instruction_following")
    )
    total = sum(c["task_type"] in ("qa", "instruction_following") for c in cases.values())
    assert close / total >= 0.6, (close, total)


@pytest.fixture(scope="module")
def report():
    return measure(str(BENCH), str(SIM), seeds=2)


def test_report_is_labelled_as_simulated(report):
    assert report["simulated"] is True and "NOT measurements" in report["disclaimer"]
    assert "SIMULATED" in format_markdown(report)


@pytest.mark.parametrize("suite", ["pr", "full"])
def test_new_gate_blocks_regressions_and_allows_paraphrases(report, suite):
    h = report["suites"][suite]["headline"]
    det, fb = h["detection_of_regressions"], h["false_blocks_on_acceptable_changes"]
    assert det["new"]["count"] == det["new"]["of"] > 0
    assert fb["new"]["count"] == 0 and fb["new"]["of"] > 0


@pytest.mark.parametrize("suite", ["pr", "full"])
def test_old_similarity_gate_has_the_blind_spots_the_new_gate_closes(report, suite):
    s = report["suites"][suite]
    h = s["headline"]
    assert h["false_blocks_on_acceptable_changes"]["old_similarity_only"]["rate"] > 0.5   # paraphrases look like regressions
    assert h["detection_of_regressions"]["old_similarity_only"]["rate"] < 1.0
    by = s["by_scenario_type"]
    wc = by["wrong_close@25%"]
    assert wc["new_blocked"] == wc["of"] and wc["old_blocked"] < wc["of"]      # similarity misses some close-but-wrong answers
    assert by["format_break"]["old_blocked"] == 0 and by["format_break"]["new_blocked"] == 1


def test_every_simulated_comparison_was_a_valid_comparison(report):
    for s in report["suites"].values():
        assert all(r["new_status"] == "OK" and r["old_status"] == "OK" for r in s["scenarios"])
