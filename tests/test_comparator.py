"""
Baseline/candidate comparability (audit findings 3 and 4).

Scenarios A and B reproduce the audit's findings against the OLD comparator, where
a candidate missing 2 of 3 cases, or evaluated on entirely different cases, still
returned "No regressions detected."
"""
from types import SimpleNamespace as NS

import pytest

from app.core.comparator import (
    POLICY_VERSION, ComparisonStatus, check_comparability, compare_runs, evaluate_comparison,
)
from app.db.models import ErrorCategory, ResultStatus, RunStatus


def run(rid, status=RunStatus.COMPLETED, expected=3, error_message=None):
    return NS(id=rid, status=status, expected_cases=expected, error_message=error_message)


def ok(ds, sim=0.9, lat=1.0, cost=0.0):
    return NS(dataset_id=ds, status=ResultStatus.SUCCEEDED, error_category=None,
              similarity_score=sim, latency_seconds=lat, token_cost=cost)


def failed(ds, category=ErrorCategory.INFRASTRUCTURE):
    return NS(dataset_id=ds, status=ResultStatus.FAILED, error_category=category,
              similarity_score=None, latency_seconds=None, token_cost=None)


def pending(ds, status=ResultStatus.PENDING):
    return NS(dataset_id=ds, status=status, error_category=None,
              similarity_score=None, latency_seconds=None, token_cost=None)


BASELINE = [ok(1), ok(2), ok(3)]


def compare(b_results, c_results, b_run=None, c_run=None):
    return evaluate_comparison(b_run or run(1), b_results, c_run or run(2), c_results)


def codes(outcome):
    return {i.code for i in outcome.issues}


# ── Valid comparisons still work ──────────────────────────────────────────────────

def test_identical_runs_pass():
    out = compare(BASELINE, [ok(1), ok(2), ok(3)])
    assert out.status == ComparisonStatus.OK and out.passed
    assert out.report is not None and out.issues == []
    assert out.policy_version == POLICY_VERSION
    assert len(out.report.per_dataset) == 3


def test_real_quality_regression_is_detected():
    out = compare(BASELINE, [ok(1, 0.3), ok(2, 0.3), ok(3, 0.3)])
    assert out.status == ComparisonStatus.OK          # comparable...
    assert not out.passed                              # ...and genuinely worse
    assert out.report.similarity.regression_detected
    assert "similarity dropped" in out.summary


def test_acceptable_change_is_not_blocked():
    # 0.90 -> 0.855 is a 5% relative drop, inside the 10% threshold.
    out = compare(BASELINE, [ok(1, 0.855), ok(2, 0.855), ok(3, 0.855)])
    assert out.status == ComparisonStatus.OK and out.passed


def test_latency_regression_is_detected():
    out = compare(BASELINE, [ok(1, lat=5.0), ok(2, lat=5.0), ok(3, lat=5.0)])
    assert out.status == ComparisonStatus.OK and not out.passed
    assert out.report.latency.regression_detected and not out.report.similarity.regression_detected


# ── Audit scenario A: candidate lost cases to failures ───────────────────────────────

def test_candidate_with_failed_cases_is_incomplete_not_passed():
    out = compare(BASELINE, [failed(1), failed(2), ok(3)])
    assert out.status == ComparisonStatus.INCOMPLETE
    assert not out.passed
    assert out.report is None                          # no aggregate score is reported
    (issue,) = [i for i in out.issues if i.code == "CASE_FAILED"]
    assert issue.run == "candidate" and issue.dataset_ids == [1, 2]
    assert "2 of 3 cases failed" in issue.message
    assert "No regressions" not in out.summary


def test_infrastructure_failure_message_says_it_is_not_a_regression():
    out = compare(BASELINE, [failed(1, ErrorCategory.INFRASTRUCTURE), ok(2), ok(3)])
    (issue,) = [i for i in out.issues if i.code == "CASE_FAILED"]
    assert "not quality regressions" in issue.message and "INFRASTRUCTURE=1" in issue.message


def test_model_error_message_has_no_rerun_hint():
    out = compare(BASELINE, [failed(1, ErrorCategory.MODEL_ERROR), ok(2), ok(3)])
    (issue,) = [i for i in out.issues if i.code == "CASE_FAILED"]
    assert "MODEL_ERROR=1" in issue.message and "re-run" not in issue.message


def test_failed_baseline_cases_also_block():
    out = compare([ok(1), failed(2, ErrorCategory.EVALUATOR_ERROR), ok(3)], BASELINE)
    assert out.status == ComparisonStatus.INCOMPLETE and not out.passed
    assert [i.run for i in out.issues if i.code == "CASE_FAILED"] == ["baseline"]


def test_failures_on_both_sides_are_each_reported():
    out = compare([failed(1), ok(2), ok(3)], [ok(1), failed(2), ok(3)])
    assert {(i.run, tuple(i.dataset_ids)) for i in out.issues if i.code == "CASE_FAILED"} == {
        ("baseline", (1,)), ("candidate", (2,)),
    }


# ── Audit scenario B: runs cover different cases ───────────────────────────────────────

def test_disjoint_cases_are_incomparable():
    out = compare(BASELINE, [ok(10), ok(11), ok(12)])
    assert out.status == ComparisonStatus.INCOMPARABLE and not out.passed
    (issue,) = [i for i in out.issues if i.code == "COVERAGE_MISMATCH"]
    assert issue.dataset_ids == [1, 2, 3, 10, 11, 12]
    assert "only in baseline: 1, 2, 3" in issue.message and "only in candidate: 10, 11, 12" in issue.message


def test_candidate_missing_rows_entirely_is_incomparable():
    # Only one row came back for a run that expected three.
    out = compare(BASELINE, [ok(3)], c_run=run(2, expected=3))
    assert out.status == ComparisonStatus.INCOMPARABLE and not out.passed
    assert {"COVERAGE_MISMATCH", "RESULT_COUNT_MISMATCH"} <= codes(out)


def test_extra_case_in_candidate_is_incomparable():
    out = compare(BASELINE, [ok(1), ok(2), ok(3), ok(4)], c_run=run(2, expected=4))
    assert out.status == ComparisonStatus.INCOMPARABLE
    (issue,) = [i for i in out.issues if i.code == "COVERAGE_MISMATCH"]
    assert issue.dataset_ids == [4]


def test_duplicate_case_results_are_incomparable():
    out = compare(BASELINE, [ok(1), ok(1), ok(2), ok(3)], c_run=run(2, expected=3))
    assert out.status == ComparisonStatus.INCOMPARABLE and not out.passed
    (issue,) = [i for i in out.issues if i.code == "DUPLICATE_CASES"]
    assert issue.dataset_ids == [1]


# ── Run-level and result-level incompleteness ──────────────────────────────────────────

@pytest.mark.parametrize("status", [RunStatus.PARTIAL, RunStatus.FAILED])
def test_non_completed_run_is_incomplete_and_explains_why(status):
    out = compare(BASELINE, BASELINE, c_run=run(2, status=status, error_message="1 of 3 cases failed"))
    assert out.status == ComparisonStatus.INCOMPLETE and not out.passed
    (issue,) = [i for i in out.issues if i.code == "RUN_NOT_COMPLETED"]
    assert status.value in issue.message and "1 of 3 cases failed" in issue.message


def test_cases_that_never_finished_are_reported():
    out = compare(BASELINE, [ok(1), pending(2), pending(3, ResultStatus.RUNNING)])
    assert out.status == ComparisonStatus.INCOMPLETE
    (issue,) = [i for i in out.issues if i.code == "CASE_NOT_FINISHED"]
    assert issue.dataset_ids == [2, 3]


def test_succeeded_case_without_a_score_is_not_silently_averaged():
    out = compare(BASELINE, [ok(1), ok(2, sim=None), ok(3)])
    assert out.status == ComparisonStatus.INCOMPLETE and not out.passed
    (issue,) = [i for i in out.issues if i.code == "MISSING_SCORES"]
    assert issue.dataset_ids == [2]


def test_succeeded_case_without_latency_is_not_silently_averaged():
    out = compare(BASELINE, [ok(1), ok(2, lat=None), ok(3)])
    assert "MISSING_SCORES" in codes(out)


def test_run_with_no_results_is_incomplete():
    out = compare(BASELINE, [], c_run=run(2, expected=0))
    assert out.status == ComparisonStatus.INCOMPLETE and "NO_RESULTS" in codes(out)


def test_legacy_run_without_expected_count_skips_only_the_count_check():
    out = compare(BASELINE, BASELINE, c_run=run(2, expected=0))
    assert out.status == ComparisonStatus.OK


# ── Policy: nothing that is not OK may pass ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "candidate",
    [
        [failed(1), ok(2), ok(3)],
        [ok(10), ok(11), ok(12)],
        [ok(1), ok(2)],
        [ok(1), ok(1), ok(2), ok(3)],
        [pending(1), pending(2), pending(3)],
        [],
    ],
)
def test_invalid_comparisons_never_pass_and_never_report_metrics(candidate):
    out = compare(BASELINE, candidate, c_run=run(2, expected=len(candidate)))
    assert out.status != ComparisonStatus.OK
    assert out.passed is False
    assert out.report is None
    assert out.issues
    assert "No regressions" not in out.summary
    assert out.summary.startswith("❌ Evaluation ")


def test_issues_are_collected_not_short_circuited():
    out = compare([failed(1), ok(2), ok(3)], [ok(1), ok(2), ok(3), ok(9)],
                  c_run=run(2, expected=4))
    assert {"CASE_FAILED", "COVERAGE_MISMATCH"} <= codes(out)
    assert out.status == ComparisonStatus.INCOMPARABLE        # the more severe status wins


def test_check_comparability_returns_empty_list_when_valid():
    assert check_comparability(run(1), BASELINE, run(2), BASELINE) == []


# ── The metric engine refuses invalid input if called directly ──────────────────────────

def test_compare_runs_rejects_different_coverage():
    with pytest.raises(ValueError, match="different test-case coverage"):
        compare_runs(BASELINE, [ok(1), ok(2)], 1, 2)


def test_compare_runs_rejects_unscored_results():
    with pytest.raises(ValueError, match="unscored"):
        compare_runs(BASELINE, [ok(1), ok(2), ok(3, sim=None)], 1, 2)


def test_compare_runs_rejects_duplicates_and_empty():
    with pytest.raises(ValueError, match="duplicate"):
        compare_runs(BASELINE, [ok(1), ok(1), ok(2), ok(3)], 1, 2)
    with pytest.raises(ValueError, match="at least one"):
        compare_runs([], [], 1, 2)
