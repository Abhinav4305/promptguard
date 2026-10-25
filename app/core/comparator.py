"""
Regression Comparator.

Two layers:

1. evaluate_comparison() / check_comparability()  — decides whether two runs may be
   compared at all (same cases, all executed, all scored).
2. compare_runs()                                 — the metric engine. Compares a
   candidate run against a baseline run across three axes:
     - Similarity  (higher is better  -> regression if it drops)
     - Latency     (lower is better   -> regression if it rises)
     - Cost        (lower is better   -> regression if it rises)
   Each axis is evaluated independently; the verdict is FAIL if any single axis
   exceeds its configured threshold.

COMPARISON POLICY (policy_version = POLICY_VERSION)
  * Two runs are compared only if they cover exactly the same test-case IDs, every
    case in both runs executed successfully, and every case has all its metrics.
  * Incomplete evaluations BLOCK the release: INCOMPLETE and INCOMPARABLE both give
    passed=False. They are reported as such, never as a quality regression, and
    no aggregate score or per-metric verdict is produced for them, because an
    average over a subset of cases is not a measurement of the whole benchmark.
  * Failed cases are never silently dropped. Re-run the evaluation (infrastructure
    failures) or fix the cause (model/evaluator errors) and compare again.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import settings
from app.db.models.evaluation_result import ErrorCategory, EvaluationResult, ResultStatus
from app.db.models.evaluation_run import RunStatus

POLICY_VERSION = "comparison-policy-v1"
MAX_IDS_IN_MESSAGE = 10


class ComparisonStatus(str, Enum):
    OK = "OK"                        # runs are comparable; passed reflects the quality verdict
    INCOMPLETE = "INCOMPLETE"        # a run has failed/missing/unscored cases
    INCOMPARABLE = "INCOMPARABLE"    # runs do not cover the same cases, or a run has duplicates


@dataclass
class MetricDelta:
    baseline_avg: float | None
    candidate_avg: float | None
    delta_absolute: float | None      # candidate − baseline
    delta_relative: float | None      # (candidate − baseline) / baseline
    regression_detected: bool
    threshold_used: float


@dataclass
class ComparisonReport:
    baseline_run_id: int
    candidate_run_id: int
    similarity: MetricDelta
    latency: MetricDelta
    cost: MetricDelta
    passed: bool                      # True only if ALL metrics pass
    summary: str
    per_dataset: list[dict] = field(default_factory=list)


@dataclass
class ComparisonIssue:
    code: str
    severity: ComparisonStatus        # INCOMPLETE or INCOMPARABLE
    run: str                          # "baseline" | "candidate" | "both"
    message: str
    dataset_ids: list[int] = field(default_factory=list)


@dataclass
class ComparisonOutcome:
    baseline_run_id: int
    candidate_run_id: int
    status: ComparisonStatus
    passed: bool
    summary: str
    issues: list[ComparisonIssue] = field(default_factory=list)
    report: ComparisonReport | None = None   # only when status == OK
    policy_version: str = POLICY_VERSION


# --------------------------------------------------------------------------------
# Layer 1: may these two runs be compared?
# --------------------------------------------------------------------------------

def _fmt_ids(ids: list[int]) -> str:
    shown = ", ".join(str(i) for i in ids[:MAX_IDS_IN_MESSAGE])
    return shown + (f", … (+{len(ids) - MAX_IDS_IN_MESSAGE} more)" if len(ids) > MAX_IDS_IN_MESSAGE else "")


def _is_scored(r) -> bool:
    return (
        r.status == ResultStatus.SUCCEEDED
        and r.similarity_score is not None
        and r.latency_seconds is not None
        and r.token_cost is not None
    )


def _check_one_run(label: str, run, results) -> list[ComparisonIssue]:
    issues: list[ComparisonIssue] = []
    name = f"{label} run {run.id}"

    if run.status != RunStatus.COMPLETED:
        detail = f" — {run.error_message}" if getattr(run, "error_message", None) else ""
        issues.append(ComparisonIssue(
            "RUN_NOT_COMPLETED", ComparisonStatus.INCOMPLETE, label,
            f"{name} has status {run.status.value}, expected COMPLETED{detail}",
        ))

    if not results:
        issues.append(ComparisonIssue(
            "NO_RESULTS", ComparisonStatus.INCOMPLETE, label, f"{name} has no result rows",
        ))
        return issues

    counts = Counter(r.dataset_id for r in results)
    dupes = sorted(d for d, n in counts.items() if n > 1)
    if dupes:
        issues.append(ComparisonIssue(
            "DUPLICATE_CASES", ComparisonStatus.INCOMPARABLE, label,
            f"{name} has more than one result for case(s) {_fmt_ids(dupes)}", dupes,
        ))

    expected = getattr(run, "expected_cases", 0) or 0
    if expected and expected != len(counts):
        issues.append(ComparisonIssue(
            "RESULT_COUNT_MISMATCH", ComparisonStatus.INCOMPLETE, label,
            f"{name} expected {expected} cases but has results for {len(counts)}",
        ))

    failed = [r for r in results if r.status == ResultStatus.FAILED]
    if failed:
        by_cat = Counter((r.error_category or ErrorCategory.UNKNOWN).value for r in failed)
        cats = ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items()))
        hint = (
            " Infrastructure failures are not quality regressions; re-run the evaluation."
            if by_cat.get(ErrorCategory.INFRASTRUCTURE.value) else ""
        )
        ids = sorted(r.dataset_id for r in failed)
        issues.append(ComparisonIssue(
            "CASE_FAILED", ComparisonStatus.INCOMPLETE, label,
            f"{name}: {len(failed)} of {len(counts)} cases failed ({cats}): case(s) {_fmt_ids(ids)}.{hint}",
            ids,
        ))

    unfinished = [r for r in results if r.status in (ResultStatus.PENDING, ResultStatus.RUNNING)]
    if unfinished:
        ids = sorted(r.dataset_id for r in unfinished)
        issues.append(ComparisonIssue(
            "CASE_NOT_FINISHED", ComparisonStatus.INCOMPLETE, label,
            f"{name}: {len(unfinished)} case(s) never finished: {_fmt_ids(ids)}", ids,
        ))

    unscored = [r for r in results if r.status == ResultStatus.SUCCEEDED and not _is_scored(r)]
    if unscored:
        ids = sorted(r.dataset_id for r in unscored)
        issues.append(ComparisonIssue(
            "MISSING_SCORES", ComparisonStatus.INCOMPLETE, label,
            f"{name}: {len(unscored)} succeeded case(s) are missing a score or metric: {_fmt_ids(ids)}",
            ids,
        ))
    return issues


def check_comparability(baseline_run, baseline_results, candidate_run, candidate_results) -> list[ComparisonIssue]:
    """
    Return every reason the two runs must not be compared (empty list = comparable).

    Runs need: id, status, expected_cases, error_message.
    Results need: dataset_id, status, error_category, similarity_score,
                  latency_seconds, token_cost.
    """
    issues = _check_one_run("baseline", baseline_run, baseline_results)
    issues += _check_one_run("candidate", candidate_run, candidate_results)

    b_ids = {r.dataset_id for r in baseline_results}
    c_ids = {r.dataset_id for r in candidate_results}
    if baseline_results and candidate_results and b_ids != c_ids:
        only_b, only_c = sorted(b_ids - c_ids), sorted(c_ids - b_ids)
        parts = []
        if only_b:
            parts.append(f"only in baseline: {_fmt_ids(only_b)}")
        if only_c:
            parts.append(f"only in candidate: {_fmt_ids(only_c)}")
        issues.append(ComparisonIssue(
            "COVERAGE_MISMATCH", ComparisonStatus.INCOMPARABLE, "both",
            "runs do not cover the same test cases (" + "; ".join(parts) + ")",
            sorted(b_ids ^ c_ids),
        ))
    return issues


def evaluate_comparison(
    baseline_run,
    baseline_results: list[EvaluationResult],
    candidate_run,
    candidate_results: list[EvaluationResult],
) -> ComparisonOutcome:
    """
    The single entry point for API, dashboard and CI: preflight first, then the
    metric comparison only if the runs are valid to compare.

    `baseline_results` / `candidate_results` must be ALL rows of each run, including
    failed and unfinished ones. Pre-filtering them defeats the preflight.
    """
    issues = check_comparability(baseline_run, baseline_results, candidate_run, candidate_results)

    if issues:
        status = (
            ComparisonStatus.INCOMPARABLE
            if any(i.severity == ComparisonStatus.INCOMPARABLE for i in issues)
            else ComparisonStatus.INCOMPLETE
        )
        headline = "; ".join(i.message for i in issues[:3])
        more = f" (+{len(issues) - 3} more issue(s))" if len(issues) > 3 else ""
        return ComparisonOutcome(
            baseline_run_id=baseline_run.id,
            candidate_run_id=candidate_run.id,
            status=status,
            passed=False,  # policy: incomplete or incomparable evaluations block the release
            summary=f"❌ Evaluation {status.value}, not a quality verdict: {headline}{more}",
            issues=issues,
        )

    report = compare_runs(
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
    )
    return ComparisonOutcome(
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
        status=ComparisonStatus.OK,
        passed=report.passed,
        summary=report.summary,
        report=report,
    )


# --------------------------------------------------------------------------------
# Layer 2: metric comparison
# --------------------------------------------------------------------------------

def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _delta(baseline: float | None, candidate: float | None) -> tuple[float | None, float | None]:
    if baseline is None or candidate is None:
        return None, None
    abs_delta = round(candidate - baseline, 6)
    rel_delta = round(abs_delta / baseline, 6) if baseline != 0 else None
    return abs_delta, rel_delta


def _require_comparable_inputs(baseline_results, candidate_results) -> None:
    """Defense in depth: compare_runs must never average over a mismatched or partial set."""
    b_ids = [r.dataset_id for r in baseline_results]
    c_ids = [r.dataset_id for r in candidate_results]
    if not b_ids or not c_ids:
        raise ValueError("compare_runs needs at least one result in each run")
    if len(set(b_ids)) != len(b_ids) or len(set(c_ids)) != len(c_ids):
        raise ValueError("compare_runs received duplicate results for a test case")
    if set(b_ids) != set(c_ids):
        raise ValueError("compare_runs received runs with different test-case coverage")
    for r in [*baseline_results, *candidate_results]:
        if r.similarity_score is None or r.latency_seconds is None or r.token_cost is None:
            raise ValueError(f"compare_runs received an unscored result for case {r.dataset_id}")


def compare_runs(
    baseline_results: list[EvaluationResult],
    candidate_results: list[EvaluationResult],
    baseline_run_id: int,
    candidate_run_id: int,
) -> ComparisonReport:
    """
    Metric engine. Precondition: both result sets are complete and cover the same
    cases; use evaluate_comparison() unless you have already run check_comparability().
    """
    _require_comparable_inputs(baseline_results, candidate_results)

    # ── Aggregate averages ───────────────────────────────────────────────────
    b_sim = _avg([r.similarity_score for r in baseline_results])
    c_sim = _avg([r.similarity_score for r in candidate_results])
    b_lat = _avg([r.latency_seconds for r in baseline_results])
    c_lat = _avg([r.latency_seconds for r in candidate_results])
    b_cost = _avg([r.token_cost for r in baseline_results])
    c_cost = _avg([r.token_cost for r in candidate_results])

    # ── Compute deltas ───────────────────────────────────────────────────────
    sim_abs, sim_rel = _delta(b_sim, c_sim)
    lat_abs, lat_rel = _delta(b_lat, c_lat)
    cost_abs, cost_rel = _delta(b_cost, c_cost)

    # ── Evaluate thresholds ──────────────────────────────────────────────────
    # Similarity regression: candidate drops MORE than threshold below baseline
    sim_regressed = (
        sim_rel is not None
        and sim_rel < -settings.regression_similarity_drop_threshold
    )
    # Latency regression: candidate rises MORE than threshold above baseline
    lat_regressed = (
        lat_rel is not None
        and lat_rel > settings.regression_latency_spike_threshold
    )
    # Cost regression: candidate rises MORE than threshold above baseline
    cost_regressed = (
        cost_rel is not None
        and cost_rel > settings.regression_cost_spike_threshold
    )

    passed = not any([sim_regressed, lat_regressed, cost_regressed])

    # ── Per-dataset breakdown (useful for PR comments) ───────────────────────
    # Coverage is identical on both sides (checked above), so no case is skipped.
    baseline_by_dataset = {r.dataset_id: r for r in baseline_results}
    per_dataset = []
    for c_result in candidate_results:
        b_result = baseline_by_dataset[c_result.dataset_id]
        row_sim_abs, row_sim_rel = _delta(b_result.similarity_score, c_result.similarity_score)
        row_lat_abs, _ = _delta(b_result.latency_seconds, c_result.latency_seconds)
        per_dataset.append({
            "dataset_id": c_result.dataset_id,
            "baseline_similarity": b_result.similarity_score,
            "candidate_similarity": c_result.similarity_score,
            "similarity_delta": row_sim_abs,
            "similarity_delta_pct": round(row_sim_rel * 100, 2) if row_sim_rel is not None else None,
            "baseline_latency_s": b_result.latency_seconds,
            "candidate_latency_s": c_result.latency_seconds,
            "latency_delta_s": row_lat_abs,
        })

    # ── Human-readable summary ───────────────────────────────────────────────
    flags = []
    if sim_regressed:
        flags.append(
            f"similarity dropped {abs(sim_rel)*100:.1f}% "
            f"(threshold {settings.regression_similarity_drop_threshold*100:.0f}%)"
        )
    if lat_regressed:
        flags.append(
            f"latency increased {lat_rel*100:.1f}% "
            f"(threshold {settings.regression_latency_spike_threshold*100:.0f}%)"
        )
    if cost_regressed:
        flags.append(
            f"cost increased {cost_rel*100:.1f}% "
            f"(threshold {settings.regression_cost_spike_threshold*100:.0f}%)"
        )

    summary = "✅ No regressions detected." if passed else "❌ Regressions: " + "; ".join(flags)

    return ComparisonReport(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        similarity=MetricDelta(
            baseline_avg=b_sim,
            candidate_avg=c_sim,
            delta_absolute=sim_abs,
            delta_relative=sim_rel,
            regression_detected=sim_regressed,
            threshold_used=settings.regression_similarity_drop_threshold,
        ),
        latency=MetricDelta(
            baseline_avg=b_lat,
            candidate_avg=c_lat,
            delta_absolute=lat_abs,
            delta_relative=lat_rel,
            regression_detected=lat_regressed,
            threshold_used=settings.regression_latency_spike_threshold,
        ),
        cost=MetricDelta(
            baseline_avg=b_cost,
            candidate_avg=c_cost,
            delta_absolute=cost_abs,
            delta_relative=cost_rel,
            regression_detected=cost_regressed,
            threshold_used=settings.regression_cost_spike_threshold,
        ),
        passed=passed,
        summary=summary,
        per_dataset=per_dataset,
    )
