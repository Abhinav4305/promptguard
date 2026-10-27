"""
Regression Comparator.

Two layers:

1. evaluate_comparison() / check_comparability()  — decides whether two runs may be
   compared at all (same cases, all executed, all scored, scored by the same evaluators).
2. compare_runs()                                 — the metric engine. Compares a
   candidate run against a baseline run across:
     - one axis PER EVALUATOR (higher is better -> regression if the mean drops
       by more than that evaluator kind's threshold). Only GATING evaluators can fail
       the verdict; non-gating ones are reported as advisory.
     - Latency     (lower is better   -> regression if it rises)
     - Cost        (lower is better   -> regression if it rises)
   Each axis is evaluated independently; the verdict is FAIL if any single gating
   axis exceeds its configured threshold.

COMPARISON POLICY (policy_version = POLICY_VERSION)
  * Two runs are compared only if they cover exactly the same test-case IDs, every
    case in both runs executed successfully, every case has a valid score from every
    evaluator (a missing, ERROR or out-of-range score is never skipped), every case has
    at least one gating score, and both runs scored each case with the same evaluators
    (name, version, config, gating). A changed evaluator is a different measurement.
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
from app.core.evaluators.base import MissingScoreError, aggregate, is_valid_score
from app.db.models.evaluation_result import ErrorCategory, EvaluationResult, ResultStatus
from app.db.models.evaluation_run import RunStatus

POLICY_VERSION = "comparison-policy-v2"
TEXT_SIMILARITY = "text_similarity"
SIMILARITY_KIND = "similarity"
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
class EvaluatorDelta:
    """Aggregate of one evaluator (name+version, gating or not) over the cases it scored."""
    name: str
    version: str
    kind: str
    gating: bool
    n_cases: int
    baseline_avg: float
    candidate_avg: float
    delta_absolute: float
    delta_relative: float | None
    threshold_used: float
    regression_detected: bool         # only ever True for gating evaluators
    advisory_regression: bool         # non-gating evaluator dropped past its threshold


@dataclass
class RegressedCase:
    """A case where a gating score went down. Informational: the verdict uses aggregates."""
    dataset_id: int
    evaluator: str
    baseline_score: float
    candidate_score: float


@dataclass
class ComparisonReport:
    baseline_run_id: int
    candidate_run_id: int
    quality: MetricDelta              # per-case mean of the GATING scores, averaged over cases (headline number)
    similarity: MetricDelta | None    # text_similarity over all cases that have it; None if none do (legacy detail)
    latency: MetricDelta
    cost: MetricDelta
    passed: bool                      # True only if ALL gating metrics pass
    summary: str
    per_dataset: list[dict] = field(default_factory=list)
    evaluators: list[EvaluatorDelta] = field(default_factory=list)
    regressed_cases: list[RegressedCase] = field(default_factory=list)


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


def _scores_of(r) -> list:
    return list(getattr(r, "scores", None) or [])


def _score_ok(sc) -> bool:
    status = getattr(sc.status, "value", sc.status)
    return status == "OK" and is_valid_score(sc.score)


def _is_scored(r) -> bool:
    scores = _scores_of(r)
    return (
        r.status == ResultStatus.SUCCEEDED
        and r.latency_seconds is not None
        and r.token_cost is not None
        and bool(scores)
        and all(_score_ok(sc) for sc in scores)
    )


def _signature(r) -> frozenset:
    """Which evaluators scored this case, and how: what two runs must agree on."""
    return frozenset(
        (sc.evaluator_name, sc.evaluator_version, sc.config_hash, bool(sc.gating)) for sc in _scores_of(r)
    )


def _fmt_sig(sig: frozenset) -> str:
    return ", ".join(
        f"{n} v{v} [{h}]{'' if g else ' (advisory)'}" for n, v, h, g in sorted(sig)
    ) or "none"


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
            f"{name}: {len(unscored)} succeeded case(s) have a missing, failed or invalid evaluator score "
            f"or metric: {_fmt_ids(ids)}",
            ids,
        ))

    ungated = [
        r for r in results
        if _is_scored(r) and not any(sc.gating for sc in _scores_of(r))
    ]
    if ungated:
        ids = sorted(r.dataset_id for r in ungated)
        issues.append(ComparisonIssue(
            "NO_GATING_SCORE", ComparisonStatus.INCOMPLETE, label,
            f"{name}: {len(ungated)} case(s) have no gating evaluator, so they could never block "
            f"a release: {_fmt_ids(ids)}",
            ids,
        ))
    return issues


def check_comparability(baseline_run, baseline_results, candidate_run, candidate_results) -> list[ComparisonIssue]:
    """
    Return every reason the two runs must not be compared (empty list = comparable).

    Runs need: id, status, expected_cases, error_message.
    Results need: dataset_id, status, error_category, latency_seconds, token_cost and
                  `scores` (objects with evaluator_name, evaluator_version, config_hash,
                  gating, status, score).
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

    # Same case must have been scored by the same evaluators in both runs. Only checked
    # where both sides are fully scored: other problems are already reported above.
    b_by = {r.dataset_id: r for r in baseline_results if _is_scored(r)}
    c_by = {r.dataset_id: r for r in candidate_results if _is_scored(r)}
    diff = sorted(d for d in b_by.keys() & c_by.keys() if _signature(b_by[d]) != _signature(c_by[d]))
    if diff:
        first = diff[0]
        issues.append(ComparisonIssue(
            "EVALUATOR_MISMATCH", ComparisonStatus.INCOMPARABLE, "both",
            f"baseline and candidate were scored by different evaluators (or evaluator versions/configs) "
            f"on case(s) {_fmt_ids(diff)}; e.g. case {first}: baseline [{_fmt_sig(_signature(b_by[first]))}] "
            f"vs candidate [{_fmt_sig(_signature(c_by[first]))}]. Re-run the baseline with the current "
            f"benchmark/evaluators.",
            diff,
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
        if not _is_scored(r):
            raise ValueError(f"compare_runs received an unscored result for case {r.dataset_id}")
        if not any(sc.gating for sc in _scores_of(r)):
            raise ValueError(f"compare_runs received case {r.dataset_id} with no gating evaluator")
    c_by = {r.dataset_id: r for r in candidate_results}
    for b in baseline_results:
        if _signature(b) != _signature(c_by[b.dataset_id]):
            raise ValueError(f"compare_runs received different evaluators for case {b.dataset_id}")


def _threshold_for(kind: str) -> float:
    if kind == SIMILARITY_KIND:
        return settings.regression_similarity_drop_threshold
    return settings.regression_correctness_drop_threshold


def _evaluator_deltas(baseline_results, candidate_results):
    """Aggregate every (evaluator, version, gating) group; returns deltas and regressed cases."""
    c_by = {r.dataset_id: r for r in candidate_results}
    groups: dict[tuple, dict[str, list]] = {}
    kinds: dict[tuple, str] = {}
    for b in sorted(baseline_results, key=lambda r: r.dataset_id):
        c = c_by[b.dataset_id]
        c_scores = {sc.evaluator_name: sc for sc in _scores_of(c)}
        for bs in _scores_of(b):
            cs = c_scores[bs.evaluator_name]
            key = (bs.evaluator_name, bs.evaluator_version, bool(bs.gating))
            g = groups.setdefault(key, {"b": [], "c": [], "ids": []})
            g["b"].append(bs.score)
            g["c"].append(cs.score)
            g["ids"].append(b.dataset_id)
            kinds[key] = bs.kind

    deltas: list[EvaluatorDelta] = []
    regressed: list[RegressedCase] = []
    for key in sorted(groups):
        name, version, gating = key
        g = groups[key]
        try:
            b_avg, c_avg = aggregate(g["b"]), aggregate(g["c"])
        except MissingScoreError as exc:      # unreachable after the preflight; fail loudly if not
            raise ValueError(f"cannot aggregate {name}: {exc}") from exc
        b_avg, c_avg = round(b_avg, 6), round(c_avg, 6)
        abs_d, rel_d = _delta(b_avg, c_avg)
        threshold = _threshold_for(kinds[key])
        dropped = rel_d is not None and rel_d < -threshold
        deltas.append(EvaluatorDelta(
            name=name, version=version, kind=kinds[key], gating=gating, n_cases=len(g["ids"]),
            baseline_avg=b_avg, candidate_avg=c_avg, delta_absolute=abs_d, delta_relative=rel_d,
            threshold_used=threshold, regression_detected=dropped and gating,
            advisory_regression=dropped and not gating,
        ))
        if gating:
            for did, bv, cv in zip(g["ids"], g["b"], g["c"]):
                if cv < bv - 1e-9:
                    regressed.append(RegressedCase(did, name, bv, cv))
    regressed.sort(key=lambda x: (x.dataset_id, x.evaluator))
    return deltas, regressed


def _similarity_delta(baseline_results, candidate_results, deltas) -> MetricDelta | None:
    def sims(results):
        out = []
        for r in results:
            for sc in _scores_of(r):
                if sc.evaluator_name == TEXT_SIMILARITY:
                    out.append(sc.score)
        return out

    b, c = sims(baseline_results), sims(candidate_results)
    if not b or not c:
        return None
    b_avg, c_avg = round(aggregate(b), 6), round(aggregate(c), 6)
    abs_d, rel_d = _delta(b_avg, c_avg)
    return MetricDelta(
        baseline_avg=b_avg, candidate_avg=c_avg, delta_absolute=abs_d, delta_relative=rel_d,
        regression_detected=any(d.regression_detected for d in deltas if d.name == TEXT_SIMILARITY),
        threshold_used=settings.regression_similarity_drop_threshold,
    )


def _case_quality(r) -> float:
    """One number per case: the mean of its gating scores (advisory scores do not count)."""
    gating = [sc.score for sc in _scores_of(r) if sc.gating]
    return aggregate(gating)


def _quality_delta(baseline_results, candidate_results, regressed: bool) -> MetricDelta:
    """
    The single quality figure. `regression_detected` is the verdict of the per-check
    comparison (any gating evaluator dropped past its threshold), NOT a threshold on this
    overall mean: a collapse in one small group of checks can move the overall mean by less
    than the threshold and must still block.
    """
    b = round(aggregate([_case_quality(r) for r in baseline_results]), 6)
    c = round(aggregate([_case_quality(r) for r in candidate_results]), 6)
    abs_d, rel_d = _delta(b, c)
    return MetricDelta(
        baseline_avg=b, candidate_avg=c, delta_absolute=abs_d, delta_relative=rel_d,
        regression_detected=regressed, threshold_used=settings.regression_correctness_drop_threshold,
    )


def _text_similarity(r) -> float | None:
    return next((sc.score for sc in _scores_of(r) if sc.evaluator_name == TEXT_SIMILARITY), None)


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

    # ── Per-evaluator aggregates ─────────────────────────────────────────────
    ev_deltas, regressed_cases = _evaluator_deltas(baseline_results, candidate_results)
    similarity = _similarity_delta(baseline_results, candidate_results, ev_deltas)

    # ── Latency and cost ─────────────────────────────────────────────────────
    b_lat = _avg([r.latency_seconds for r in baseline_results])
    c_lat = _avg([r.latency_seconds for r in candidate_results])
    b_cost = _avg([r.token_cost for r in baseline_results])
    c_cost = _avg([r.token_cost for r in candidate_results])
    lat_abs, lat_rel = _delta(b_lat, c_lat)
    cost_abs, cost_rel = _delta(b_cost, c_cost)

    # Latency regression: candidate rises MORE than threshold above baseline
    lat_regressed = lat_rel is not None and lat_rel > settings.regression_latency_spike_threshold
    # Cost regression: candidate rises MORE than threshold above baseline
    cost_regressed = cost_rel is not None and cost_rel > settings.regression_cost_spike_threshold
    quality_regressed = any(d.regression_detected for d in ev_deltas)
    quality = _quality_delta(baseline_results, candidate_results, quality_regressed)

    passed = not (quality_regressed or lat_regressed or cost_regressed)

    # ── Per-dataset breakdown (useful for PR comments) ───────────────────────
    # Coverage is identical on both sides (checked above), so no case is skipped.
    baseline_by_dataset = {r.dataset_id: r for r in baseline_results}
    per_dataset = []
    for c_result in candidate_results:
        b_result = baseline_by_dataset[c_result.dataset_id]
        b_sim, c_sim = _text_similarity(b_result), _text_similarity(c_result)
        row_sim_abs, row_sim_rel = _delta(b_sim, c_sim)
        row_lat_abs, _ = _delta(b_result.latency_seconds, c_result.latency_seconds)
        b_q, c_q = _case_quality(b_result), _case_quality(c_result)
        row_q_abs, row_q_rel = _delta(b_q, c_q)
        b_scores = {sc.evaluator_name: sc for sc in _scores_of(b_result)}
        scores = {}
        for sc in _scores_of(c_result):
            bs = b_scores[sc.evaluator_name]
            scores[sc.evaluator_name] = {
                "baseline": bs.score, "candidate": sc.score,
                "delta": round(sc.score - bs.score, 6), "gating": bool(sc.gating),
                "kind": sc.kind, "version": sc.evaluator_version,
            }
        per_dataset.append({
            "dataset_id": c_result.dataset_id,
            "baseline_quality": round(b_q, 6),
            "candidate_quality": round(c_q, 6),
            "quality_delta": row_q_abs,
            "quality_delta_pct": round(row_q_rel * 100, 2) if row_q_rel is not None else None,
            "baseline_similarity": b_sim,
            "candidate_similarity": c_sim,
            "similarity_delta": row_sim_abs,
            "similarity_delta_pct": round(row_sim_rel * 100, 2) if row_sim_rel is not None else None,
            "baseline_latency_s": b_result.latency_seconds,
            "candidate_latency_s": c_result.latency_seconds,
            "latency_delta_s": row_lat_abs,
            "scores": scores,
        })

    # ── Human-readable summary ───────────────────────────────────────────────
    flags = []
    for d in ev_deltas:
        if d.regression_detected:
            label = "similarity" if d.name == TEXT_SIMILARITY else d.name
            flags.append(
                f"{label} dropped {abs(d.delta_relative)*100:.1f}% "
                f"(threshold {d.threshold_used*100:.0f}%)"
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

    if passed:
        summary = "✅ No regressions detected."
        notes = []
        if regressed_cases:
            n = len({x.dataset_id for x in regressed_cases})
            notes.append(f"{n} case(s) scored lower on a gating evaluator but the change is within threshold")
        advisory = [d.name for d in ev_deltas if d.advisory_regression]
        if advisory:
            notes.append(f"advisory (non-gating) drop in {', '.join(sorted(set(advisory)))}")
        if notes:
            summary += " Note: " + "; ".join(notes) + "."
    else:
        summary = "❌ Regressions: " + "; ".join(flags)

    return ComparisonReport(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        quality=quality,
        similarity=similarity,
        latency=MetricDelta(
            baseline_avg=b_lat, candidate_avg=c_lat, delta_absolute=lat_abs, delta_relative=lat_rel,
            regression_detected=lat_regressed, threshold_used=settings.regression_latency_spike_threshold,
        ),
        cost=MetricDelta(
            baseline_avg=b_cost, candidate_avg=c_cost, delta_absolute=cost_abs, delta_relative=cost_rel,
            regression_detected=cost_regressed, threshold_used=settings.regression_cost_spike_threshold,
        ),
        passed=passed,
        summary=summary,
        per_dataset=per_dataset,
        evaluators=ev_deltas,
        regressed_cases=regressed_cases,
    )
