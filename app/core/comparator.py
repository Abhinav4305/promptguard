"""
Regression Comparator — Phase 4 core logic.

Compares a candidate run against a baseline run across three axes:
  - Similarity  (higher is better  → regression if it drops)
  - Latency     (lower is better   → regression if it rises)
  - Cost        (lower is better   → regression if it rises)

Each axis is evaluated independently. The overall verdict is FAIL if any
single axis exceeds its configured threshold.
"""

from dataclasses import dataclass, field

from app.core.config import settings
from app.db.models.evaluation_result import EvaluationResult


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


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _delta(baseline: float | None, candidate: float | None) -> tuple[float | None, float | None]:
    if baseline is None or candidate is None:
        return None, None
    abs_delta = round(candidate - baseline, 6)
    rel_delta = round(abs_delta / baseline, 6) if baseline != 0 else None
    return abs_delta, rel_delta


def compare_runs(
    baseline_results: list[EvaluationResult],
    candidate_results: list[EvaluationResult],
    baseline_run_id: int,
    candidate_run_id: int,
) -> ComparisonReport:
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
    baseline_by_dataset = {r.dataset_id: r for r in baseline_results}
    per_dataset = []
    for c_result in candidate_results:
        b_result = baseline_by_dataset.get(c_result.dataset_id)
        if not b_result:
            continue
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
