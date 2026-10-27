"""
Dashboard route — Phase 4 visual companion.

Serves a single HTML page at GET /dashboard/compare?baseline_run_id=X&candidate_run_id=Y
that renders the same data as /evaluations/compare, but as bar charts and
per-question breakdowns instead of raw JSON.

If the two runs cannot be validly compared (failed, unfinished or mismatched cases)
the page says so, lists the reasons and the failed cases, and shows NO quality
verdict, mirroring the comparison_status of the JSON endpoint.

No new dependencies — uses Chart.js from a CDN inside the returned HTML string.
Everything that comes from the database or the model is HTML-escaped.
"""

import json
from dataclasses import asdict
from html import escape

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session, selectinload

from app.core.comparator import ComparisonOutcome, evaluate_comparison
from app.db.models.dataset import Dataset
from app.db.models.evaluation_result import EvaluationResult, ResultStatus
from app.db.models.evaluation_run import EvaluationRun, RunStatus
from app.db.session import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _css(verdict_color: str, verdict_bg: str) -> str:
    return f"""
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #F1EFE8; color: #2C2C2A; max-width: 880px; margin: 0 auto; padding: 32px 20px 80px; }}
  h1 {{ font-size: 20px; font-weight: 500; margin: 0 0 4px; }}
  .subtitle {{ font-size: 14px; color: #5F5E5A; margin: 0 0 24px; }}
  .verdict {{ display: flex; align-items: center; gap: 10px; background: {verdict_bg}; color: {verdict_color}; padding: 14px 18px; border-radius: 10px; font-size: 15px; font-weight: 500; margin-bottom: 24px; }}
  .verdict-icon {{ width: 24px; height: 24px; border-radius: 50%; background: {verdict_color}; color: white; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 32px; }}
  .metric-card {{ background: white; border-radius: 10px; padding: 16px; border: 0.5px solid #D3D1C7; }}
  .metric-label {{ font-size: 12px; color: #888780; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .metric-row {{ display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }}
  .metric-val-old {{ font-size: 14px; color: #888780; text-decoration: line-through; }}
  .metric-arrow {{ font-size: 14px; color: #B4B2A9; }}
  .metric-val-new {{ font-size: 18px; font-weight: 500; }}
  .metric-delta {{ font-size: 13px; margin: 6px 0 0; font-weight: 500; }}
  .chart-card {{ background: white; border-radius: 10px; padding: 20px; border: 0.5px solid #D3D1C7; margin-bottom: 32px; }}
  .chart-title {{ font-size: 14px; font-weight: 500; margin: 0 0 16px; }}
  .section-title {{ font-size: 14px; font-weight: 500; color: #5F5E5A; margin: 0 0 12px; }}
  .q-card {{ background: white; border-radius: 10px; padding: 16px 18px; border: 0.5px solid #D3D1C7; margin-bottom: 10px; }}
  .q-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px; }}
  .q-text {{ font-size: 14px; font-weight: 500; }}
  .q-delta {{ font-size: 13px; font-weight: 500; white-space: nowrap; }}
  .q-answer-label {{ font-size: 11px; color: #888780; text-transform: uppercase; letter-spacing: 0.03em; margin: 8px 0 2px; }}
  .q-answer {{ font-size: 13px; color: #444441; margin: 0; line-height: 1.5; background: #F1EFE8; padding: 8px 10px; border-radius: 6px; }}
  .issue {{ background: white; border-radius: 10px; padding: 12px 16px; border: 0.5px solid #D3D1C7; margin-bottom: 8px; font-size: 13px; line-height: 1.5; }}
  .issue-code {{ font-size: 11px; color: #888780; text-transform: uppercase; letter-spacing: 0.03em; }}
  .run-stats {{ font-size: 13px; color: #5F5E5A; margin: 0 0 16px; }}
  .score-line {{ font-size: 12px; color: #5F5E5A; margin: 4px 0 0; }}
  .tag {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.03em; color: #888780; border: 0.5px solid #D3D1C7; border-radius: 4px; padding: 0 4px; margin-left: 4px; }}
  .note {{ font-size: 13px; color: #5F5E5A; margin: 0 0 24px; }}
  .checks {{ background: white; border-radius: 10px; padding: 12px 16px; border: 0.5px solid #D3D1C7; margin-bottom: 24px; font-size: 13px; }}
  .checks summary {{ cursor: pointer; font-weight: 500; }}
  .checks table {{ width: 100%; border-collapse: collapse; margin: 12px 0 8px; }}
  .checks th {{ text-align: left; font-size: 11px; color: #888780; text-transform: uppercase; letter-spacing: 0.03em; padding: 4px 8px 4px 0; }}
  .checks td {{ padding: 4px 8px 4px 0; border-top: 0.5px solid #EEECE4; }}
  .checks .note {{ margin: 8px 0 0; }}
"""


def _render_dashboard_html(
    baseline_run_id: int,
    candidate_run_id: int,
    passed: bool,
    summary: str,
    quality: dict,
    evaluators: list[dict],
    latency: dict,
    cost: dict,
    rows: list[dict],
    regressed_cases: list[dict] | None = None,
) -> str:
    verdict_color = "#1D9E75" if passed else "#D85A30"
    verdict_bg = "#E1F5EE" if passed else "#FAECE7"
    verdict_letter = "C" if passed else "X"

    def fmt_pct(value):
        if value is None:
            return "—"
        return f"{value * 100:+.1f}%"

    def metric_card(label, baseline, candidate, delta_rel, regressed, fmt):
        color = "#D85A30" if regressed else "#1D9E75"
        if fmt == "pct":
            b_str = f"{baseline * 100:.1f}%" if baseline is not None else "—"
            c_str = f"{candidate * 100:.1f}%" if candidate is not None else "—"
        elif fmt == "sec":
            b_str = f"{baseline:.2f}s" if baseline is not None else "—"
            c_str = f"{candidate:.2f}s" if candidate is not None else "—"
        else:
            b_str = f"${baseline:.6f}" if baseline is not None else "—"
            c_str = f"${candidate:.6f}" if candidate is not None else "—"
        return f"""
        <div class="metric-card">
          <p class="metric-label">{label}</p>
          <div class="metric-row">
            <span class="metric-val-old">{b_str}</span>
            <span class="metric-arrow">-&gt;</span>
            <span class="metric-val-new" style="color:{color}">{c_str}</span>
          </div>
          <p class="metric-delta" style="color:{color}">{fmt_pct(delta_rel)}</p>
        </div>
        """

    rows_html = ""
    labels = []
    baseline_json = []
    candidate_json = []
    for row in rows:
        b = row["baseline_gate"]
        c = row["candidate_gate"]
        delta_pct = row["gate_delta_pct"]
        delta_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"
        worse = delta_pct is not None and delta_pct < -5
        delta_color = "#D85A30" if worse else "#5F5E5A"
        labels.append(row["question"][:40])
        baseline_json.append(round(b * 100, 1))
        candidate_json.append(round(c * 100, 1))
        # Only checks that changed are listed; unchanged ones would just be noise.
        score_lines = "".join(
            f"""<p class="score-line">{escape(name.replace('_', ' '))}{'' if sc['gating'] else '<span class="tag">advisory</span>'}: """
            f"""{sc['baseline']:.2f} -&gt; {sc['candidate']:.2f}</p>"""
            for name, sc in sorted(row["scores"].items())
            if abs(sc["delta"]) > 1e-9
        )
        rows_html += f"""
        <div class="q-card">
          <div class="q-head">
            <span class="q-text">{escape(row['question'])}</span>
            <span class="q-delta" style="color:{delta_color}">{delta_str}</span>
          </div>
          {score_lines}
          <p class="q-answer-label">baseline answer</p>
          <p class="q-answer">{escape(row['baseline_output'])}</p>
          <p class="q-answer-label">candidate answer</p>
          <p class="q-answer">{escape(row['candidate_output'])}</p>
        </div>
        """

    # One headline number per metric. The per-check detail sits in a collapsed section.
    check_rows = "".join(
        f"""<tr><td>{escape(e['name'].replace('_', ' '))}{'' if e['gating'] else '<span class="tag">advisory</span>'}</td>"""
        f"""<td>{e['n_cases']}</td><td>{e['baseline_avg'] * 100:.1f}%</td><td>{e['candidate_avg'] * 100:.1f}%</td>"""
        f"""<td style="color:{'#D85A30' if e['regression_detected'] else '#5F5E5A'}">"""
        f"""{fmt_pct(e['delta_relative'])}{' (regression)' if e['regression_detected'] else ''}</td></tr>"""
        for e in evaluators
    )
    checks_html = f"""
  <details class="checks">
    <summary>Quality checks ({len(evaluators)})</summary>
    <table>
      <tr><th>Check</th><th>Cases</th><th>Baseline</th><th>Candidate</th><th>Change</th></tr>
      {check_rows}
    </table>
    <p class="note">Quality is the average of each question's gating checks. The verdict looks at every gating check on its own, so one failing group of checks blocks even when the overall average barely moves.</p>
  </details>"""
    regressed_note = ""
    if regressed_cases:
        ids = sorted({c["dataset_id"] for c in regressed_cases})
        regressed_note = (
            f'<p class="note">{len(ids)} case(s) scored lower than baseline on a gating evaluator '
            f"(dataset ids: {escape(', '.join(map(str, ids[:15])))}"
            f"{'…' if len(ids) > 15 else ''}). The verdict uses the aggregate score per evaluator, "
            f"so individual drops below the threshold do not block on their own.</p>"
        )

    # json.dumps gives valid JS. Every "<" becomes < (still the same string in JS) so
    # a question can neither close the script block nor open a new tag inside it.
    labels_json = json.dumps(labels).replace("<", "\\u003c")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PromptGuard - Run {baseline_run_id} vs {candidate_run_id}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>{_css(verdict_color, verdict_bg)}</style>
</head>
<body>

  <h1>Prompt comparison</h1>
  <p class="subtitle">Baseline run #{baseline_run_id} vs candidate run #{candidate_run_id}</p>

  <div class="verdict">
    <span class="verdict-icon">{verdict_letter}</span>
    <span>{escape(summary)}</span>
  </div>

  <div class="metrics-grid">
    {metric_card("Quality", quality['baseline_avg'], quality['candidate_avg'], quality['delta_relative'], quality['regression_detected'], "pct")}
    {metric_card("Latency", latency['baseline_avg'], latency['candidate_avg'], latency['delta_relative'], latency['regression_detected'], "sec")}
    {metric_card("Cost", cost['baseline_avg'], cost['candidate_avg'], cost['delta_relative'], cost['regression_detected'], "cost")}
  </div>

  {checks_html}

  {regressed_note}

  <div class="chart-card">
    <p class="chart-title">Quality by question</p>
    <canvas id="simChart" height="90"></canvas>
  </div>

  <p class="section-title">Per-question breakdown</p>
  {rows_html}

<script>
  const ctx = document.getElementById('simChart');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {labels_json},
      datasets: [
        {{
          label: 'Baseline',
          data: {baseline_json},
          backgroundColor: '#85B7EB',
          borderRadius: 4,
        }},
        {{
          label: 'Candidate',
          data: {candidate_json},
          backgroundColor: '#378ADD',
          borderRadius: 4,
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top', labels: {{ usePointStyle: true, boxWidth: 8 }} }} }},
      scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'quality %' }} }} }}
    }}
  }});
</script>

</body>
</html>"""


def _render_invalid_html(
    outcome: ComparisonOutcome,
    runs: dict[str, EvaluationRun],
    failed_cases: list[dict],
) -> str:
    """Shown instead of the verdict when the runs cannot be validly compared."""
    issues_html = "".join(
        f"""<div class="issue"><span class="issue-code">{escape(i.severity.value)} · {escape(i.code)} · {escape(i.run)}</span><br>{escape(i.message)}</div>"""
        for i in outcome.issues
    )

    stats_html = "".join(
        f"""<p class="run-stats"><b>{label} run #{run.id}</b> — status {escape(run.status.value)}: """
        f"""{run.succeeded_cases} succeeded, {run.failed_cases} failed of {run.expected_cases} expected"""
        f"""{(' — ' + escape(run.error_message)) if run.error_message else ''}</p>"""
        for label, run in runs.items()
    )

    failed_html = "".join(
        f"""
        <div class="q-card">
          <div class="q-head">
            <span class="q-text">{escape(c['question'])}</span>
            <span class="q-delta" style="color:#D85A30">{escape(c['run'])} · {escape(c['category'])}</span>
          </div>
          <p class="q-answer">{escape(c['message'])}</p>
        </div>"""
        for c in failed_cases
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PromptGuard - Run {outcome.baseline_run_id} vs {outcome.candidate_run_id}</title>
<style>{_css("#B26A00", "#FDF1DC")}</style>
</head>
<body>

  <h1>Prompt comparison</h1>
  <p class="subtitle">Baseline run #{outcome.baseline_run_id} vs candidate run #{outcome.candidate_run_id}</p>

  <div class="verdict">
    <span class="verdict-icon">!</span>
    <span>Evaluation {escape(outcome.status.value)} — no quality verdict. Release gate: BLOCKED (policy {escape(outcome.policy_version)}).</span>
  </div>

  {stats_html}

  <p class="section-title">Why the runs cannot be compared</p>
  {issues_html}

  <p class="section-title" style="margin-top:24px">Failed cases</p>
  {failed_html or '<p class="run-stats">No individual case failures were recorded.</p>'}

</body>
</html>"""


@router.get("/compare", response_class=HTMLResponse)
def dashboard_compare(
    baseline_run_id: int = Query(...),
    candidate_run_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """
    Visual HTML version of /evaluations/compare.
    Open in browser: /dashboard/compare?baseline_run_id=1&candidate_run_id=2
    """
    baseline_run = db.get(EvaluationRun, baseline_run_id)
    candidate_run = db.get(EvaluationRun, candidate_run_id)

    if not baseline_run or not candidate_run:
        raise HTTPException(status_code=404, detail="One or both runs not found")
    if any(r.status in (RunStatus.PENDING, RunStatus.RUNNING) for r in (baseline_run, candidate_run)):
        raise HTTPException(status_code=422, detail="Both runs must be finished (not PENDING or RUNNING)")

    # ALL rows, including failed/unfinished ones, so the preflight sees them.
    baseline_results = (
        db.query(EvaluationResult)
        .options(selectinload(EvaluationResult.scores))
        .filter(EvaluationResult.run_id == baseline_run_id)
        .order_by(EvaluationResult.dataset_id)
        .all()
    )
    candidate_results = (
        db.query(EvaluationResult)
        .options(selectinload(EvaluationResult.scores))
        .filter(EvaluationResult.run_id == candidate_run_id)
        .order_by(EvaluationResult.dataset_id)
        .all()
    )

    outcome = evaluate_comparison(baseline_run, baseline_results, candidate_run, candidate_results)

    dataset_ids = {r.dataset_id for r in baseline_results} | {r.dataset_id for r in candidate_results}
    datasets = {d.id: d for d in db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()}

    def question(did: int) -> str:
        ds = datasets.get(did)
        return ds.input_query if ds else f"Dataset #{did}"

    if outcome.report is None:
        failed_cases = [
            {
                "run": label,
                "question": question(r.dataset_id),
                "category": (r.error_category.value if r.error_category else r.status.value),
                "message": r.error_message or f"case is {r.status.value}",
            }
            for label, results in (("baseline", baseline_results), ("candidate", candidate_results))
            for r in results
            if r.status != ResultStatus.SUCCEEDED
        ]
        html = _render_invalid_html(
            outcome,
            runs={"Baseline": baseline_run, "Candidate": candidate_run},
            failed_cases=failed_cases,
        )
        return HTMLResponse(content=html)

    report = outcome.report
    candidate_by_dataset = {r.dataset_id: r for r in candidate_results}
    baseline_by_dataset = {r.dataset_id: r for r in baseline_results}

    per_dataset = {row["dataset_id"]: row for row in report.per_dataset}

    def gate_mean(results_scores: dict, side: str) -> float:
        vals = [sc[side] for sc in results_scores.values() if sc["gating"]]
        return sum(vals) / len(vals)

    rows = []
    for did in sorted(dataset_ids):
        b = baseline_by_dataset[did]   # coverage is identical on both sides (preflight)
        c = candidate_by_dataset[did]
        scores = per_dataset[did]["scores"]
        b_gate, c_gate = gate_mean(scores, "baseline"), gate_mean(scores, "candidate")
        rows.append({
            "question": question(did),
            "scores": scores,
            "baseline_gate": b_gate,
            "candidate_gate": c_gate,
            "gate_delta_pct": round((c_gate - b_gate) * 100, 1),
            "baseline_output": (b.actual_output or "")[:300],
            "candidate_output": (c.actual_output or "")[:300],
        })

    html = _render_dashboard_html(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        passed=report.passed,
        summary=report.summary,
        quality=asdict(report.quality),
        evaluators=[asdict(e) for e in report.evaluators],
        latency=asdict(report.latency),
        cost=asdict(report.cost),
        rows=rows,
        regressed_cases=[asdict(c) for c in report.regressed_cases],
    )
    return HTMLResponse(content=html)
