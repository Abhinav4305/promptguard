"""
Dashboard route — Phase 4 visual companion.

Serves a single HTML page at GET /dashboard/compare?baseline_run_id=X&candidate_run_id=Y
that renders the same data as /evaluations/compare, but as bar charts and
per-question breakdowns instead of raw JSON.

No new dependencies — uses Chart.js from a CDN inside the returned HTML string.
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.comparator import compare_runs
from app.db.models.dataset import Dataset
from app.db.models.evaluation_result import EvaluationResult
from app.db.models.evaluation_run import EvaluationRun, RunStatus
from app.db.session import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _render_dashboard_html(
    baseline_run_id: int,
    candidate_run_id: int,
    passed: bool,
    summary: str,
    similarity: dict,
    latency: dict,
    cost: dict,
    rows: list[dict],
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
    labels_json = []
    baseline_json = []
    candidate_json = []
    for row in rows:
        b = row["baseline_similarity"] or 0
        c = row["candidate_similarity"] or 0
        delta_pct = row["similarity_delta_pct"]
        delta_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"
        worse = delta_pct is not None and delta_pct < -5
        delta_color = "#D85A30" if worse else "#5F5E5A"
        labels_json.append(row["question"][:40])
        baseline_json.append(round(b * 100, 1))
        candidate_json.append(round(c * 100, 1))
        rows_html += f"""
        <div class="q-card">
          <div class="q-head">
            <span class="q-text">{row['question']}</span>
            <span class="q-delta" style="color:{delta_color}">{delta_str}</span>
          </div>
          <p class="q-answer-label">baseline answer</p>
          <p class="q-answer">{row['baseline_output']}</p>
          <p class="q-answer-label">candidate answer</p>
          <p class="q-answer">{row['candidate_output']}</p>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PromptGuard - Run {baseline_run_id} vs {candidate_run_id}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
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
</style>
</head>
<body>

  <h1>Prompt comparison</h1>
  <p class="subtitle">Baseline run #{baseline_run_id} vs candidate run #{candidate_run_id}</p>

  <div class="verdict">
    <span class="verdict-icon">{verdict_letter}</span>
    <span>{summary}</span>
  </div>

  <div class="metrics-grid">
    {metric_card("Similarity", similarity['baseline_avg'], similarity['candidate_avg'], similarity['delta_relative'], similarity['regression_detected'], "pct")}
    {metric_card("Latency", latency['baseline_avg'], latency['candidate_avg'], latency['delta_relative'], latency['regression_detected'], "sec")}
    {metric_card("Cost", cost['baseline_avg'], cost['candidate_avg'], cost['delta_relative'], cost['regression_detected'], "cost")}
  </div>

  <div class="chart-card">
    <p class="chart-title">Similarity score by question</p>
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
      scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'similarity %' }} }} }}
    }}
  }});
</script>

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
    if baseline_run.status != RunStatus.COMPLETED or candidate_run.status != RunStatus.COMPLETED:
        raise HTTPException(status_code=422, detail="Both runs must be COMPLETED")

    baseline_results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.run_id == baseline_run_id, EvaluationResult.similarity_score.is_not(None))
        .all()
    )
    candidate_results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.run_id == candidate_run_id, EvaluationResult.similarity_score.is_not(None))
        .all()
    )

    if not baseline_results or not candidate_results:
        raise HTTPException(status_code=422, detail="Both runs need completed result rows")

    report = compare_runs(
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
    )

    dataset_ids = {r.dataset_id for r in baseline_results} | {r.dataset_id for r in candidate_results}
    datasets = {d.id: d for d in db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()}
    candidate_by_dataset = {r.dataset_id: r for r in candidate_results}
    baseline_by_dataset = {r.dataset_id: r for r in baseline_results}

    rows = []
    for did in sorted(dataset_ids):
        b = baseline_by_dataset.get(did)
        c = candidate_by_dataset.get(did)
        if not b or not c:
            continue
        ds = datasets.get(did)
        rows.append({
            "question": ds.input_query if ds else f"Dataset #{did}",
            "baseline_similarity": b.similarity_score,
            "candidate_similarity": c.similarity_score,
            "similarity_delta_pct": round((c.similarity_score - b.similarity_score) * 100, 1)
                if b.similarity_score is not None and c.similarity_score is not None else None,
            "baseline_output": (b.actual_output or "")[:300],
            "candidate_output": (c.actual_output or "")[:300],
        })

    html = _render_dashboard_html(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        passed=report.passed,
        summary=report.summary,
        similarity=asdict(report.similarity),
        latency=asdict(report.latency),
        cost=asdict(report.cost),
        rows=rows,
    )
    return HTMLResponse(content=html)