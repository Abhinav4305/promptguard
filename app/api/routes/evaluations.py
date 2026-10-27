import logging
from collections import Counter
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, selectinload

from app.core.comparator import ComparisonOutcome, evaluate_comparison
from app.core.run_executor import mark_run_failed
from app.db.models.dataset import Dataset
from app.db.models.evaluation_result import EvaluationResult, ResultStatus
from app.db.models.evaluation_run import EvaluationRun, RunStatus
from app.db.models.prompt import Prompt
from app.db.session import get_db
from app.schemas.evaluation import (
    ComparisonIssueResponse,
    ComparisonResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    MetricDeltaResponse,
)
from app.tasks.evaluation_tasks import run_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/run", response_model=EvaluationRunResponse, status_code=202)
def trigger_evaluation(payload: EvaluationRunRequest, db: Session = Depends(get_db)):
    """
    Creates an EvaluationRun (PENDING) together with one PENDING EvaluationResult per
    test case, then dispatches to Celery. The set of cases is fixed here, so
    `expected_cases` is known up front and a run can never "complete" with fewer.

    An empty dataset_ids list means every dataset that exists right now.
    Callers poll GET /evaluations/{id} until the status is terminal:
    COMPLETED (all cases succeeded), PARTIAL (some failed) or FAILED.
    """
    if not db.get(Prompt, payload.prompt_id):
        raise HTTPException(status_code=404, detail=f"Prompt {payload.prompt_id} not found")

    dataset_ids = payload.dataset_ids
    duplicates = sorted(d for d, n in Counter(dataset_ids).items() if n > 1)
    if duplicates:
        raise HTTPException(status_code=422, detail=f"Duplicate dataset_ids: {duplicates}")

    if dataset_ids:
        existing = {
            d for (d,) in db.query(Dataset.id).filter(Dataset.id.in_(dataset_ids)).all()
        }
        missing = sorted(set(dataset_ids) - existing)
        if missing:
            raise HTTPException(status_code=422, detail=f"Unknown dataset_ids: {missing}")
    else:
        dataset_ids = [d for (d,) in db.query(Dataset.id).order_by(Dataset.id).all()]
        if not dataset_ids:
            raise HTTPException(status_code=422, detail="No datasets exist to evaluate")

    # Run and placeholders are created in one transaction: no half-created runs.
    run = EvaluationRun(
        prompt_id=payload.prompt_id,
        status=RunStatus.PENDING,
        is_baseline=payload.is_baseline,
        expected_cases=len(dataset_ids),
    )
    db.add(run)
    db.flush()
    db.add_all(
        EvaluationResult(run_id=run.id, dataset_id=did, status=ResultStatus.PENDING)
        for did in dataset_ids
    )
    db.commit()
    db.refresh(run)

    try:
        run_evaluation.delay(run.id)
    except Exception as exc:  # broker down: do not leave a run PENDING forever
        logger.exception("Could not enqueue run %s", run.id)
        mark_run_failed(db, run.id, f"could not enqueue evaluation task: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail=f"Run {run.id} could not be queued") from exc

    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results).selectinload(EvaluationResult.scores))
        .filter(EvaluationRun.id == run.id)
        .first()
    )
    return run


def build_comparison_response(outcome: ComparisonOutcome) -> ComparisonResponse:
    report = outcome.report
    return ComparisonResponse(
        baseline_run_id=outcome.baseline_run_id,
        candidate_run_id=outcome.candidate_run_id,
        comparison_status=outcome.status,
        policy_version=outcome.policy_version,
        passed=outcome.passed,
        summary=outcome.summary,
        issues=[ComparisonIssueResponse(**asdict(i)) for i in outcome.issues],
        quality=MetricDeltaResponse(**asdict(report.quality)) if report else None,
        similarity=MetricDeltaResponse(**asdict(report.similarity)) if report and report.similarity else None,
        latency=MetricDeltaResponse(**asdict(report.latency)) if report else None,
        cost=MetricDeltaResponse(**asdict(report.cost)) if report else None,
        per_dataset=report.per_dataset if report else [],
        evaluators=[asdict(e) for e in report.evaluators] if report else [],
        regressed_cases=[asdict(c) for c in report.regressed_cases] if report else [],
    )


@router.get("/compare", response_model=ComparisonResponse)
def compare(
    baseline_run_id: int = Query(..., description="Run marked as baseline"),
    candidate_run_id: int = Query(..., description="New run to evaluate"),
    db: Session = Depends(get_db),
):
    """
    Compare a candidate run against a baseline run.

    Before any metric is computed the two runs are checked for comparability
    (same test cases, every case executed and scored by the same evaluators). The result carries a
    `comparison_status`:

      OK            runs are comparable. `passed` is the quality verdict:
                      - Drop of a gating evaluator's mean score beyond its threshold
                        (similarity 10%, correctness 10%)   → FAIL
                      - Latency spike  > 50%   → FAIL
                      - Cost spike     > 20%   → FAIL
                    (thresholds overridable via environment variables)
      INCOMPLETE    a run has failed, unfinished or unscored cases
      INCOMPARABLE  the runs do not cover the same cases
    For INCOMPLETE and INCOMPARABLE `passed` is always false (policy: incomplete
    evaluations block the release), `issues` says exactly why, and no aggregate
    score is reported.

    HTTP 200 = comparison ran (check `comparison_status` and `passed`).
    HTTP 404 = run not found.
    HTTP 422 = a run is still PENDING/RUNNING. Terminal runs that did not fully
               succeed (PARTIAL, FAILED) return 200 with comparison_status=INCOMPLETE.
    """
    baseline_run = db.get(EvaluationRun, baseline_run_id)
    candidate_run = db.get(EvaluationRun, candidate_run_id)

    if not baseline_run:
        raise HTTPException(status_code=404, detail=f"Baseline run {baseline_run_id} not found")
    if not candidate_run:
        raise HTTPException(status_code=404, detail=f"Candidate run {candidate_run_id} not found")

    for label, run in (("Baseline", baseline_run), ("Candidate", candidate_run)):
        if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
            raise HTTPException(
                status_code=422,
                detail=f"{label} run {run.id} is not finished (status={run.status.value})",
            )

    # ALL rows of each run: failed and unfinished ones must reach the preflight.
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

    outcome = evaluate_comparison(
        baseline_run, baseline_results, candidate_run, candidate_results
    )
    return build_comparison_response(outcome)


@router.patch("/{run_id}/set-baseline", response_model=EvaluationRunResponse)
def set_baseline(run_id: int, db: Session = Depends(get_db)):
    """
    Mark a COMPLETED run as the new baseline for its prompt.
    Clears the baseline flag on any previously-marked run for the same prompt.
    PARTIAL and FAILED runs are rejected: a baseline must have every case scored.
    """
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="EvaluationRun not found")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Only COMPLETED runs can be set as baseline (status={run.status.value})",
        )

    # Clear existing baselines for this prompt
    db.query(EvaluationRun).filter(
        EvaluationRun.prompt_id == run.prompt_id,
        EvaluationRun.is_baseline == True,  # noqa: E712
        EvaluationRun.id != run_id,
    ).update({"is_baseline": False})

    run.is_baseline = True
    db.commit()

    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results).selectinload(EvaluationResult.scores))
        .filter(EvaluationRun.id == run_id)
        .first()
    )
    return run


@router.get("/{run_id}", response_model=EvaluationRunResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results).selectinload(EvaluationResult.scores))
        .filter(EvaluationRun.id == run_id)
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="EvaluationRun not found")
    return run


@router.get("/", response_model=list[EvaluationRunResponse])
def list_runs(db: Session = Depends(get_db)):
    return (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results).selectinload(EvaluationResult.scores))
        .all()
    )
