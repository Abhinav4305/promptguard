from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, selectinload
from dataclasses import asdict

from app.core.comparator import compare_runs
from app.db.models.evaluation_result import EvaluationResult
from app.db.models.evaluation_run import EvaluationRun, RunStatus
from app.db.session import get_db
from app.schemas.evaluation import (
    ComparisonResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    MetricDeltaResponse
)
from app.tasks.evaluation_tasks import run_evaluation

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/run", response_model=EvaluationRunResponse, status_code=202)
def trigger_evaluation(payload: EvaluationRunRequest, db: Session = Depends(get_db)):
    """
    Creates an EvaluationRun record (PENDING), optionally pre-seeds placeholder
    EvaluationResult rows for the requested dataset_ids, then dispatches to Celery.
    Callers poll GET /evaluations/{id} until status == COMPLETED.
    """
    run = EvaluationRun(
        prompt_id=payload.prompt_id,
        status=RunStatus.PENDING,
        is_baseline=payload.is_baseline,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if payload.dataset_ids:
        placeholders = [
            EvaluationResult(run_id=run.id, dataset_id=did)
            for did in payload.dataset_ids
        ]
        db.add_all(placeholders)
        db.commit()

    run_evaluation.delay(run.id)

    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results))
        .filter(EvaluationRun.id == run.id)
        .first()
    )
    return run


@router.get("/compare", response_model=ComparisonResponse)
def compare(
    baseline_run_id: int = Query(..., description="Run marked as baseline"),
    candidate_run_id: int = Query(..., description="New run to evaluate"),
    db: Session = Depends(get_db),
):
    """
    Phase 4: Compare a candidate run against a baseline run.

    Returns per-metric deltas and a pass/fail verdict based on configured thresholds:
      - Similarity drop > 10%  → FAIL
      - Latency spike  > 50%   → FAIL
      - Cost spike     > 20%   → FAIL

    Thresholds are overridable via environment variables.
    HTTP 200 = comparison ran (check `passed` field).
    HTTP 422 = run not found or not yet completed.
    """
    baseline_run = db.get(EvaluationRun, baseline_run_id)
    candidate_run = db.get(EvaluationRun, candidate_run_id)

    if not baseline_run:
        raise HTTPException(status_code=404, detail=f"Baseline run {baseline_run_id} not found")
    if not candidate_run:
        raise HTTPException(status_code=404, detail=f"Candidate run {candidate_run_id} not found")

    if baseline_run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Baseline run {baseline_run_id} is not COMPLETED (status={baseline_run.status})",
        )
    if candidate_run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Candidate run {candidate_run_id} is not COMPLETED (status={candidate_run.status})",
        )

    baseline_results = (
        db.query(EvaluationResult)
        .filter(
            EvaluationResult.run_id == baseline_run_id,
            EvaluationResult.similarity_score.is_not(None),
        )
        .all()
    )
    candidate_results = (
        db.query(EvaluationResult)
        .filter(
            EvaluationResult.run_id == candidate_run_id,
            EvaluationResult.similarity_score.is_not(None),
        )
        .all()
    )

    if not baseline_results:
        raise HTTPException(
            status_code=422, detail="Baseline run has no completed result rows"
        )
    if not candidate_results:
        raise HTTPException(
            status_code=422, detail="Candidate run has no completed result rows"
        )

    report = compare_runs(
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
    )

    return ComparisonResponse(
        baseline_run_id=report.baseline_run_id,
        candidate_run_id=report.candidate_run_id,
        passed=report.passed,
        summary=report.summary,
        similarity=MetricDeltaResponse(**asdict(report.similarity)),
        latency=MetricDeltaResponse(**asdict(report.latency)),
        cost=MetricDeltaResponse(**asdict(report.cost)),
        per_dataset=report.per_dataset,
    )


@router.patch("/{run_id}/set-baseline", response_model=EvaluationRunResponse)
def set_baseline(run_id: int, db: Session = Depends(get_db)):
    """
    Mark a completed run as the new baseline for its prompt.
    Clears the baseline flag on any previously-marked run for the same prompt.
    """
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="EvaluationRun not found")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Only COMPLETED runs can be set as baseline (status={run.status})",
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
        .options(selectinload(EvaluationRun.evaluation_results))
        .filter(EvaluationRun.id == run_id)
        .first()
    )
    return run


@router.get("/{run_id}", response_model=EvaluationRunResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results))
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
        .options(selectinload(EvaluationRun.evaluation_results))
        .all()
    )
