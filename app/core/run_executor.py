"""
Failure-aware execution of one evaluation run.

Kept free of Celery imports so the lifecycle rules can be unit-tested with a fake
LLM and a plain SQLAlchemy session. app/tasks/evaluation_tasks.py is a thin wrapper
that adds Celery retry on top of this.

Rules
  * Every case ends SUCCEEDED or FAILED. A failure is never hidden in actual_output.
  * A failed case records a category and a diagnostic message; partial data that is
    still valid (e.g. model output when only scoring failed) is kept for debugging.
  * Run counters are recomputed from the result rows after every pass, never
    incremented, so a retry cannot double count.
  * COMPLETED only when every expected case SUCCEEDED. Otherwise PARTIAL (some
    succeeded) or FAILED (none did). A run being retried stays RUNNING.
  * Only INFRASTRUCTURE failures are worth retrying. A retry re-executes just the
    cases that did not SUCCEED, with their fields reset first.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx
from sqlalchemy.orm import Session, selectinload

from app.core.llm_gateway import call_llm
from app.core.metrics import compute_similarity
from app.db.models.dataset import Dataset
from app.db.models.evaluation_result import ErrorCategory, EvaluationResult, ResultStatus
from app.db.models.evaluation_run import EvaluationRun, RunStatus
from app.db.models.prompt import Prompt

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_LEN = 1000
_RETRYABLE_HTTP_CODES = {408, 429}


@dataclass
class ExecutionOutcome:
    run_id: int
    status: RunStatus
    expected_cases: int
    succeeded_cases: int
    failed_cases: int
    needs_retry: bool = False


# --------------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------------

def classify_llm_error(exc: BaseException) -> ErrorCategory:
    """Map an exception from the LLM call to an ErrorCategory."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code >= 500 or code in _RETRYABLE_HTTP_CODES:
            return ErrorCategory.INFRASTRUCTURE
        return ErrorCategory.MODEL_ERROR          # e.g. 404 model not found, 400 bad request
    if isinstance(exc, httpx.TransportError):     # timeouts, connection refused, DNS, ...
        return ErrorCategory.INFRASTRUCTURE
    if isinstance(exc, (ValueError, KeyError)):    # includes JSONDecodeError: malformed reply
        return ErrorCategory.MODEL_ERROR
    return ErrorCategory.UNKNOWN


def format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_MESSAGE_LEN]


def _fail(result: EvaluationResult, category: ErrorCategory, message: str) -> None:
    result.status = ResultStatus.FAILED
    result.error_category = category
    result.error_message = message[:MAX_ERROR_MESSAGE_LEN]
    # Never leave a score behind on a failed case: a stale value would be averaged.
    result.similarity_score = None


def _reset(result: EvaluationResult) -> None:
    result.error_category = None
    result.error_message = None
    result.latency_seconds = None
    result.token_cost = None
    result.similarity_score = None
    result.actual_output = None


# --------------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------------

def _execute_case(
    db: Session,
    run_id: int,
    prompt: Prompt,
    result: EvaluationResult,
    llm_fn: Callable,
    score_fn: Callable,
) -> None:
    dataset = db.get(Dataset, result.dataset_id)

    result.status = ResultStatus.RUNNING
    result.attempts = (result.attempts or 0) + 1
    _reset(result)
    db.commit()

    if dataset is None:
        _fail(result, ErrorCategory.UNKNOWN, f"dataset {result.dataset_id} not found")
        db.commit()
        return

    logger.info(
        "Run %s | case %s | attempt %s | model=%s | query='%s...'",
        run_id, dataset.id, result.attempts, prompt.model_name, dataset.input_query[:60],
    )

    try:
        llm_result = llm_fn(
            model_name=prompt.model_name,
            system_prompt=prompt.prompt_text,
            user_query=dataset.input_query,
        )
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded, not raised
        logger.warning("Run %s | case %s | LLM call failed: %s", run_id, dataset.id, exc)
        _fail(result, classify_llm_error(exc), format_error(exc))
        db.commit()
        return

    # Keep what the model produced even if scoring fails below.
    result.actual_output = llm_result.output
    result.latency_seconds = round(llm_result.latency_seconds, 4)
    result.token_cost = round(llm_result.token_cost, 8)

    try:
        score = score_fn(prediction=llm_result.output, reference=dataset.expected_output)
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"score {score!r} is not a finite number in [0, 1]")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Run %s | case %s | scoring failed: %s", run_id, dataset.id, exc)
        _fail(result, ErrorCategory.EVALUATOR_ERROR, format_error(exc))
        db.commit()
        return

    result.similarity_score = score
    result.status = ResultStatus.SUCCEEDED
    db.commit()


# --------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------

def _recount(run: EvaluationRun, results: list[EvaluationResult]) -> None:
    run.expected_cases = len(results)
    run.succeeded_cases = sum(1 for r in results if r.status == ResultStatus.SUCCEEDED)
    run.failed_cases = sum(1 for r in results if r.status == ResultStatus.FAILED)


def _failure_summary(run: EvaluationRun, results: list[EvaluationResult]) -> str:
    by_category = Counter(
        (r.error_category.value if r.error_category else "UNKNOWN")
        for r in results
        if r.status == ResultStatus.FAILED
    )
    detail = ", ".join(f"{k}={v}" for k, v in sorted(by_category.items()))
    unfinished = run.expected_cases - run.succeeded_cases - run.failed_cases
    text = f"{run.failed_cases} of {run.expected_cases} cases failed ({detail})"
    if unfinished > 0:
        text += f"; {unfinished} not executed"
    return text


def _finalize(run: EvaluationRun, results: list[EvaluationResult], note: str | None = None) -> None:
    """Move a run to its terminal status from the current result rows."""
    _recount(run, results)
    now = datetime.now(timezone.utc)
    run.completed_at = now
    if run.expected_cases > 0 and run.succeeded_cases == run.expected_cases:
        run.status = RunStatus.COMPLETED
        run.error_message = None
        return
    run.status = RunStatus.PARTIAL if run.succeeded_cases > 0 else RunStatus.FAILED
    summary = _failure_summary(run, results)
    run.error_message = f"{note}: {summary}" if note else summary


def mark_run_failed(db: Session, run_id: int, message: str) -> None:
    """
    Terminal handling for an unexpected error (e.g. database trouble) once retries
    are exhausted. Counters are recomputed from whatever rows were saved, so the
    partial results stay visible and the run can never look COMPLETED.
    """
    db.rollback()
    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results))
        .filter(EvaluationRun.id == run_id)
        .first()
    )
    if run is None:
        return
    _finalize(run, sorted(run.evaluation_results, key=lambda r: r.id), note=message[:300])
    db.commit()


def execute_run(
    db: Session,
    run_id: int,
    *,
    llm_fn: Callable = call_llm,
    score_fn: Callable = compute_similarity,
    final_attempt: bool = True,
) -> ExecutionOutcome | None:
    """
    Execute (or resume) a run. Returns None if the run does not exist.

    final_attempt=False means the caller can retry: if INFRASTRUCTURE failures remain
    the run is left RUNNING and the outcome has needs_retry=True. Re-calling this
    function resumes only the cases that did not succeed.
    """
    run = (
        db.query(EvaluationRun)
        .options(selectinload(EvaluationRun.evaluation_results))
        .filter(EvaluationRun.id == run_id)
        .first()
    )
    if run is None:
        logger.error("EvaluationRun %s not found", run_id)
        return None

    results = sorted(run.evaluation_results, key=lambda r: r.id)
    prompt = db.get(Prompt, run.prompt_id)

    if prompt is None or not results:
        reason = f"prompt {run.prompt_id} not found" if prompt is None else "run has no test cases"
        logger.error("Run %s cannot start: %s", run_id, reason)
        _finalize(run, results, note=reason)
        # Nothing can be executed, whatever earlier attempts left in the counters.
        run.status = RunStatus.FAILED
        run.error_message = reason
        db.commit()
        return _outcome(run)

    run.status = RunStatus.RUNNING
    if run.started_at is None:
        run.started_at = datetime.now(timezone.utc)
    run.completed_at = None
    run.error_message = None
    run.expected_cases = len(results)
    db.commit()

    for result in results:
        if result.status == ResultStatus.SUCCEEDED:
            continue  # already done in an earlier attempt: never redo, never recount
        _execute_case(db, run_id, prompt, result, llm_fn, score_fn)

    _recount(run, results)
    retryable = any(
        r.status == ResultStatus.FAILED and r.error_category == ErrorCategory.INFRASTRUCTURE
        for r in results
    )

    if retryable and not final_attempt and run.succeeded_cases < run.expected_cases:
        run.status = RunStatus.RUNNING  # stays RUNNING: pollers must not see FAILED mid-retry
        run.error_message = f"retrying: {_failure_summary(run, results)}"
        db.commit()
        logger.warning("Run %s will be retried: %s", run_id, run.error_message)
        outcome = _outcome(run)
        outcome.needs_retry = True
        return outcome

    _finalize(run, results)
    db.commit()
    logger.info(
        "Run %s %s: %d/%d succeeded, %d failed",
        run_id, run.status.value, run.succeeded_cases, run.expected_cases, run.failed_cases,
    )
    return _outcome(run)


def _outcome(run: EvaluationRun) -> ExecutionOutcome:
    return ExecutionOutcome(
        run_id=run.id,
        status=run.status,
        expected_cases=run.expected_cases,
        succeeded_cases=run.succeeded_cases,
        failed_cases=run.failed_cases,
    )
