import logging

from app.core.run_executor import execute_run, mark_run_failed
from app.db.session import SessionLocal
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

RETRY_COUNTDOWN_SECONDS = 10


@celery_app.task(name="run_evaluation", queue="evaluations", bind=True, max_retries=2)
def run_evaluation(self, run_id: int) -> dict:
    """
    Thin Celery wrapper around app.core.run_executor.execute_run.

    All lifecycle rules (per-case status, counters, COMPLETED/PARTIAL/FAILED) live in
    the executor. This task only decides whether to retry:
      * infrastructure failures on some cases  -> retry the unfinished cases
      * an unexpected error (e.g. database)    -> retry the run
    The run stays RUNNING between attempts and only reaches a terminal status on the
    last one, so pollers never see a transient FAILED.
    """
    logger.info("Starting evaluation run_id=%s (attempt %s)", run_id, self.request.retries + 1)

    final_attempt = self.request.retries >= self.max_retries
    db = SessionLocal()
    try:
        try:
            outcome = execute_run(db, run_id, final_attempt=final_attempt)
        except Exception as exc:
            logger.exception("Run %s hit an unexpected error: %s", run_id, exc)
            db.rollback()
            if final_attempt:
                mark_run_failed(db, run_id, f"unexpected error: {type(exc).__name__}: {exc}")
                raise
            raise self.retry(exc=exc, countdown=RETRY_COUNTDOWN_SECONDS)

        if outcome is None:
            return {"error": "run not found"}

        if outcome.needs_retry:
            raise self.retry(
                exc=RuntimeError(f"run {run_id}: infrastructure failures, retrying unfinished cases"),
                countdown=RETRY_COUNTDOWN_SECONDS,
            )

        return {
            "run_id": run_id,
            "status": outcome.status.value,
            # kept for backward compatibility: number of cases scored successfully
            "results_count": outcome.succeeded_cases,
            "expected_cases": outcome.expected_cases,
            "succeeded_cases": outcome.succeeded_cases,
            "failed_cases": outcome.failed_cases,
        }
    finally:
        db.close()
