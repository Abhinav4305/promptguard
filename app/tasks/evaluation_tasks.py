import logging
from datetime import datetime, timezone

from sqlalchemy.orm import selectinload

from app.core.llm_gateway import call_llm
from app.core.metrics import compute_similarity
from app.db.models.dataset import Dataset
from app.db.models.evaluation_result import EvaluationResult
from app.db.models.evaluation_run import EvaluationRun, RunStatus
from app.db.models.prompt import Prompt
from app.db.session import SessionLocal
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="run_evaluation", queue="evaluations", bind=True, max_retries=2)
def run_evaluation(self, run_id: int) -> dict:
    """
    Phase 3: Gemini evaluation loop via LiteLLM.

    For each dataset row attached to this run:
      1. Call Gemini via LiteLLM gateway (captures latency + token cost).
      2. Compute similarity between actual output and expected output.
      3. Update the pre-seeded EvaluationResult row with real metrics.

    The run status progresses: PENDING -> RUNNING -> COMPLETED (or FAILED).
    """
    logger.info("Starting evaluation run_id=%s", run_id)

    db = SessionLocal()
    try:
        # -- Load the run WITH its placeholder results eagerly ----------------
        # Must use selectinload here — db.get() does not load relationships,
        # so run.evaluation_results would be an empty list without this query.
        run = (
            db.query(EvaluationRun)
            .options(selectinload(EvaluationRun.evaluation_results))
            .filter(EvaluationRun.id == run_id)
            .first()
        )
        if not run:
            logger.error("EvaluationRun %s not found", run_id)
            return {"error": "run not found"}

        prompt: Prompt = db.get(Prompt, run.prompt_id)
        if not prompt:
            logger.error("Prompt %s not found for run %s", run.prompt_id, run_id)
            run.status = RunStatus.FAILED
            db.commit()
            return {"error": "prompt not found"}

        run.status = RunStatus.RUNNING
        db.commit()

        # -- Resolve dataset IDs from pre-seeded placeholder rows -------------
        # The route pre-seeds EvaluationResult rows with only run_id + dataset_id
        # (all metrics NULL). We update them in-place rather than inserting new
        # rows, so there are never orphaned null rows in the table.
        placeholder_map: dict[int, EvaluationResult] = {
            r.dataset_id: r for r in run.evaluation_results
        }
        dataset_ids = list(placeholder_map.keys())

        # If no placeholders were pre-seeded, fall back to ALL datasets (dev convenience)
        if not dataset_ids:
            datasets = db.query(Dataset).all()
        else:
            datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()

        if not datasets:
            logger.warning("Run %s has no datasets — completing with 0 results", run_id)
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
            return {"run_id": run_id, "status": "COMPLETED", "results_count": 0}

        # -- Per-dataset Gemini call + metric computation ---------------------
        results_saved = 0
        for dataset in datasets:
            logger.info(
                "Run %s | dataset %s | model=%s | query='%s...'",
                run_id,
                dataset.id,
                prompt.model_name,
                dataset.input_query[:60],
            )

            # Reuse the placeholder row if it exists; otherwise create a fresh one
            result = placeholder_map.get(dataset.id) or EvaluationResult(
                run_id=run_id, dataset_id=dataset.id
            )

            try:
                llm_result = call_llm(
                    model_name=prompt.model_name,
                    system_prompt=prompt.prompt_text,
                    user_query=dataset.input_query,
                )
                similarity = compute_similarity(
                    prediction=llm_result.output,
                    reference=dataset.expected_output,
                )
                result.latency_seconds = round(llm_result.latency_seconds, 4)
                result.token_cost = round(llm_result.token_cost, 8)
                result.similarity_score = similarity
                result.actual_output = llm_result.output

            except Exception as call_exc:
                logger.exception(
                    "Gemini call failed for run=%s dataset=%s: %s",
                    run_id,
                    dataset.id,
                    call_exc,
                )
                result.actual_output = f"ERROR: {call_exc}"
                # Leave metrics as NULL so comparator excludes this row cleanly

            db.add(result)
            db.commit()

            if result.similarity_score is not None:
                results_saved += 1
                logger.info(
                    "Run %s | dataset %s -> similarity=%.4f latency=%.3fs cost=$%.6f",
                    run_id,
                    dataset.id,
                    result.similarity_score,
                    result.latency_seconds,
                    result.token_cost,
                )

        # -- Mark run complete ------------------------------------------------
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(
            "Run %s COMPLETED — %d/%d results saved",
            run_id,
            results_saved,
            len(datasets),
        )
        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "results_count": results_saved,
        }

    except Exception as exc:
        logger.exception("Run %s FAILED: %s", run_id, exc)
        db.rollback()
        run = db.get(EvaluationRun, run_id)
        if run:
            run.status = RunStatus.FAILED
            db.commit()
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()
