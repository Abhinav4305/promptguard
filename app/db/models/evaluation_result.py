import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class ResultStatus(str, enum.Enum):
    """
    Execution status of one test case within one run.

    SUCCEEDED means "the model was called and the output was scored" — it says
    nothing about whether the answer was good. Quality is judged by comparison.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ErrorCategory(str, enum.Enum):
    """
    Why a case FAILED. Execution failures are deliberately separate from
    quality regressions, which are never an error category.
    """

    INFRASTRUCTURE = "INFRASTRUCTURE"      # timeout, connection refused, HTTP 5xx/429 (retryable)
    MODEL_ERROR = "MODEL_ERROR"            # gateway rejected the request or response was malformed
    EVALUATOR_ERROR = "EVALUATOR_ERROR"    # model output obtained, but scoring raised
    UNKNOWN = "UNKNOWN"


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (
        # One row per (run, case). Duplicates would silently skew every average.
        UniqueConstraint("run_id", "dataset_id", name="uq_evaluation_results_run_dataset"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)

    # -- execution metadata (separate from model output and from scores) --------
    status: Mapped[ResultStatus] = mapped_column(
        Enum(ResultStatus, native_enum=False, length=20),
        default=ResultStatus.PENDING,
        server_default=ResultStatus.PENDING.value,
        nullable=False,
    )
    error_category: Mapped[ErrorCategory | None] = mapped_column(
        Enum(ErrorCategory, native_enum=False, length=30), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )

    # -- measurements -----------------------------------------------------------
    latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- model-generated output (NULL when the model call itself failed) ----------
    actual_output: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    evaluation_run: Mapped["EvaluationRun"] = relationship(
        back_populates="evaluation_results"
    )
    dataset: Mapped["Dataset"] = relationship(back_populates="evaluation_results")
