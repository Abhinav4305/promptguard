import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class RunStatus(str, enum.Enum):
    """
    Run lifecycle.

      PENDING   -> created, not yet picked up by a worker
      RUNNING   -> worker is executing cases (also held between retry attempts)
      COMPLETED -> every expected case executed successfully (failed_cases == 0)
      PARTIAL   -> run ended with some cases failed; results are kept for debugging
                   but the run is NOT comparable and cannot become a baseline
      FAILED    -> run ended with no successful cases (or could not start)

    NOTE: COMPLETED means "all cases executed", never "quality is acceptable".
    Quality verdicts only exist in a comparison (see app/core/comparator.py).
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


# Statuses after which a run will never change again.
TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.PARTIAL, RunStatus.FAILED})


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompts.id"), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.PENDING, nullable=False
    )
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Set on the first attempt only, so retries do not move the start time.
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Time the run reached a terminal status (COMPLETED, PARTIAL or FAILED).
    # Kept under its original column name for API/data compatibility.
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Case counters. They are recomputed from the result rows on every update,
    # never incremented, so retries cannot double count.
    expected_cases: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    succeeded_cases: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    failed_cases: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    # Run-level diagnostic (why the run is PARTIAL/FAILED). NULL for clean runs.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    prompt: Mapped["Prompt"] = relationship(back_populates="evaluation_runs")
    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="evaluation_run"
    )

    @property
    def completed_cases(self) -> int:
        """Cases that reached a terminal state (succeeded + failed)."""
        return self.succeeded_cases + self.failed_cases
