import enum
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class ScoreStatus(str, enum.Enum):
    OK = "OK"            # score holds a valid value
    ERROR = "ERROR"      # the evaluator could not score; score is NULL and error_message says why


class EvaluationScore(Base):
    """
    One evaluator's verdict on one case result.

    Kept apart from EvaluationResult on purpose: the result holds execution facts
    (status, latency, the model's output); a score holds a judgement plus everything
    needed to interpret it later (evaluator name, version, config, kind). Adding a new
    evaluator adds rows here, never columns, and old rows stay readable after the
    evaluator implementation changes because the version is stored with the score.
    """

    __tablename__ = "evaluation_scores"
    __table_args__ = (
        UniqueConstraint("result_id", "evaluator_name", name="uq_evaluation_scores_result_evaluator"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    result_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_results.id", ondelete="CASCADE"), nullable=False, index=True
    )
    evaluator_name: Mapped[str] = mapped_column(String(50), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(20), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)       # "correctness" | "similarity"
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    config_hash: Mapped[str] = mapped_column(String(16), nullable=False)
    # Whether this score may block a release. Recorded per score so the decision rule
    # in force when the run was scored stays visible later.
    gating: Mapped[bool] = mapped_column(Boolean, nullable=False)

    status: Mapped[ScoreStatus] = mapped_column(Enum(ScoreStatus, native_enum=False, length=10), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)    # raw sub-scores / diagnostics
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    result: Mapped["EvaluationResult"] = relationship(back_populates="scores")
