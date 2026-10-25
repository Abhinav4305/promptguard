from datetime import datetime
from sqlalchemy import JSON, DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class Dataset(Base):
    """
    One test case. Rows created by the legacy endpoints have only input_query and
    expected_output; they are scored with text_similarity alone, as before.

    Rows imported from a versioned benchmark (POST /datasets/import) also carry a
    stable case_key, task metadata and an evaluator spec. A case definition is
    immutable for a given (benchmark, benchmark_version): content_hash proves it.
    """

    __tablename__ = "datasets"
    __table_args__ = (
        # NULLs are distinct, so any number of legacy rows (no case_key) can coexist.
        UniqueConstraint("benchmark", "benchmark_version", "case_key", name="uq_datasets_benchmark_case"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    input_query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_output: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # -- benchmark identity and metadata (all NULL for legacy rows) ----------------
    benchmark: Mapped[str | None] = mapped_column(String(100), nullable=True)
    benchmark_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    case_key: Mapped[str | None] = mapped_column(String(100), nullable=True)   # stable, human-readable ID
    task_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    difficulty: Mapped[str | None] = mapped_column(String(20), nullable=True)
    suites: Mapped[list | None] = mapped_column(JSON, nullable=True)           # e.g. ["pr", "full"]
    # How this case is scored: [{"name": ..., "config": {...}, "gate": ...}]. NULL = text_similarity.
    eval_spec: Mapped[list | None] = mapped_column(JSON, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="dataset"
    )
