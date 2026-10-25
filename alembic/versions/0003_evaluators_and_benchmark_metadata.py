"""Evaluator scores in their own table; benchmark identity/metadata on datasets.

Schema changes
  datasets            + benchmark, benchmark_version, case_key, task_type, category,
                        difficulty, suites, eval_spec, content_hash
                        UNIQUE (benchmark, benchmark_version, case_key)
  evaluation_scores   new: one row per (result, evaluator) with evaluator name,
                        version, kind, config, gating flag, score, raw details

Data backfill
  Every SUCCEEDED legacy result gets a text_similarity v1 score row copied from its
  similarity_score. That is exactly how the old pipeline scored it (same formula, no
  config), so legacy runs remain comparable with each other and with new runs of
  legacy-style cases.

Revision ID: 0003
Revises: 0002
"""
import hashlib
import json
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

# sha256 of canonical JSON of the empty config; must equal app.core.evaluators.base.config_hash({}).
_EMPTY_CONFIG_HASH = hashlib.sha256(json.dumps({}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def upgrade() -> None:
    op.add_column("datasets", sa.Column("benchmark", sa.String(100), nullable=True))
    op.add_column("datasets", sa.Column("benchmark_version", sa.String(50), nullable=True))
    op.add_column("datasets", sa.Column("case_key", sa.String(100), nullable=True))
    op.add_column("datasets", sa.Column("task_type", sa.String(50), nullable=True))
    op.add_column("datasets", sa.Column("category", sa.String(50), nullable=True))
    op.add_column("datasets", sa.Column("difficulty", sa.String(20), nullable=True))
    op.add_column("datasets", sa.Column("suites", sa.JSON(), nullable=True))
    op.add_column("datasets", sa.Column("eval_spec", sa.JSON(), nullable=True))
    op.add_column("datasets", sa.Column("content_hash", sa.String(64), nullable=True))
    with op.batch_alter_table("datasets") as batch:
        batch.create_unique_constraint(
            "uq_datasets_benchmark_case", ["benchmark", "benchmark_version", "case_key"]
        )

    op.create_table(
        "evaluation_scores",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("result_id", sa.Integer(), sa.ForeignKey("evaluation_results.id", ondelete="CASCADE"), nullable=False),
        sa.Column("evaluator_name", sa.String(50), nullable=False),
        sa.Column("evaluator_version", sa.String(20), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("config_hash", sa.String(16), nullable=False),
        sa.Column("gating", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("result_id", "evaluator_name", name="uq_evaluation_scores_result_evaluator"),
    )
    op.create_index("ix_evaluation_scores_id", "evaluation_scores", ["id"])
    op.create_index("ix_evaluation_scores_result_id", "evaluation_scores", ["result_id"])

    # -- backfill: legacy successful results -> text_similarity v1 ----------------------
    bind = op.get_bind()
    legacy = bind.execute(sa.text(
        "SELECT id, similarity_score FROM evaluation_results "
        "WHERE status = 'SUCCEEDED' AND similarity_score IS NOT NULL"
    )).fetchall()
    if legacy:
        scores = sa.table(
            "evaluation_scores",
            sa.column("result_id", sa.Integer), sa.column("evaluator_name", sa.String),
            sa.column("evaluator_version", sa.String), sa.column("kind", sa.String),
            sa.column("config", sa.JSON), sa.column("config_hash", sa.String),
            sa.column("gating", sa.Boolean), sa.column("status", sa.String),
            sa.column("score", sa.Float), sa.column("created_at", sa.DateTime(timezone=True)),
        )
        now = datetime.now(timezone.utc)
        op.bulk_insert(scores, [
            {
                "result_id": rid, "evaluator_name": "text_similarity", "evaluator_version": "1",
                "kind": "similarity", "config": {}, "config_hash": _EMPTY_CONFIG_HASH,
                "gating": True, "status": "OK", "score": float(sim), "created_at": now,
            }
            for rid, sim in legacy
        ])


def downgrade() -> None:
    op.drop_table("evaluation_scores")
    with op.batch_alter_table("datasets") as batch:
        batch.drop_constraint("uq_datasets_benchmark_case", type_="unique")
        for col in ("content_hash", "eval_spec", "suites", "difficulty", "category",
                    "task_type", "case_key", "benchmark_version", "benchmark"):
            batch.drop_column(col)
