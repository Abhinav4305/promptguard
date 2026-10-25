"""Baseline: the schema as originally created by Base.metadata.create_all().

Databases that already exist (created by the old startup hook) are NOT re-created:
app/db/migrate.py stamps them at this revision, then applies 0002 and later.

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version_tag", sa.String(100), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_prompts_id", "prompts", ["id"])

    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("input_query", sa.Text(), nullable=False),
        sa.Column("expected_output", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_datasets_id", "datasets", ["id"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prompt_id", sa.Integer(), sa.ForeignKey("prompts.id"), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", name="runstatus"),
            nullable=False,
        ),
        sa.Column("is_baseline", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_evaluation_runs_id", "evaluation_runs", ["id"])

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("evaluation_runs.id"), nullable=False),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("latency_seconds", sa.Float(), nullable=True),
        sa.Column("token_cost", sa.Float(), nullable=True),
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("actual_output", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_evaluation_results_id", "evaluation_results", ["id"])


def downgrade() -> None:
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_table("datasets")
    op.drop_table("prompts")
    sa.Enum(name="runstatus").drop(op.get_bind(), checkfirst=True)
