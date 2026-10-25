"""Failure-aware runs: per-case status/error fields, run counters, PARTIAL status.

Schema changes
  evaluation_runs     + started_at, expected_cases, succeeded_cases, failed_cases,
                        error_message; RunStatus gains PARTIAL
  evaluation_results  + status, error_category, error_message, attempts;
                        UNIQUE (run_id, dataset_id)

Data backfill for rows written by the old code (which recorded failures as
actual_output = 'ERROR: ...' with NULL metrics and still marked the run COMPLETED):
  * result 'ERROR: ...' rows            -> FAILED / UNKNOWN, message copied from the output
  * result rows with a similarity score -> SUCCEEDED
  * rows of finished runs with neither  -> FAILED / UNKNOWN ("no result recorded")
  * duplicate (run_id, dataset_id) rows -> keep one (prefer SUCCEEDED, then newest);
                                           required for the unique constraint
  * run counters are recomputed from the result rows
  * a legacy COMPLETED run that contains failed cases is corrected to PARTIAL (or
    FAILED when nothing succeeded) and its is_baseline flag is cleared, so a run
    that only looked successful can no longer be used as a baseline

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # PostgreSQL enums need ALTER TYPE; ADD VALUE must be committed before the new
    # value is used, hence the autocommit block. SQLite stores the enum as VARCHAR.
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE runstatus ADD VALUE IF NOT EXISTS 'PARTIAL'")

    # -- evaluation_runs ------------------------------------------------------
    op.add_column("evaluation_runs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "evaluation_runs",
        sa.Column("expected_cases", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column("succeeded_cases", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column("failed_cases", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("evaluation_runs", sa.Column("error_message", sa.Text(), nullable=True))

    # -- evaluation_results -----------------------------------------------------
    op.add_column(
        "evaluation_results",
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
    )
    op.add_column("evaluation_results", sa.Column("error_category", sa.String(30), nullable=True))
    op.add_column("evaluation_results", sa.Column("error_message", sa.Text(), nullable=True))
    op.add_column(
        "evaluation_results",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )

    # -- backfill result rows (order matters: ERROR rows win over stale scores) -----
    op.execute(
        """
        UPDATE evaluation_results
           SET status = 'FAILED', error_category = 'UNKNOWN',
               error_message = actual_output, attempts = 1
         WHERE actual_output LIKE 'ERROR:%'
        """
    )
    op.execute(
        """
        UPDATE evaluation_results
           SET status = 'SUCCEEDED', attempts = 1
         WHERE status = 'PENDING' AND similarity_score IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE evaluation_results
           SET status = 'FAILED', error_category = 'UNKNOWN',
               error_message = 'legacy row: no result was recorded', attempts = 1
         WHERE status = 'PENDING'
           AND run_id IN (SELECT id FROM evaluation_runs WHERE status IN ('COMPLETED', 'FAILED'))
        """
    )

    # -- remove duplicate (run_id, dataset_id) rows so the constraint can be added ----
    op.execute(
        """
        DELETE FROM evaluation_results
         WHERE id IN (
               SELECT id FROM (
                   SELECT id,
                          ROW_NUMBER() OVER (
                              PARTITION BY run_id, dataset_id
                              ORDER BY CASE WHEN status = 'SUCCEEDED' THEN 0 ELSE 1 END, id DESC
                          ) AS rn
                     FROM evaluation_results
               ) ranked
                WHERE rn > 1)
        """
    )

    with op.batch_alter_table("evaluation_results") as batch:
        batch.create_unique_constraint(
            "uq_evaluation_results_run_dataset", ["run_id", "dataset_id"]
        )

    # -- recompute run counters from the result rows --------------------------------
    op.execute(
        """
        UPDATE evaluation_runs
           SET expected_cases = (SELECT COUNT(*) FROM evaluation_results r
                                  WHERE r.run_id = evaluation_runs.id),
               succeeded_cases = (SELECT COUNT(*) FROM evaluation_results r
                                   WHERE r.run_id = evaluation_runs.id AND r.status = 'SUCCEEDED'),
               failed_cases = (SELECT COUNT(*) FROM evaluation_results r
                                WHERE r.run_id = evaluation_runs.id AND r.status = 'FAILED')
        """
    )

    # -- correct legacy runs that were marked COMPLETED despite failed cases --------
    # Two plain UPDATEs rather than one CASE: on PostgreSQL a CASE expression is typed
    # text and cannot be assigned to the runstatus enum column without a cast.
    false = "false" if bind.dialect.name == "postgresql" else "0"
    for new_status, succeeded_filter in (("PARTIAL", "succeeded_cases > 0"), ("FAILED", "succeeded_cases = 0")):
        op.execute(
            f"""
            UPDATE evaluation_runs
               SET status = '{new_status}',
                   is_baseline = {false},
                   error_message = 'legacy run: ' || failed_cases || ' of ' || expected_cases
                                   || ' cases failed but the run was recorded as COMPLETED'
             WHERE status = 'COMPLETED' AND failed_cases > 0 AND {succeeded_filter}
            """
        )


def downgrade() -> None:
    bind = op.get_bind()

    # PARTIAL cannot be removed from a PostgreSQL enum; map it to FAILED instead and
    # leave the (now unused) enum value in place.
    op.execute("UPDATE evaluation_runs SET status = 'FAILED' WHERE status = 'PARTIAL'")

    with op.batch_alter_table("evaluation_results") as batch:
        batch.drop_constraint("uq_evaluation_results_run_dataset", type_="unique")
        batch.drop_column("attempts")
        batch.drop_column("error_message")
        batch.drop_column("error_category")
        batch.drop_column("status")

    with op.batch_alter_table("evaluation_runs") as batch:
        batch.drop_column("error_message")
        batch.drop_column("failed_cases")
        batch.drop_column("succeeded_cases")
        batch.drop_column("expected_cases")
        batch.drop_column("started_at")
