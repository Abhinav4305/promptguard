"""
Alembic migrations, exercised on SQLite files.

Covered: fresh install, upgrading data written by the OLD code (which hid failures
behind COMPLETED), adoption of a database created by the old create_all() hook, and
drift between the models and the migrations.

NOT covered here: PostgreSQL-only statements (ALTER TYPE ... ADD VALUE). Run
`TEST_POSTGRES_URL=postgresql://... pytest tests/test_migrations.py` to include a
real-Postgres upgrade test.
"""
import os
import sqlite3

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from app.db.migrate import alembic_config, run_migrations
from app.db.session import Base


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture()
def cfg(db_path):
    return alembic_config(f"sqlite:///{db_path}")


def _legacy_database(cfg, db_path):
    """Schema 0001 plus rows exactly as the old code wrote them."""
    command.upgrade(cfg, "0001")
    c = sqlite3.connect(db_path)
    c.executescript(
        """
        INSERT INTO prompts(id,version_tag,prompt_text,model_name) VALUES (1,'v1','p','llama3.2');
        INSERT INTO datasets(id,input_query,expected_output) VALUES (1,'q1','a1'),(2,'q2','a2'),(3,'q3','a3');

        -- run 1: marked COMPLETED although two cases failed (one swallowed error, one unfilled placeholder)
        INSERT INTO evaluation_runs(id,prompt_id,status,is_baseline) VALUES (1,1,'COMPLETED',1);
        INSERT INTO evaluation_results(run_id,dataset_id,similarity_score,latency_seconds,token_cost,actual_output)
               VALUES (1,1,0.9,1.0,0.0,'ok');
        INSERT INTO evaluation_results(run_id,dataset_id,actual_output) VALUES (1,2,'ERROR: connection refused');
        INSERT INTO evaluation_results(run_id,dataset_id) VALUES (1,3);

        -- run 2: genuinely clean, but with an orphan duplicate placeholder for case 1
        INSERT INTO evaluation_runs(id,prompt_id,status,is_baseline) VALUES (2,1,'COMPLETED',0);
        INSERT INTO evaluation_results(run_id,dataset_id) VALUES (2,1);
        INSERT INTO evaluation_results(run_id,dataset_id,similarity_score,latency_seconds,token_cost,actual_output)
               VALUES (2,1,0.8,1.0,0.0,'ok');
        INSERT INTO evaluation_results(run_id,dataset_id,similarity_score,latency_seconds,token_cost,actual_output)
               VALUES (2,2,0.7,1.0,0.0,'ok');

        -- run 3: still PENDING when the upgrade happens
        INSERT INTO evaluation_runs(id,prompt_id,status,is_baseline) VALUES (3,1,'PENDING',0);
        INSERT INTO evaluation_results(run_id,dataset_id) VALUES (3,1);
        """
    )
    c.commit()
    c.close()


def test_fresh_database_upgrades_to_head_with_new_columns(cfg, db_path):
    command.upgrade(cfg, "head")
    insp = inspect(create_engine(f"sqlite:///{db_path}"))

    run_cols = {c["name"] for c in insp.get_columns("evaluation_runs")}
    assert {"started_at", "expected_cases", "succeeded_cases", "failed_cases", "error_message"} <= run_cols
    res_cols = {c["name"] for c in insp.get_columns("evaluation_results")}
    assert {"status", "error_category", "error_message", "attempts"} <= res_cols
    uniques = [u["name"] for u in insp.get_unique_constraints("evaluation_results")]
    assert "uq_evaluation_results_run_dataset" in uniques


def test_models_and_migrations_have_not_drifted(cfg, db_path):
    command.upgrade(cfg, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    # Ignore pure type-rendering noise; any added/removed table, column, index or
    # constraint means a model change was made without a migration.
    # (compare_metadata nests type/default "modify_*" diffs in lists; tuples are structural.)
    structural = [d for d in diff if isinstance(d, tuple) and d[0] in {
        "add_table", "remove_table", "add_column", "remove_column", "add_index", "remove_index",
        "add_constraint", "remove_constraint", "add_fk", "remove_fk",
    }]
    assert structural == []


def test_legacy_rows_are_reclassified_and_counters_backfilled(cfg, db_path):
    _legacy_database(cfg, db_path)
    command.upgrade(cfg, "head")

    c = sqlite3.connect(db_path)
    runs = {r[0]: r[1:] for r in c.execute(
        "SELECT id,status,is_baseline,expected_cases,succeeded_cases,failed_cases FROM evaluation_runs")}
    # Run 1 only LOOKED successful: now PARTIAL, no longer a baseline.
    assert runs[1] == ("PARTIAL", 0, 3, 1, 2)
    # Run 2 was fine; its orphan duplicate row is gone, so it has 2 cases.
    assert runs[2] == ("COMPLETED", 0, 2, 2, 0)
    # Run 3 had not run yet: untouched, its placeholder stays PENDING.
    assert runs[3] == ("PENDING", 0, 1, 0, 0)

    results = {(r[0], r[1]): r[2:] for r in c.execute(
        "SELECT run_id,dataset_id,status,error_category,error_message,attempts FROM evaluation_results")}
    assert results[(1, 1)][0] == "SUCCEEDED"
    assert results[(1, 2)] == ("FAILED", "UNKNOWN", "ERROR: connection refused", 1)
    assert results[(1, 3)][0] == "FAILED" and "no result was recorded" in results[(1, 3)][2]
    assert results[(2, 1)][0] == "SUCCEEDED"            # the scored row survived, not the empty one
    assert results[(3, 1)][0] == "PENDING"
    assert c.execute("SELECT COUNT(*) FROM evaluation_results WHERE run_id=2").fetchone()[0] == 2
    assert "legacy run: 2 of 3 cases failed" in c.execute(
        "SELECT error_message FROM evaluation_runs WHERE id=1").fetchone()[0]


def test_legacy_scores_are_backfilled_as_text_similarity_v1(cfg, db_path):
    from app.core.evaluators.base import config_hash

    _legacy_database(cfg, db_path)
    command.upgrade(cfg, "head")
    c = sqlite3.connect(db_path)
    rows = c.execute(
        """SELECT r.run_id, r.dataset_id, s.evaluator_name, s.evaluator_version, s.kind,
                  s.config_hash, s.gating, s.status, s.score
             FROM evaluation_scores s JOIN evaluation_results r ON r.id = s.result_id
            ORDER BY r.run_id, r.dataset_id"""
    ).fetchall()
    # Only SUCCEEDED legacy results get a score; failed/pending ones must not.
    assert [(r[0], r[1]) for r in rows] == [(1, 1), (2, 1), (2, 2)]
    for _, _, name, version, kind, chash, gating, status, _score in rows:
        assert (name, version, kind, gating, status) == ("text_similarity", "1", "similarity", 1, "OK")
        assert chash == config_hash({})           # same identity a new default-spec run records
    assert [r[8] for r in rows] == [0.9, 0.8, 0.7]


def test_benchmark_case_identity_is_unique_but_legacy_rows_are_unconstrained(cfg, db_path):
    command.upgrade(cfg, "head")
    c = sqlite3.connect(db_path)
    # any number of legacy rows (no benchmark identity) is fine
    for _ in range(3):
        c.execute("INSERT INTO datasets(input_query, expected_output) VALUES ('q', 'a')")
    c.execute("INSERT INTO datasets(input_query, expected_output, benchmark, benchmark_version, case_key) "
              "VALUES ('q','a','core','1','case-1')")
    # same key in another version is a different case
    c.execute("INSERT INTO datasets(input_query, expected_output, benchmark, benchmark_version, case_key) "
              "VALUES ('q','a','core','2','case-1')")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO datasets(input_query, expected_output, benchmark, benchmark_version, case_key) "
                  "VALUES ('q2','a2','core','1','case-1')")


def test_unique_constraint_blocks_duplicate_results_after_upgrade(cfg, db_path):
    _legacy_database(cfg, db_path)
    command.upgrade(cfg, "head")
    c = sqlite3.connect(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO evaluation_results(run_id,dataset_id) VALUES (2,1)")


def test_downgrade_then_upgrade_round_trips(cfg, db_path):
    _legacy_database(cfg, db_path)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0001")
    insp = inspect(create_engine(f"sqlite:///{db_path}"))
    assert "expected_cases" not in {c["name"] for c in insp.get_columns("evaluation_runs")}
    command.upgrade(cfg, "head")                       # must be repeatable


def test_run_migrations_adopts_a_database_created_by_the_old_create_all(tmp_path):
    """Old deployments have tables but no alembic_version. They must be stamped, not re-created."""
    path = tmp_path / "old_deployment.db"
    url = f"sqlite:///{path}"
    command.upgrade(alembic_config(url), "0001")                 # build the legacy schema...
    c = sqlite3.connect(path)
    c.execute("DROP TABLE alembic_version")                      # ...and make it look un-versioned
    c.execute("INSERT INTO prompts(id,version_tag,prompt_text,model_name) VALUES (1,'v1','p','m')")
    c.commit()
    c.close()

    engine = create_engine(url)
    run_migrations(engine)                                       # must not raise "table already exists"
    run_migrations(engine)                                       # and must be idempotent

    insp = inspect(engine)
    assert "expected_cases" in {col["name"] for col in insp.get_columns("evaluation_runs")}
    c = sqlite3.connect(path)
    head = ScriptDirectory.from_config(alembic_config(url)).get_current_head()
    assert c.execute("SELECT version_num FROM alembic_version").fetchone()[0] == head
    assert c.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 1       # data kept
    engine.dispose()


def test_run_migrations_on_empty_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    run_migrations(engine)
    assert {"prompts", "datasets", "evaluation_runs", "evaluation_results"} <= set(inspect(engine).get_table_names())
    engine.dispose()


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="set TEST_POSTGRES_URL to run against Postgres")
def test_postgres_upgrade_adds_partial_status_and_backfills():
    url = os.environ["TEST_POSTGRES_URL"]
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    cfg = alembic_config(url)
    command.upgrade(cfg, "0001")
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO prompts(id,version_tag,prompt_text,model_name) VALUES (1,'v','p','m')")
        conn.exec_driver_sql("INSERT INTO datasets(id,input_query,expected_output) VALUES (1,'q','a'),(2,'q2','a2')")
        conn.exec_driver_sql("INSERT INTO evaluation_runs(id,prompt_id,status,is_baseline) VALUES (1,1,'COMPLETED',true)")
        conn.exec_driver_sql("INSERT INTO evaluation_results(run_id,dataset_id,similarity_score,latency_seconds,token_cost,actual_output) VALUES (1,1,0.9,1,0,'ok')")
        conn.exec_driver_sql("INSERT INTO evaluation_results(run_id,dataset_id,actual_output) VALUES (1,2,'ERROR: x')")
        # run 2: COMPLETED although every case failed -> must become FAILED, not PARTIAL
        conn.exec_driver_sql("INSERT INTO evaluation_runs(id,prompt_id,status,is_baseline) VALUES (2,1,'COMPLETED',true)")
        conn.exec_driver_sql("INSERT INTO evaluation_results(run_id,dataset_id,actual_output) VALUES (2,1,'ERROR: y')")
        conn.exec_driver_sql("INSERT INTO evaluation_results(run_id,dataset_id,actual_output) VALUES (2,2,'ERROR: z')")
    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        rows = {r[0]: r[1:] for r in conn.exec_driver_sql(
            "SELECT id, status::text, is_baseline, succeeded_cases, failed_cases FROM evaluation_runs")}
    assert rows[1] == ("PARTIAL", False, 1, 1)
    assert rows[2] == ("FAILED", False, 0, 2)
    command.downgrade(cfg, "0001")
    command.upgrade(cfg, "head")
    engine.dispose()


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="set TEST_POSTGRES_URL to run against Postgres")
def test_postgres_migrated_schema_works_end_to_end_with_the_orm():
    """Fresh Postgres built only by migrations, then executor + comparator on top of it."""
    import httpx
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    from app.core.comparator import ComparisonStatus, evaluate_comparison
    from app.core.run_executor import execute_run
    from app.db.models import Dataset, EvaluationResult, EvaluationRun, Prompt, ResultStatus, RunStatus
    from tests.conftest import FakeLLM, fake_resolver, make_resolver

    engine = create_engine(os.environ["TEST_POSTGRES_URL"])
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    run_migrations(engine)

    db = sessionmaker(bind=engine)()
    prompt = Prompt(version_tag="v1", prompt_text="p", model_name="m")
    cases = [Dataset(input_query=f"q{i}", expected_output=f"a{i}") for i in (1, 2, 3)]
    db.add_all([prompt, *cases])
    db.commit()

    def new_run():
        r = EvaluationRun(prompt_id=prompt.id, status=RunStatus.PENDING, is_baseline=False, expected_cases=3)
        db.add(r)
        db.flush()
        db.add_all(EvaluationResult(run_id=r.id, dataset_id=d.id, status=ResultStatus.PENDING) for d in cases)
        db.commit()
        return r

    good, bad = new_run(), new_run()
    execute_run(db, good.id, llm_fn=FakeLLM({"q1": "a1", "q2": "a2", "q3": "a3"}), resolver=fake_resolver)
    execute_run(db, bad.id, llm_fn=FakeLLM({"q2": httpx.ConnectError("refused")}), resolver=fake_resolver)
    db.expire_all()

    assert db.get(EvaluationRun, good.id).status == RunStatus.COMPLETED
    assert db.get(EvaluationRun, bad.id).status == RunStatus.PARTIAL      # enum value added by migration 0002

    def rows(run_id):
        return db.query(EvaluationResult).filter_by(run_id=run_id).order_by(EvaluationResult.dataset_id).all()

    outcome = evaluate_comparison(db.get(EvaluationRun, good.id), rows(good.id), db.get(EvaluationRun, bad.id), rows(bad.id))
    assert outcome.status == ComparisonStatus.INCOMPLETE and outcome.passed is False

    with pytest.raises(IntegrityError):                                    # unique (run_id, dataset_id)
        db.add(EvaluationResult(run_id=good.id, dataset_id=cases[0].id))
        db.commit()
    db.rollback()
    db.close()
    engine.dispose()


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="set TEST_POSTGRES_URL to run against Postgres")
def test_postgres_0003_backfills_scores_and_enforces_benchmark_identity():
    from sqlalchemy.exc import IntegrityError

    url = os.environ["TEST_POSTGRES_URL"]
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    cfg = alembic_config(url)
    command.upgrade(cfg, "0002")
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO prompts(id,version_tag,prompt_text,model_name) VALUES (1,'v','p','m')")
        conn.exec_driver_sql("INSERT INTO datasets(id,input_query,expected_output) VALUES (1,'q','a'),(2,'q2','a2')")
        conn.exec_driver_sql("INSERT INTO evaluation_runs(id,prompt_id,status,is_baseline) VALUES (1,1,'COMPLETED',false)")
        conn.exec_driver_sql("INSERT INTO evaluation_results(run_id,dataset_id,status,similarity_score,latency_seconds,token_cost,actual_output) VALUES (1,1,'SUCCEEDED',0.9,1,0,'ok')")
        conn.exec_driver_sql("INSERT INTO evaluation_results(run_id,dataset_id,status,actual_output) VALUES (1,2,'FAILED','x')")
    command.upgrade(cfg, "head")

    from app.core.evaluators.base import config_hash
    with engine.connect() as conn:
        scores = conn.exec_driver_sql(
            "SELECT result_id, evaluator_name, evaluator_version, kind, gating, status, score, config_hash FROM evaluation_scores"
        ).fetchall()
    assert len(scores) == 1                                         # the FAILED row gets no score
    rid, name, version, kind, gating, status, score, chash = scores[0]
    assert (name, version, kind, gating, status, score) == ("text_similarity", "1", "similarity", True, "OK", 0.9)
    assert chash == config_hash({})

    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO datasets(id,input_query,expected_output,benchmark,benchmark_version,case_key) VALUES (10,'a','b','core','1','k1')")
        conn.exec_driver_sql("INSERT INTO datasets(id,input_query,expected_output) VALUES (11,'legacy1','x'),(12,'legacy2','y')")   # NULL keys never collide
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.exec_driver_sql("INSERT INTO datasets(id,input_query,expected_output,benchmark,benchmark_version,case_key) VALUES (13,'c','d','core','1','k1')")

    command.downgrade(cfg, "0002")
    command.upgrade(cfg, "head")
    engine.dispose()


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="set TEST_POSTGRES_URL to run against Postgres")
def test_postgres_real_evaluators_end_to_end_through_the_comparator():
    from sqlalchemy.orm import selectinload, sessionmaker

    from app.core.comparator import ComparisonStatus, evaluate_comparison
    from app.core.run_executor import execute_run
    from app.db.models import Dataset, EvaluationResult, EvaluationRun, Prompt, ResultStatus, RunStatus
    from tests.conftest import FakeLLM

    engine = create_engine(os.environ["TEST_POSTGRES_URL"])
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    run_migrations(engine)
    db = sessionmaker(bind=engine)()
    prompt = Prompt(version_tag="v1", prompt_text="p", model_name="m")
    spec = [{"name": "contains", "config": {"all": ["paris"]}}, {"name": "text_similarity"}]
    cases = [Dataset(input_query=f"q{i}", expected_output="The capital is Paris.", eval_spec=spec) for i in range(4)]
    db.add_all([prompt, *cases])
    db.commit()

    def run_with(answer):
        r = EvaluationRun(prompt_id=prompt.id, status=RunStatus.PENDING, is_baseline=False, expected_cases=4)
        db.add(r)
        db.flush()
        db.add_all(EvaluationResult(run_id=r.id, dataset_id=d.id, status=ResultStatus.PENDING) for d in cases)
        db.commit()
        execute_run(db, r.id, llm_fn=FakeLLM(default=answer))
        db.expire_all()
        rows = (db.query(EvaluationResult).options(selectinload(EvaluationResult.scores))
                .filter_by(run_id=r.id).order_by(EvaluationResult.dataset_id).all())
        return db.get(EvaluationRun, r.id), rows

    base_run, base = run_with("The capital is Paris.")
    para_run, para = run_with("Paris, of course, is the one you want.")
    wrong_run, wrong = run_with("The capital is Lyon.")

    assert base_run.status == RunStatus.COMPLETED
    assert {s.evaluator_name for s in base[0].scores} == {"contains", "text_similarity"}
    ok = evaluate_comparison(base_run, base, para_run, para)
    assert ok.status == ComparisonStatus.OK and ok.passed                  # correct paraphrase allowed
    bad = evaluate_comparison(base_run, base, wrong_run, wrong)
    assert bad.status == ComparisonStatus.OK and not bad.passed            # wrong-but-similar blocked
    assert "contains dropped" in bad.summary

    assert db.query(EvaluationResult).count() == 12                        # 3 runs x 4 cases, nothing duplicated
    db.close()
    engine.dispose()
