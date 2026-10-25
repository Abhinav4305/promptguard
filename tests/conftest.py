"""
Shared fixtures. Tests run on in-memory SQLite with a fake LLM: no Postgres, Redis,
Celery broker or Ollama needed. (Postgres-specific migration behaviour is covered
separately; see tests/test_migrations.py.)
"""
import os
import tempfile
from types import SimpleNamespace

# Must be set before any `app` import: app.db.session builds its engine at import time.
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/default.db"

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import (  # noqa: E402
    Dataset, EvaluationResult, EvaluationRun, Prompt, ResultStatus, RunStatus,
)
from app.db.session import Base  # noqa: E402


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db(session_factory):
    session = session_factory()
    yield session
    session.close()


class FakeLLM:
    """
    Callable with the same keyword signature as call_llm.

    `plan` maps a user query to a list of behaviours consumed one per call; the last
    entry repeats. A behaviour is a string (the model output) or an Exception to raise.
    Unlisted queries answer with `default`.
    """

    def __init__(self, plan: dict | None = None, default: str = "answer"):
        self.plan = {k: list(v) if isinstance(v, list) else [v] for k, v in (plan or {}).items()}
        self.default = default
        self.calls: list[str] = []

    def __call__(self, model_name, system_prompt, user_query, temperature=0.0):
        self.calls.append(user_query)
        steps = self.plan.get(user_query)
        if steps is None:
            behaviour = self.default
        else:
            behaviour = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(behaviour, BaseException):
            raise behaviour
        return SimpleNamespace(
            output=behaviour, latency_seconds=0.5, token_cost=0.0,
            prompt_tokens=1, completion_tokens=1,
        )

    def count(self, query: str) -> int:
        return self.calls.count(query)


def fake_score(prediction: str, reference: str) -> float:
    return 1.0 if prediction == reference else 0.5


@pytest.fixture()
def seed(db):
    """Factory helpers that persist rows and return them."""

    def prompt(model_name="llama3.2"):
        p = Prompt(version_tag="v1", prompt_text="be brief", model_name=model_name)
        db.add(p)
        db.commit()
        return p

    def datasets(n=3):
        rows = [Dataset(input_query=f"q{i}", expected_output=f"a{i}") for i in range(1, n + 1)]
        db.add_all(rows)
        db.commit()
        return rows

    def run(prompt_row, dataset_rows, is_baseline=False):
        r = EvaluationRun(
            prompt_id=prompt_row.id, status=RunStatus.PENDING, is_baseline=is_baseline,
            expected_cases=len(dataset_rows),
        )
        db.add(r)
        db.flush()
        db.add_all(
            EvaluationResult(run_id=r.id, dataset_id=d.id, status=ResultStatus.PENDING)
            for d in dataset_rows
        )
        db.commit()
        return r

    return SimpleNamespace(prompt=prompt, datasets=datasets, run=run)
