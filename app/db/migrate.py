"""
Schema migrations at API startup.

Replaces `Base.metadata.create_all()`, which can create missing tables but never
alters existing ones, so it cannot deliver schema changes to a database that
already exists.

Three cases are handled:
  * empty database                      -> alembic upgrade head (0001, 0002, ...)
  * database created by the old
    create_all() hook (no alembic_version
    table, but evaluation_runs exists)  -> stamp 0001, then upgrade head
  * database already under alembic      -> alembic upgrade head

Stamping assumes the legacy schema equals revision 0001, i.e. it was created by
the original create_all() code. Do not stamp a database that was modified by hand.
"""

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from app.db.session import engine as default_engine

logger = logging.getLogger(__name__)

BASELINE_REVISION = "0001"
_ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url: str | None = None) -> Config:
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    if url:
        # alembic.ini treats '%' specially; escape it so URL-encoded passwords survive.
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def run_migrations(bind_engine: Engine | None = None) -> None:
    eng = bind_engine or default_engine
    cfg = alembic_config(eng.url.render_as_string(hide_password=False))
    tables = set(inspect(eng).get_table_names())
    if "alembic_version" not in tables and "evaluation_runs" in tables:
        logger.info("Existing un-versioned schema found; stamping baseline %s", BASELINE_REVISION)
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, "head")
    logger.info("Database schema is up to date")
