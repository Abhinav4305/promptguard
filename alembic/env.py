from alembic import context
from sqlalchemy import create_engine

from app.core.config import settings
from app.db import models  # noqa: F401  (registers all tables on Base.metadata)
from app.db.session import Base

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    # An explicit URL (set by tests or `alembic -x`) wins over the app settings.
    return config.get_main_option("sqlalchemy.url") or settings.database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url())
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                # SQLite cannot ALTER constraints in place; batch mode recreates the
                # table there and is a no-op wrapper on PostgreSQL.
                render_as_batch=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
