from fastapi import FastAPI
from app.api import api_router
from app.db import models  # noqa: F401
from app.db.migrate import run_migrations

app = FastAPI(
    title="PromptGuard",
    description="Lightweight LLM Regression Detection Platform",
    version="0.1.0",
)

@app.on_event("startup")
def migrate_database():
    # Was Base.metadata.create_all(); that cannot alter existing tables.
    # See app/db/migrate.py for how pre-existing databases are adopted.
    run_migrations()

app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}
