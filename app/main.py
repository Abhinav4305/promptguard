from fastapi import FastAPI
from app.api import api_router
from app.db import models  # noqa: F401
from app.db.session import Base, engine

app = FastAPI(
    title="PromptGuard",
    description="Lightweight LLM Regression Detection Platform",
    version="0.1.0",
)

@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}