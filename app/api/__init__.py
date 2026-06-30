from fastapi import APIRouter

from app.api.routes import datasets, evaluations, prompts

api_router = APIRouter()
api_router.include_router(prompts.router)
api_router.include_router(datasets.router)
api_router.include_router(evaluations.router)