"""Versionado de rutas v1."""
from fastapi import APIRouter

from .endpoints import corpus, datasets, evaluation, health, jobs, summaries, translation

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(corpus.router, prefix="/corpus", tags=["corpus"])
api_router.include_router(translation.router, prefix="/translation", tags=["translation"])
api_router.include_router(summaries.router, prefix="/summaries", tags=["summaries"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

__all__ = ["api_router"]
