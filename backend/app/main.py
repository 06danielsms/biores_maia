"""Punto de entrada de la aplicación FastAPI."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1 import api_router
from .core.config import settings
from .core.logging import setup_logging
from .core.observability import init_observability

setup_logging()
init_observability()


def create_application() -> FastAPI:
    """Crea y configura la instancia de FastAPI."""

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/health/live", tags=["health"])
    async def live_probe() -> dict[str, str]:
        """Endpoint simple para sondas de Kubernetes."""

        return {"status": "alive"}

    return app


app = create_application()
