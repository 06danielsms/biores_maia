"""Endpoints de health-check."""
from datetime import datetime, timezone

from fastapi import APIRouter, status

from ....core.config import settings

router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Devuelve el estado básico del servicio."""

    return {
        "status": "ok",
        "service": settings.app_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment,
    }
