"""Endpoints operativos para jobs y costos."""
from fastapi import APIRouter, Depends

from app.dependencies import get_database
from app.services.data_access import fetch_jobs_costs, fetch_jobs_status

router = APIRouter()


@router.get("/status")
async def jobs_status(database=Depends(get_database)) -> dict[str, object]:
    """Estado de jobs activos."""

    return await fetch_jobs_status(database)


@router.get("/costs")
async def jobs_costs(database=Depends(get_database)) -> dict[str, object]:
    """Costos AWS agregados."""

    return await fetch_jobs_costs(database)
