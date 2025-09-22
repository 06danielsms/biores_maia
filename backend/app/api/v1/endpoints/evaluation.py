"""Endpoints de evaluación y métricas avanzadas."""
from fastapi import APIRouter, Depends

from app.dependencies import get_database
from app.services.data_access import (
    fetch_alignment_findings,
    fetch_evaluation_overview,
    fetch_rag_metrics,
)

router = APIRouter()


@router.get("/overview")
async def evaluation_overview(database=Depends(get_database)) -> dict[str, object]:
    """Resumen de métricas agregadas con fallback."""

    return await fetch_evaluation_overview(database)


@router.get("/alignment/{summary_id}")
async def evaluation_alignment(
    summary_id: str,
    database=Depends(get_database),
) -> dict[str, object]:
    """Detalle de alineación para un resumen."""

    return await fetch_alignment_findings(database, summary_id)


@router.get("/rag/metrics")
async def rag_metrics(database=Depends(get_database)) -> dict[str, object]:
    """Métricas de RAG."""

    return await fetch_rag_metrics(database)
