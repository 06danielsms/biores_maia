"""Endpoints para gestión del corpus."""
from fastapi import APIRouter, Depends, Query

from app.dependencies import get_database
from app.services.data_access import (
    DocumentStatus,
    MetricStatus,
    fetch_corpus_document_detail,
    fetch_corpus_documents,
)

router = APIRouter()


@router.get("/documents")
async def list_documents(
    source: str | None = Query(default=None),
    status: DocumentStatus | None = Query(default=None),
    metrics: MetricStatus | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    database=Depends(get_database),
) -> dict[str, object]:
    """Lista documentos desde Mongo o fallback mock."""

    return await fetch_corpus_documents(
        database,
        source=source,
        status=status,
        metrics=metrics,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}")
async def get_document_detail(
    document_id: str,
    database=Depends(get_database),
) -> dict[str, object]:
    """Detalle de documento con métricas y comentarios."""

    return await fetch_corpus_document_detail(database, document_id)
