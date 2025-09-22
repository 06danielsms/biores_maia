"""Endpoints para información de datasets y orígenes."""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_database
from app.models import DatasetSplit, TokenizedSliceResponse
from app.services.data_access import fetch_dataset_state
from app.services.tokenized_dataset import (
    TokenizedDatasetNotAvailable,
    get_tokenized_dataset_slice,
)

router = APIRouter()


@router.get("/state")
async def get_dataset_state(database=Depends(get_database)) -> dict[str, list[dict[str, Any]]]:
    """Devuelve el estado de los datasets DVC/S3, con fallback a mocks."""

    datasets = await fetch_dataset_state(database)
    return {"datasets": datasets}


@router.get(
    "/tokenized/beto",
    response_model=TokenizedSliceResponse,
    summary="Dataset tokenizado BETO listo para UI",
)
async def get_beto_tokenized_dataset(
    split: DatasetSplit = Query("train", description="Split a consultar: train o test"),
    limit: int = Query(20, ge=1, le=200, description="Número de filas a devolver"),
    offset: int = Query(0, ge=0, description="Desplazamiento para paginación"),
    preview_chars: int = Query(
        280,
        ge=40,
        le=1000,
        description="Longitud máxima del preview de texto incluido en la respuesta",
    ),
) -> TokenizedSliceResponse:
    """Expone segmentos del dataset tokenizado (BETO) para consumo en frontend."""

    try:
        return get_tokenized_dataset_slice(
            split,
            offset=offset,
            limit=limit,
            preview_chars=preview_chars,
        )
    except TokenizedDatasetNotAvailable as exc:  # pragma: no cover - depende de IO externo
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
