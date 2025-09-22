"""Endpoints de traducción y métricas de calidad."""
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.dependencies import get_database
from app.services.data_access import fetch_translation_metrics, store_translation_job
from app.services.notifications import notify_slack_async

router = APIRouter()


@router.post("/recompute", status_code=status.HTTP_202_ACCEPTED)
async def recompute_translation(
    payload: dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "summary": "Recomputo de traducción para documento CT-2024-0001",
                "value": {
                    "document_id": "CT-2024-0001",
                    "model": "helsinki",
                    "force": True,
                    "notify": True,
                },
            }
        },
    ),
    database=Depends(get_database),
) -> dict[str, str]:
    """Lanza un recomputo de traducción, registrándolo si hay base de datos."""

    document_id = payload.get("document_id")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id es requerido")

    job_id = f"translation-job-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    notify_slack_async(
        f"Recomputo de traducción solicitado para {document_id} con modelo {payload.get('model', 'desconocido')}",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Recomputo de traducción*\n"
                    f"Documento: `{document_id}`\n"
                    f"Modelo: `{payload.get('model', 'n/a')}`",
                },
            }
        ],
    )

    await store_translation_job(database, payload, job_id)

    return {
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",
        "message": "Recomputo encolado. Revisar pipeline celery-worker",
    }


@router.get("/metrics/{document_id}")
async def get_translation_metrics(
    document_id: str,
    database=Depends(get_database),
) -> dict[str, object]:
    """Devuelve métricas de traducción desde Mongo o fallback."""

    record = await fetch_translation_metrics(database, document_id)
    if "generated_at" not in record:
        record["generated_at"] = datetime.now(timezone.utc).isoformat()
    return record
