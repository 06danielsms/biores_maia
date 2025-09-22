"""Endpoints para jobs de resúmenes."""
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.dependencies import get_database
from app.services.data_access import fetch_summary_job, store_summary_job
from app.services.notifications import notify_slack_async

router = APIRouter()


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_summary_job(
    payload: dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "summary": "Job de resumen para Pfizer Q4",
                "value": {
                    "dataset": "pfizer-2024-q4",
                    "model": "ollama-phi3-mini",
                    "max_tokens": 600,
                    "tone": "coloquial",
                    "include_metrics": True,
                },
            }
        },
    ),
    database=Depends(get_database),
) -> dict[str, str]:
    """Crea un job de resumen y lo persiste si hay base de datos."""

    dataset = payload.get("dataset")
    if not dataset:
        raise HTTPException(status_code=400, detail="dataset es requerido")

    job_id = f"summary-job-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    notify_slack_async(
        f"Nuevo job de resumen para dataset {dataset} ({job_id})",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Job de resumen creado*\n"
                    f"Dataset: `{dataset}`\n"
                    f"Modelo: `{payload.get('model', 'n/a')}`",
                },
            }
        ],
    )

    await store_summary_job(database, payload, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job recibido. Revisar worker de resúmenes",
    }


@router.get("/jobs/{job_id}")
async def get_summary_job(job_id: str, database=Depends(get_database)) -> dict[str, object]:
    """Detalle de un job de resumen desde Mongo o fallback."""

    return await fetch_summary_job(database, job_id)
