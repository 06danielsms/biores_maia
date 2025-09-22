"""Servicios de acceso a datos con fallback a información mock."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

try:  # pragma: no cover - permite correr sin motor instalado
    from motor.motor_asyncio import AsyncIOMotorDatabase
except ModuleNotFoundError:  # pragma: no cover
    AsyncIOMotorDatabase = Any  # type: ignore[assignment]

try:  # pragma: no cover - pymongo opcional
    from pymongo.errors import PyMongoError
except ModuleNotFoundError:  # pragma: no cover
    PyMongoError = Exception  # type: ignore[assignment]

from app.services.local_alignment import (
    LocalAlignmentUnavailable,
    get_local_alignment,
)
from app.services.local_corpus import (
    LocalCorpusUnavailable,
    get_local_corpus_document,
    list_local_corpus_documents,
)
from app.services.local_metrics import (
    get_dataset_state_snapshot,
    get_evaluation_summary,
    get_jobs_costs_snapshot,
    get_jobs_status_snapshot,
    get_rag_summary,
)
from app.services.local_summary_jobs import (
    LocalSummaryJobsUnavailable,
    get_local_summary_job,
)
from app.services.local_translation_metrics import (
    LocalTranslationMetricsUnavailable,
    get_local_translation_metrics,
)
from app.services.s3_corpus import (
    S3CorpusUnavailable,
    get_s3_corpus_document,
    list_s3_corpus_documents,
)

logger = logging.getLogger(__name__)

DocumentStatus = Literal["translated", "pending"]
MetricStatus = Literal["published", "processing"]


DEFAULT_DATASETS = [
    {
        "name": "ClinicalTrials Highlights",
        "version": "v2024.11.20",
        "last_sync": "2024-11-22T08:45:00Z",
        "storage": "s3://biores-maia/dvc/clinical-trials",
        "status": "synced",
    },
    {
        "name": "Cochrane Summaries",
        "version": "v2024.11.19",
        "last_sync": "2024-11-21T21:05:00Z",
        "storage": "s3://biores-maia/dvc/cochrane",
        "status": "resync_required",
    },
    {
        "name": "Pfizer Lote Q4",
        "version": "pending",
        "last_sync": "2024-11-18T14:30:00Z",
        "storage": "s3://biores-maia/dvc/pfizer-q4",
        "status": "queued",
    },
]

DEFAULT_CORPUS_ITEMS = [
    {
        "id": "CT-2024-0001",
        "file_name": "CT-2024-0001.json",
        "source": "ClinicalTrials.gov",
        "language": "en",
        "readability_fkgl": 46.2,
        "translated": True,
        "metrics_ready": True,
        "tokens": 1874,
        "domain": "Cardiología",
        "alignment_risk": "bajo",
        "updated_at": "2024-11-21T22:14:00Z",
    },
    {
        "id": "COCH-2023-0812",
        "file_name": "COCH-2023-0812.json",
        "source": "Cochrane",
        "language": "en",
        "readability_fkgl": 32.5,
        "translated": True,
        "metrics_ready": False,
        "tokens": 2140,
        "domain": "Oncología",
        "alignment_risk": "medio",
        "updated_at": "2024-11-19T17:02:00Z",
    },
    {
        "id": "PFZ-2024-0032",
        "file_name": "PFZ-2024-0032.json",
        "source": "Pfizer",
        "language": "es",
        "readability_fkgl": 58.9,
        "translated": False,
        "metrics_ready": False,
        "tokens": 980,
        "domain": "Vacunas",
        "alignment_risk": "pendiente",
        "updated_at": "2024-11-10T10:22:00Z",
    },
]

DEFAULT_CORPUS_DETAIL = {
    "id": "CT-2024-0001",
    "source": "ClinicalTrials.gov",
    "language": "en",
    "original": "Phase III randomized study evaluating the efficacy...",
    "translation": "Estudio fase III aleatorizado que evalúa la eficacia...",
    "metrics": {
        "bleu": 54.9,
        "chrf2": 0.721,
        "fkgl": 6.3,
        "bertscore": 0.89,
    },
    "comments": [
        {
            "author": "María Ortega",
            "role": "Especialista clínica",
            "content": "Validar terminología cardiológica con comité médico.",
            "timestamp": "2024-11-22T10:30:00Z",
        }
    ],
}

DEFAULT_TRANSLATION_METRICS = {
    "document_id": "CT-2024-0001",
    "generated_at": "2024-11-22T09:14:33Z",
    "metrics": {
        "bleu": 54.9,
        "chrf2": 0.721,
        "fkgl": 6.3,
        "length_ratio": 0.98,
        "medical_terms": 38,
    },
}

DEFAULT_SUMMARY_JOB = {
    "job_id": "summary-job-4221",
    "dataset": "pfizer-2024-q4",
    "model": "phi3:mini",
    "status": "streaming",
    "progress": 62,
    "started_at": "2024-11-22T09:47:00Z",
    "metrics": {
        "readability_fkgl": 6.1,
        "coherence": 0.84,
        "alignscore": 0.73,
    },
}

DEFAULT_EVALUATION_OVERVIEW = {
    "generated_at": "2024-11-22T10:30:00Z",
    "metrics": {
        "bertscore_f1": 0.873,
        "alignscore": 0.74,
        "fkgl": 6.2,
        "coverage_evidence": 0.87,
    },
    "alerts": [
        {
            "summary_id": "TECH-332",
            "level": "critical",
            "detail": "TECH-332 requiere evidencia adicional",
        }
    ],
}

DEFAULT_ALIGNMENT_FINDINGS = {
    "summary_id": "TECH-332",
    "alignscore": 0.73,
    "hallucinations": [
        {
            "severity": "critical",
            "excerpt": "El resumen menciona pacientes pediátricos cuando el ensayo original solo incluye adultos > 18 años.",
            "evidence": "ClinicalTrials.gov NCT04122111",
        }
    ],
    "actions": ["Solicitar nueva generación", "Adjuntar evidencia"],
}

DEFAULT_RAG_METRICS = {
    "datasets": [
        {
            "name": "ClinicalTrials Highlights",
            "precision_at_5": 0.82,
            "recall_at_5": 0.76,
            "ndcg": 0.88,
        },
        {
            "name": "Cochrane Summaries",
            "precision_at_5": 0.79,
            "recall_at_5": 0.71,
            "ndcg": 0.84,
        },
    ]
}

DEFAULT_JOBS_STATUS = {
    "jobs": [
        {
            "job": "Traducción Helsinki batch 2024-Q4",
            "owner": "worker-03",
            "started_at": "2024-11-22T09:01:00Z",
            "duration": "17m",
            "state": "in_progress",
        },
        {
            "job": "AlignScore EC2 stack",
            "owner": "aws-ollama-eval",
            "started_at": "2024-11-22T08:45:00Z",
            "duration": "41m",
            "state": "active",
        },
    ]
}

DEFAULT_JOBS_COSTS = {
    "services": [
        {"service": "EC2 (GPU/CPU)", "monthly_cost": 1240, "trend": "+8.2%"},
        {"service": "S3 storage", "monthly_cost": 310, "trend": "+1.5%"},
        {"service": "Data transfer", "monthly_cost": 190, "trend": "-2.1%"},
    ]
}


def _isoformat(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


async def fetch_dataset_state(db: Optional[AsyncIOMotorDatabase]) -> list[dict[str, Any]]:
    if db is None:
        snapshot = get_dataset_state_snapshot()
        if snapshot:
            return snapshot
        return []

    try:
        cursor = db["datasets_state"].find({}, projection={"_id": 0}).sort("name")
        datasets = await cursor.to_list(length=500)
        if datasets:
            for item in datasets:
                if isinstance(item, dict):
                    item["last_sync"] = _isoformat(item.get("last_sync"))
                    item["updated_at"] = _isoformat(item.get("updated_at"))
            return datasets
    except Exception as exc:  # pragma: no cover - logging defensivo
        logger.warning("Fallo consultando datasets_state en Mongo: %s", exc)
    return []


async def fetch_corpus_documents(
    db: Optional[AsyncIOMotorDatabase],
    *,
    source: Optional[str],
    status: Optional[DocumentStatus],
    metrics: Optional[MetricStatus],
    limit: int,
    offset: int,
) -> dict[str, Any]:
    if db is None:
        try:
            s3_payload = list_s3_corpus_documents(
                limit=limit,
                offset=offset,
                source=source,
                status=status,
                metrics=metrics,
            )
            if s3_payload.get("items"):
                return s3_payload
        except S3CorpusUnavailable as exc:
            logger.debug("Corpus S3 no disponible, se intentará corpus local: %s", exc)

        try:
            return list_local_corpus_documents(
                limit=limit,
                offset=offset,
                source=source,
                status=status,
                metrics=metrics,
            )
        except LocalCorpusUnavailable as exc:
            logger.warning("Corpus local no disponible: %s", exc)

        return {
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
        }

    filters: dict[str, Any] = {}
    if source:
        filters["source"] = source
    if status:
        filters["translated"] = status == "translated"
    if metrics:
        filters["metrics_ready"] = metrics == "published"

    try:
        collection = db["corpus_documents"]
        total = await collection.count_documents(filters)
        cursor = collection.find(filters, projection={"_id": 0}).skip(offset).limit(limit)
        items = await cursor.to_list(length=limit)
        if items:
            for item in items:
                if isinstance(item, dict):
                    item["updated_at"] = _isoformat(item.get("updated_at"))
            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
    except Exception as exc:  # pragma: no cover
        logger.warning("Fallo consultando corpus_documents: %s", exc)

    return {
        "items": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


async def fetch_corpus_document_detail(
    db: Optional[AsyncIOMotorDatabase], document_id: str
) -> dict[str, Any]:
    if db is not None:
        try:
            document = await db["corpus_documents"].find_one({"id": document_id}, projection={"_id": 0})
            if document:
                comments = await db["document_comments"].find({"document_id": document_id}, {"_id": 0}).to_list(length=100)
                detail = {
                    "id": document.get("id", document_id),
                    "source": document.get("source"),
                    "language": document.get("language"),
                    "original": document.get("original", document.get("content")),
                    "translation": document.get("translation", document.get("translated_content")),
                    "metrics": document.get("metrics"),
                    "comments": comments,
                }
                detail["updated_at"] = _isoformat(document.get("updated_at"))
                if comments:
                    for comment in comments:
                        if isinstance(comment, dict):
                            comment["timestamp"] = _isoformat(comment.get("timestamp"))
                return detail
        except (Exception, PyMongoError) as exc:  # pragma: no cover
            logger.warning("Fallo consultando corpus_document detail: %s", exc)

    if db is None:
        try:
            return get_s3_corpus_document(document_id)
        except S3CorpusUnavailable as exc:
            logger.debug("Documento %s no disponible en S3: %s", document_id, exc)
        try:
            return get_local_corpus_document(document_id)
        except LocalCorpusUnavailable as exc:
            logger.debug("Documento %s no disponible en corpus local: %s", document_id, exc)

    return {}


async def fetch_translation_metrics(
    db: Optional[AsyncIOMotorDatabase], document_id: str
) -> dict[str, Any]:
    if db is not None:
        try:
            record = await db["translation_metrics"].find_one(
                {"document_id": document_id},
                sort=[("generated_at", -1)],
                projection={"_id": 0},
            )
            if record:
                record["generated_at"] = _isoformat(record.get("generated_at"))
                return record
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando translation_metrics: %s", exc)
    try:
        return get_local_translation_metrics(document_id)
    except LocalTranslationMetricsUnavailable as exc:
        logger.debug("No se encontraron métricas locales para %s: %s", document_id, exc)
    return {"document_id": document_id, "metrics": {}}


async def store_translation_job(
    db: Optional[AsyncIOMotorDatabase], payload: dict[str, Any], job_id: str
) -> None:
    if db is None:
        return
    try:
        document = {
            "job_id": job_id,
            "payload": payload,
            "status": "queued",
            "created_at": datetime.utcnow(),
        }
        await db["translation_jobs"].insert_one(document)
    except Exception as exc:  # pragma: no cover
        logger.warning("Fallo guardando translation_job %s: %s", job_id, exc)


async def store_summary_job(
    db: Optional[AsyncIOMotorDatabase], payload: dict[str, Any], job_id: str
) -> None:
    if db is None:
        return
    try:
        document = {
            "job_id": job_id,
            "status": "queued",
            "payload": payload,
            "created_at": datetime.utcnow(),
        }
        await db["summary_jobs"].insert_one(document)
    except Exception as exc:  # pragma: no cover
        logger.warning("Fallo guardando summary_job %s: %s", job_id, exc)


async def fetch_summary_job(db: Optional[AsyncIOMotorDatabase], job_id: str) -> dict[str, Any]:
    if db is not None:
        try:
            job = await db["summary_jobs"].find_one({"job_id": job_id}, projection={"_id": 0})
            if job:
                job["started_at"] = _isoformat(job.get("started_at"))
                return job
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando summary_job %s: %s", job_id, exc)
    try:
        return get_local_summary_job(job_id)
    except LocalSummaryJobsUnavailable as exc:
        logger.debug("Summary job %s no disponible localmente: %s", job_id, exc)
    return {}


async def fetch_evaluation_overview(db: Optional[AsyncIOMotorDatabase]) -> dict[str, Any]:
    if db is not None:
        try:
            overview = await db["evaluation_overview"].find_one({}, projection={"_id": 0}, sort=[("generated_at", -1)])
            if overview:
                overview["generated_at"] = _isoformat(overview.get("generated_at"))
                if isinstance(overview.get("alerts"), list):
                    for alert in overview["alerts"]:
                        if isinstance(alert, dict):
                            alert["detected_at"] = _isoformat(alert.get("detected_at"))
                return overview
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando evaluation_overview: %s", exc)
    else:
        summary = get_evaluation_summary()
        if summary:
            return summary
    return {}


async def fetch_alignment_findings(
    db: Optional[AsyncIOMotorDatabase], summary_id: str
) -> dict[str, Any]:
    if db is not None:
        try:
            finding = await db["alignment_findings"].find_one(
                {"summary_id": summary_id}, projection={"_id": 0}
            )
            if finding:
                finding["generated_at"] = _isoformat(finding.get("generated_at"))
                if isinstance(finding.get("hallucinations"), list):
                    for item in finding["hallucinations"]:
                        if isinstance(item, dict):
                            item["timestamp"] = _isoformat(item.get("timestamp"))
                return finding
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando alignment_findings: %s", exc)
    try:
        return get_local_alignment(summary_id)
    except LocalAlignmentUnavailable as exc:
        logger.debug("Hallazgos de alineación no disponibles para %s: %s", summary_id, exc)
    return {"summary_id": summary_id, "hallucinations": [], "actions": []}


async def fetch_rag_metrics(db: Optional[AsyncIOMotorDatabase]) -> dict[str, Any]:
    if db is not None:
        try:
            record = await db["rag_metrics"].find_one({}, projection={"_id": 0}, sort=[("generated_at", -1)])
            if record:
                record["generated_at"] = _isoformat(record.get("generated_at"))
                return record
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando rag_metrics: %s", exc)
    else:
        summary = get_rag_summary()
        if summary:
            return summary
    return {"datasets": []}


async def fetch_jobs_status(db: Optional[AsyncIOMotorDatabase]) -> dict[str, Any]:
    if db is not None:
        try:
            cursor = db["jobs_status"].find({}, projection={"_id": 0})
            jobs = await cursor.to_list(length=500)
            if jobs:
                for job in jobs:
                    if isinstance(job, dict):
                        job["started_at"] = _isoformat(job.get("started_at"))
                        job["last_heartbeat"] = _isoformat(job.get("last_heartbeat"))
                return {"jobs": jobs}
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando jobs_status: %s", exc)
    else:
        snapshot = get_jobs_status_snapshot()
        if snapshot:
            return {"jobs": snapshot}
    return {"jobs": []}


async def fetch_jobs_costs(db: Optional[AsyncIOMotorDatabase]) -> dict[str, Any]:
    if db is not None:
        try:
            cursor = db["jobs_costs"].find({}, projection={"_id": 0})
            services = await cursor.to_list(length=100)
            if services:
                return {"services": services}
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo consultando jobs_costs: %s", exc)
    else:
        snapshot = get_jobs_costs_snapshot()
        if snapshot:
            return {"services": snapshot}
    return {"services": []}
