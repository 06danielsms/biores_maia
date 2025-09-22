"""Carga de información de jobs de resúmenes desde JSON local."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings

REPO_ROOT = Path(__file__).resolve().parents[3]


class LocalSummaryJobsUnavailable(RuntimeError):
    """Señala que no se encontró información local de summary jobs."""


@lru_cache(maxsize=1)
def _load_jobs() -> Dict[str, dict[str, Any]]:
    path_setting = settings.local_summary_jobs_file
    if not path_setting:
        raise LocalSummaryJobsUnavailable("LOCAL_SUMMARY_JOBS_FILE no está configurado")

    path = Path(path_setting)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise LocalSummaryJobsUnavailable(
            f"No se encontró el archivo de summary jobs en {path}"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        job_id = str(entry.get("job_id") or "").strip()
        if not job_id:
            continue
        mapping[job_id] = dict(entry)
    if not mapping:
        raise LocalSummaryJobsUnavailable(
            f"El archivo {path} no contiene summary jobs válidos"
        )
    return mapping


def get_local_summary_job(job_id: str) -> dict[str, Any]:
    job = _load_jobs().get(job_id)
    if job is None:
        raise LocalSummaryJobsUnavailable(f"No se encontró el summary job {job_id}")
    return job


__all__ = ["get_local_summary_job", "LocalSummaryJobsUnavailable"]
