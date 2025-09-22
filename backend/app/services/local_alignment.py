"""Lectura de hallazgos de alineación desde un JSON local."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings

REPO_ROOT = Path(__file__).resolve().parents[3]


class LocalAlignmentUnavailable(RuntimeError):
    """Señala que no existe información local de alineación."""


@lru_cache(maxsize=1)
def _load_alignment() -> Dict[str, dict[str, Any]]:
    path_setting = settings.local_alignment_findings_file
    if not path_setting:
        raise LocalAlignmentUnavailable("LOCAL_ALIGNMENT_FINDINGS_FILE no está configurado")

    path = Path(path_setting)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise LocalAlignmentUnavailable(
            f"No se encontró el archivo de hallazgos de alineación en {path}"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        summary_id = str(entry.get("summary_id") or "").strip()
        if not summary_id:
            continue
        payload = dict(entry)
        payload.setdefault("hallucinations", [])
        payload.setdefault("actions", [])
        mapping[summary_id] = payload
    if not mapping:
        raise LocalAlignmentUnavailable(
            f"El archivo {path} no contiene hallazgos de alineación válidos"
        )
    return mapping


def get_local_alignment(summary_id: str) -> dict[str, Any]:
    findings = _load_alignment().get(summary_id)
    if findings is None:
        raise LocalAlignmentUnavailable(
            f"No se encontraron hallazgos locales para {summary_id}"
        )
    return findings


__all__ = ["get_local_alignment", "LocalAlignmentUnavailable"]
