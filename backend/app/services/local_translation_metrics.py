"""Carga métricas de traducción desde un CSV local."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings

REPO_ROOT = Path(__file__).resolve().parents[3]


class LocalTranslationMetricsUnavailable(RuntimeError):
    """Señala que no se encontraron métricas locales para traducciones."""


@lru_cache(maxsize=1)
def _load_metrics() -> Dict[str, dict[str, Any]]:
    csv_path_setting = settings.local_translation_metrics_csv
    if not csv_path_setting:
        raise LocalTranslationMetricsUnavailable(
            "LOCAL_TRANSLATION_METRICS_CSV no está configurado"
        )

    csv_path = Path(csv_path_setting)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise LocalTranslationMetricsUnavailable(
            f"No se encontró el CSV de métricas en {csv_path}"
        )

    history_path_setting = settings.local_translation_history_file
    history_by_id: Dict[str, list[dict[str, Any]]] = {}
    if history_path_setting:
        history_path = Path(history_path_setting)
        if not history_path.is_absolute():
            history_path = REPO_ROOT / history_path
        if history_path.exists():
            try:
                import json

                history_raw = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(history_raw, list):
                    for entry in history_raw:
                        if not isinstance(entry, dict):
                            continue
                        doc_id = str(entry.get("document_id") or "").strip()
                        if not doc_id:
                            continue
                        items = history_by_id.setdefault(doc_id, [])
                        items.append({k: v for k, v in entry.items() if k != "document_id"})
            except Exception as exc:  # pragma: no cover - defensivo
                raise LocalTranslationMetricsUnavailable(
                    f"No se pudo cargar el historial de traducción: {history_path} ({exc})"
                ) from exc

    metrics_by_id: Dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            document_id = (row.get("file") or row.get("document_id") or "").strip()
            if not document_id:
                continue
            metrics: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"file", "document_id"}:
                    continue
                if value in (None, ""):
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    metrics[key] = value
            metrics_by_id[document_id] = {
                "document_id": document_id,
                "metrics": metrics,
                "history": history_by_id.get(document_id, []),
            }

    if not metrics_by_id:
        raise LocalTranslationMetricsUnavailable(
            f"El archivo {csv_path} no contiene métricas válidas"
        )

    return metrics_by_id


def get_local_translation_metrics(document_id: str) -> dict[str, Any]:
    metrics = _load_metrics().get(document_id)
    if metrics is None:
        raise LocalTranslationMetricsUnavailable(
            f"No se encontraron métricas locales para {document_id}"
        )
    return metrics


__all__ = ["get_local_translation_metrics", "LocalTranslationMetricsUnavailable"]
