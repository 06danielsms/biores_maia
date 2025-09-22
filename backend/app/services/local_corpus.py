"""Carga de corpus local almacenado en JSON para operar sin Mongo/S3."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings

REPO_ROOT = Path(__file__).resolve().parents[3]


class LocalCorpusUnavailable(RuntimeError):
    """Señala que el corpus local no está disponible."""


@dataclass(frozen=True)
class _CorpusDocument:
    id: str
    payload: dict[str, Any]

    @property
    def translated(self) -> bool:
        return bool(self.payload.get("translated") or self.payload.get("translation"))

    @property
    def metrics_ready(self) -> bool:
        metrics = self.payload.get("metrics")
        return bool(metrics)


def _load_corpus_file() -> list[_CorpusDocument]:
    path_setting = settings.local_corpus_file
    if not path_setting:
        raise LocalCorpusUnavailable("LOCAL_CORPUS_FILE no está configurado")

    path = Path(path_setting)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise LocalCorpusUnavailable(f"No se encontró el corpus local en {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    documents: list[_CorpusDocument] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id") or item.get("document_id") or "").strip()
        if not doc_id:
            continue
        payload = dict(item)
        payload.setdefault("file_name", f"{doc_id}.json")
        payload.setdefault("alignment_risk", payload.get("alignment_risk", "pendiente"))
        documents.append(_CorpusDocument(id=doc_id, payload=payload))
    if not documents:
        raise LocalCorpusUnavailable(f"El corpus local en {path} no contiene documentos válidos")
    return documents


@lru_cache(maxsize=1)
def _get_corpus_documents() -> list[_CorpusDocument]:
    return _load_corpus_file()


def list_local_corpus_documents(
    *,
    limit: int,
    offset: int,
    source: Optional[str] = None,
    status: Optional[str] = None,
    metrics: Optional[str] = None,
) -> dict[str, Any]:
    documents = _get_corpus_documents()

    def matches(doc: _CorpusDocument) -> bool:
        payload = doc.payload
        if source and payload.get("source") != source:
            return False
        if status == "translated" and not doc.translated:
            return False
        if status == "pending" and doc.translated:
            return False
        if metrics == "published" and not doc.metrics_ready:
            return False
        if metrics == "processing" and doc.metrics_ready:
            return False
        return True

    filtered = [doc for doc in documents if matches(doc)]
    slice_items = filtered[offset : offset + limit]

    items = []
    for doc in slice_items:
        payload = dict(doc.payload)
        payload.setdefault("id", doc.id)
        payload.setdefault("language", payload.get("language", "en"))
        payload.setdefault("translated", doc.translated)
        payload.setdefault("metrics_ready", doc.metrics_ready)
        items.append(payload)

    return {
        "items": items,
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
    }


def get_local_corpus_document(document_id: str) -> dict[str, Any]:
    for doc in _get_corpus_documents():
        if doc.id == document_id:
            payload = dict(doc.payload)
            payload.setdefault("id", doc.id)
            return payload
    raise LocalCorpusUnavailable(f"Documento {document_id} no existe en el corpus local")


__all__ = ["list_local_corpus_documents", "get_local_corpus_document", "LocalCorpusUnavailable"]
