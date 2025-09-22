"""Integración ligera con corpus almacenado en S3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from functools import lru_cache
from typing import Optional, Sequence

try:  # pragma: no cover - dependencias opcionales segun entorno
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    BotoCoreError = ClientError = Exception  # type: ignore[assignment]

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3CorpusUnavailable(RuntimeError):
    """Señala que el corpus en S3 no está accesible."""


@dataclass(frozen=True)
class _CorpusEntry:
    document_id: str
    s3_key: str
    size_bytes: int
    last_modified: datetime


def _ensure_boto3_available() -> None:
    if boto3 is None:  # pragma: no cover - depende de instalación externa
        raise S3CorpusUnavailable("boto3 no está disponible en el entorno actual")


def _get_s3_client():  # type: ignore[return-type]
    _ensure_boto3_available()
    return boto3.client("s3")  # pragma: no cover - cliente se evalúa en runtime


def _normalise_prefix(prefix: str) -> str:
    cleaned = prefix.strip()
    if not cleaned:
        return ""
    return cleaned[:-1] if cleaned.endswith("/") else cleaned


def _build_entry(raw: dict[str, object]) -> Optional[_CorpusEntry]:
    key = raw.get("Key")
    size = raw.get("Size", 0)
    last_modified = raw.get("LastModified")

    if not isinstance(key, str) or key.endswith("/"):
        return None
    if not isinstance(size, int):
        return None
    if not isinstance(last_modified, datetime):
        return None

    document_id = key.rsplit("/", 1)[-1]
    return _CorpusEntry(
        document_id=document_id,
        s3_key=key,
        size_bytes=size,
        last_modified=last_modified,
    )


@dataclass(frozen=True)
class _CorpusCatalog:
    entries: tuple[_CorpusEntry, ...]
    by_id: dict[str, _CorpusEntry]

    @property
    def total(self) -> int:
        return len(self.entries)


def _load_catalog() -> _CorpusCatalog:
    bucket = settings.corpus_s3_bucket
    if not bucket:
        raise S3CorpusUnavailable("CORPUS_S3_BUCKET no está configurado")

    prefix = _normalise_prefix(settings.corpus_s3_prefix)

    client = _get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    params = {"Bucket": bucket}
    if prefix:
        params["Prefix"] = f"{prefix}/"

    try:
        page_iterator = paginator.paginate(**params)
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - IO externo
        raise S3CorpusUnavailable(f"No se pudo listar objetos en s3://{bucket}/{prefix}: {exc}") from exc

    entries: list[_CorpusEntry] = []
    try:
        for page in page_iterator:
            contents = page.get("Contents", [])
            if not isinstance(contents, Sequence):
                continue
            for raw in contents:  # type: ignore[assignment]
                if not isinstance(raw, dict):
                    continue
                entry = _build_entry(raw)
                if entry:
                    entries.append(entry)
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - IO externo
        raise S3CorpusUnavailable(
            f"No se pudo iterar objetos en s3://{bucket}/{prefix}: {exc}"
        ) from exc

    if not entries:
        raise S3CorpusUnavailable(
            f"No se encontraron documentos en s3://{bucket}/{prefix or ''}".strip()
        )

    entries.sort(key=lambda item: item.last_modified, reverse=True)
    by_id = {entry.document_id: entry for entry in entries}
    return _CorpusCatalog(entries=tuple(entries), by_id=by_id)


@lru_cache(maxsize=1)
def _get_catalog() -> _CorpusCatalog:
    return _load_catalog()


def _approximate_tokens(size_bytes: int) -> int:
    # Heurística simple: palabras de ~5 caracteres + espacio
    tokens = max(size_bytes // 6, 1)
    return tokens


def _entry_to_payload(entry: _CorpusEntry) -> dict[str, object]:
    updated_at = entry.last_modified.astimezone().isoformat()
    return {
        "id": entry.document_id,
        "file_name": f"{entry.document_id}.txt",
        "source": settings.corpus_source_name,
        "language": settings.corpus_default_language,
        "readability_fkgl": None,
        "translated": False,
        "metrics_ready": False,
        "tokens": _approximate_tokens(entry.size_bytes),
        "domain": None,
        "alignment_risk": "pendiente",
        "updated_at": updated_at,
    }


def list_s3_corpus_documents(
    *,
    limit: int,
    offset: int,
    source: Optional[str] = None,
    status: Optional[str] = None,
    metrics: Optional[str] = None,
) -> dict[str, object]:
    catalog = _get_catalog()

    def matches(entry: _CorpusEntry) -> bool:
        if source and source != settings.corpus_source_name:
            return False
        if status == "translated":
            return False
        if metrics == "published":
            return False
        if metrics == "processing" or metrics is None:
            pass
        return True

    filtered: list[_CorpusEntry] = [entry for entry in catalog.entries if matches(entry)]
    slice_items = filtered[offset : offset + limit]

    return {
        "items": [_entry_to_payload(entry) for entry in slice_items],
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
    }


@lru_cache(maxsize=2048)
def _get_object_body(document_id: str) -> str:
    catalog = _get_catalog()
    entry = catalog.by_id.get(document_id)
    if not entry:
        raise S3CorpusUnavailable(f"Documento {document_id} no encontrado en el catálogo S3")

    bucket = settings.corpus_s3_bucket
    client = _get_s3_client()
    try:
        response = client.get_object(Bucket=bucket, Key=entry.s3_key)
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - depende de S3
        raise S3CorpusUnavailable(
            f"No se pudo descargar s3://{bucket}/{entry.s3_key}: {exc}"
        ) from exc

    body = response.get("Body")
    if body is None:  # pragma: no cover - caso excepcional
        raise S3CorpusUnavailable(f"Objeto vacío: s3://{bucket}/{entry.s3_key}")

    data = body.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    if isinstance(data, str):  # pragma: no cover
        return data
    raise S3CorpusUnavailable(f"Respuesta inesperada al descargar {document_id}")


def get_s3_corpus_document(document_id: str) -> dict[str, object]:
    catalog = _get_catalog()
    entry = catalog.by_id.get(document_id)
    if not entry:
        raise S3CorpusUnavailable(f"Documento {document_id} no está disponible en S3")

    body = _get_object_body(document_id)
    tokens = len(body.split()) if body else 0

    detail = {
        "id": entry.document_id,
        "source": settings.corpus_source_name,
        "language": settings.corpus_default_language,
        "original": body,
        "translation": settings.corpus_default_translation_placeholder,
        "metrics": {},
        "comments": [],
        "tokens": tokens,
        "updated_at": entry.last_modified.astimezone().isoformat(),
    }
    return detail
