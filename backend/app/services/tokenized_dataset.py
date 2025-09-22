"""Capa de acceso para datasets tokenizados almacenados en local o S3."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.ipc as ipc

from app.core.config import settings
from app.models import (
    DatasetSplit,
    TokenizedDatasetInfo,
    TokenizedSample,
    TokenizedSliceResponse,
)

logger = logging.getLogger(__name__)

_MODULE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _MODULE_ROOT.parent

try:  # pragma: no cover - dependencias opcionales en algunos entornos
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    BotoCoreError = ClientError = Exception  # type: ignore[assignment]


_LOCAL_DATA_FILES = {
    "train": "data-tokenized_train.arrow",
    "test": "data-tokenized_test.arrow",
}
_S3_DATA_FILES = {
    "train": "train/data-00000-of-00001.arrow",
    "test": "test/data-00000-of-00001.arrow",
}
_DEFAULT_PREVIEW = 280


class TokenizedDatasetNotAvailable(RuntimeError):
    """Señala que el dataset tokenizado no está accesible."""


def _resolve_local_file(file_name: str) -> Path | None:
    """Intenta localizar un archivo local sin depender del cwd."""

    base = Path(settings.tokenized_dataset_root)
    candidates: list[Path] = []
    if base.is_absolute():
        candidates.append(base)
    else:
        candidates.extend(
            [
                Path.cwd() / base,
                _MODULE_ROOT / base,
                _REPO_ROOT / base,
            ]
        )
    for candidate in candidates:
        path = candidate / file_name
        if path.exists():
            return path
    return None


def _read_arrow_table(native_file: pa.NativeFile) -> pa.Table:
    """Lee un archivo Arrow manejando formato stream/file."""

    try:
        reader = ipc.open_stream(native_file)
    except pa.ArrowInvalid:
        native_file.seek(0)
        reader = ipc.open_file(native_file)
    return reader.read_all()


def _load_from_local(split: DatasetSplit) -> pa.Table | None:
    file_name = _LOCAL_DATA_FILES.get(split)
    if not file_name:
        return None
    file_path = _resolve_local_file(file_name)
    if file_path is None:
        return None

    logger.debug("Loading tokenized dataset split '%s' from %s", split, file_path)
    with pa.memory_map(file_path.as_posix(), "r") as source:
        return _read_arrow_table(source)


def _get_s3_client():  # type: ignore[return-type]
    if boto3 is None:
        raise TokenizedDatasetNotAvailable("boto3 no está instalado en el entorno actual")
    return boto3.client("s3")


def _load_from_s3(split: DatasetSplit) -> pa.Table:
    bucket = settings.tokenized_dataset_bucket
    if not bucket:
        raise TokenizedDatasetNotAvailable("No se configuró TOKENIZED_DATA_BUCKET en el entorno")

    key_suffix = _S3_DATA_FILES.get(split)
    if not key_suffix:
        raise TokenizedDatasetNotAvailable(f"Split desconocido: {split}")

    key = f"{settings.tokenized_dataset_prefix.rstrip('/')}/{key_suffix}"

    logger.debug("Fetching tokenized dataset split '%s' from s3://%s/%s", split, bucket, key)
    client = _get_s3_client()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - dependencias externas
        raise TokenizedDatasetNotAvailable(
            f"No se pudo descargar s3://{bucket}/{key}: {exc}"
        ) from exc

    body = response["Body"].read()
    with pa.BufferReader(body) as buffer:
        return _read_arrow_table(buffer)


@lru_cache(maxsize=4)
def _load_table(split: DatasetSplit) -> pa.Table:
    table = _load_from_local(split)
    if table is not None:
        return table

    logger.info("Tokenized split '%s' no disponible localmente; intentando con S3", split)
    return _load_from_s3(split)


@lru_cache(maxsize=1)
def _compute_dataset_info() -> TokenizedDatasetInfo:
    for candidate in cast(tuple[DatasetSplit, ...], ("train", "test")):
        try:
            table = _load_table(candidate)
        except TokenizedDatasetNotAvailable:
            continue
        if table.num_rows:
            break
    else:
        raise TokenizedDatasetNotAvailable("No hay datos tokenizados disponibles")

    sequence_length = 0
    if table.num_rows and "input_ids" in table.column_names:
        first_value = table.column("input_ids")[0].as_py()
        sequence_length = len(first_value)

    return TokenizedDatasetInfo(
        feature_columns=list(table.column_names),
        label_map=settings.tokenized_label_map,
        sequence_length=sequence_length,
    )


def get_tokenized_dataset_slice(
    split: DatasetSplit,
    *,
    offset: int,
    limit: int,
    preview_chars: int = _DEFAULT_PREVIEW,
) -> TokenizedSliceResponse:
    """Devuelve una porción del dataset BETO lista para consumo en frontend."""

    if limit <= 0:
        raise ValueError("limit debe ser mayor a 0")
    if offset < 0:
        raise ValueError("offset no puede ser negativo")

    table = _load_table(split)
    total = table.num_rows

    slice_length = min(limit, max(total - offset, 0))
    data_slice = table.slice(offset, slice_length)

    texts = data_slice.column("text").to_pylist() if "text" in table.column_names else []
    labels = data_slice.column("label").to_pylist() if "label" in table.column_names else []
    input_ids = (
        data_slice.column("input_ids").to_pylist() if "input_ids" in table.column_names else []
    )
    attention_masks = (
        data_slice.column("attention_mask").to_pylist()
        if "attention_mask" in table.column_names
        else []
    )

    items: list[TokenizedSample] = []
    for idx, text in enumerate(texts):
        label = labels[idx] if idx < len(labels) else None
        ids = input_ids[idx] if idx < len(input_ids) else []
        mask = attention_masks[idx] if idx < len(attention_masks) else []

        preview = text[:preview_chars].strip()
        label_name = settings.tokenized_label_map.get(label) if isinstance(label, int) else None
        token_length = len(ids) if isinstance(ids, list) else 0
        attention_span = int(sum(mask)) if isinstance(mask, list) else 0

        items.append(
            TokenizedSample(
                index=offset + idx,
                text=text,
                preview=preview,
                label=int(label) if label is not None else -1,
                label_name=label_name,
                token_length=token_length,
                attention_span=attention_span,
            )
        )

    info = _compute_dataset_info()
    has_more = offset + len(items) < total

    return TokenizedSliceResponse(
        dataset=settings.tokenized_dataset_name,
        split=split,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
        info=info,
        items=items,
    )


__all__ = [
    "TokenizedDatasetNotAvailable",
    "get_tokenized_dataset_slice",
]
