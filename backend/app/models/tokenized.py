"""Modelos de respuesta para datasets tokenizados."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DatasetSplit = Literal["train", "test"]


class TokenizedDatasetInfo(BaseModel):
    """Metadatos resumidos del dataset tokenizado."""

    feature_columns: list[str]
    label_map: dict[int, str]
    sequence_length: int = Field(ge=0)


class TokenizedSample(BaseModel):
    """Representa una fila legible del dataset tokenizado."""

    index: int = Field(ge=0)
    text: str
    preview: str
    label: int
    label_name: str | None = None
    token_length: int = Field(ge=0)
    attention_span: int = Field(ge=0)


class TokenizedSliceResponse(BaseModel):
    """Respuesta paginada para una porción del dataset BETO tokenizado."""

    dataset: str
    split: DatasetSplit
    total: int = Field(ge=0)
    limit: int = Field(ge=0)
    offset: int = Field(ge=0)
    has_more: bool
    info: TokenizedDatasetInfo
    items: list[TokenizedSample]


__all__ = [
    "DatasetSplit",
    "TokenizedDatasetInfo",
    "TokenizedSample",
    "TokenizedSliceResponse",
]
