"""Preprocessing helpers that wrap the cleaning pipeline utilities."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import re
from collections import Counter

import pandas as pd

try:
    # When executed via `streamlit run app.py`, the module is loaded as a script
    # rather than a package, so we fall back to absolute imports.
    from .utils import REPO_ROOT, TEST_FILES_DIR, load_default_text
    from . import clean_en_local as clean_en
except ImportError:  # pragma: no cover - Streamlit script mode
    from utils import REPO_ROOT, TEST_FILES_DIR, load_default_text  # type: ignore
    import clean_en_local as clean_en  # type: ignore


@dataclass
class PreprocessConfig:
    lowercase: bool = False  # Preservar mayúsculas/minúsculas originales
    remove_punctuation: bool = False
    normalize_unicode: bool = True
    strip_html: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    deidentify_phi: bool = True
    replace_numbers: str = "normalize"  # "mask" | "normalize" | "keep"
    normalize_whitespace: bool = True


DEFAULT_CONFIG = PreprocessConfig()
DEFAULT_BATCH_DIR = TEST_FILES_DIR

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _build_chunk_preview(chunks: List[Any], limit: int = 3) -> List[Dict[str, Any]]:
    preview = []
    for idx, (chunk_text, n_tok) in enumerate(chunks[:limit]):
        preview.append(
            {
                "Chunk": idx,
                "Tokens": n_tok,
                "Extracto": f"{chunk_text[:180]}{'...' if len(chunk_text) > 180 else ''}",
            }
        )
    return preview


@dataclass
class BatchDocumentResult:
    file_path: Path
    relative_path: Path
    label: str
    clean_text: str
    stats: Dict[str, Any]
    chunk_preview: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]


@dataclass
class BatchPreprocessResult:
    base_dir: Path
    documents: List[BatchDocumentResult]
    chunk_records: List[Dict[str, Any]]

    def summary_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in self.documents:
            stats = doc.stats
            rows.append(
                {
                    "documento": doc.file_path.name,
                    "ruta_relativa": str(doc.relative_path),
                    "tokens_original": stats["original"]["tokens"],
                    "tokens_limpio": stats["processed"]["tokens"],
                    "reduccion_pct": stats["token_reduction_pct"],
                }
            )
        return rows

    def to_zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for doc in self.documents:
                rel_path = doc.relative_path
                sanitized = rel_path.with_name(rel_path.stem + "_clean.txt")
                zf.writestr(str(sanitized), doc.clean_text)
        buf.seek(0)
        return buf.read()

    def to_dataframe(self, id_col: str, text_col: str, label_col: str) -> pd.DataFrame:
        if not self.chunk_records:
            return pd.DataFrame(columns=[id_col, text_col, label_col, "source_path", "tokens"])
        records: List[Dict[str, Any]] = []
        for record in self.chunk_records:
            records.append(
                {
                    id_col: record["document_id"],
                    text_col: record["chunk_text"],
                    label_col: record["label"],
                    "source_path": record["source_path"],
                    "tokens": record["tokens"],
                }
            )
        return pd.DataFrame(records)


def base_clean_dict(cfg: PreprocessConfig) -> Dict[str, Any]:
    """Translate the dataclass into the dict expected by clean_en.normalize."""
    return {
        "lowercase": cfg.lowercase,
        "remove_punctuation": cfg.remove_punctuation,
        "normalize_unicode": cfg.normalize_unicode,
        "strip_html": cfg.strip_html,
        "replace_urls": cfg.replace_urls,
        "replace_emails": cfg.replace_emails,
        "deidentify_phi": cfg.deidentify_phi,
        "replace_numbers": cfg.replace_numbers,
        "normalize_whitespace": cfg.normalize_whitespace,
    }


def summarize_text(original: str, processed: str) -> Dict[str, Any]:
    """Compute light-weight stats to feed the Streamlit metric cards."""
    def _stats(txt: str) -> Dict[str, float]:
        tokens = txt.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", txt) if s.strip()]
        n_tokens = len(tokens)
        vocab = len(set(tokens))
        avg_word_len = (sum(len(t) for t in tokens) / n_tokens) if n_tokens else 0.0
        avg_sent_len = (n_tokens / len(sentences)) if sentences else n_tokens
        return {
            "tokens": n_tokens,
            "chars": len(txt),
            "unique": vocab,
            "avg_word_len": round(avg_word_len, 2),
            "avg_sent_len": round(avg_sent_len, 2),
        }

    o = _stats(original)
    p = _stats(processed)
    delta_tokens = (p["tokens"] - o["tokens"])
    reduction = 0.0 if o["tokens"] == 0 else (1 - (p["tokens"] / o["tokens"])) * 100
    return {
        "original": o,
        "processed": p,
        "delta_tokens": delta_tokens,
        "token_reduction_pct": round(reduction, 2),
    }


def preview_chunks(text: str, max_tokens: int, overlap: int, limit: int = 3) -> List[Dict[str, Any]]:
    """Generate a short chunk preview using the helper from scripts.clean_en."""
    if not text.strip():
        return []
    chunks = clean_en.chunk_by_budget([text], max_tokens=max_tokens, overlap=overlap)
    return _build_chunk_preview(chunks, limit=limit)


def apply_preprocessing(
    text: str,
    cfg: PreprocessConfig,
    chunk_tokens: int = 120,
    chunk_overlap: int = 20,
) -> Dict[str, Any]:
    """Apply cleaning + chunk preview and return stats for display."""
    cleaned = clean_en.normalize(text or "", base_clean_dict(cfg))
    stats = summarize_text(text, cleaned)
    chunk_sample = preview_chunks(cleaned, max_tokens=chunk_tokens, overlap=chunk_overlap)
    return {
        "clean_text": cleaned,
        "stats": stats,
        "chunk_preview": chunk_sample,
    }


def _discover_documents(directory: Path) -> List[Path]:
    files = [path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def _resolve_output_dir(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / path_value
    return candidate


def _detect_label(file_path: Path, repo_cfg: Optional[Dict[str, Any]]) -> str:
    labeling = (repo_cfg or {}).get("labeling", {})
    default_label = labeling.get("default_label", "NO_PLS")
    path_str = str(file_path).lower()
    for pattern in labeling.get("pls_patterns", []) or []:
        if pattern.lower() in path_str:
            return "PLS"
    for pattern in labeling.get("no_pls_patterns", []) or []:
        if pattern.lower() in path_str:
            return "NO_PLS"
    return default_label


def _build_chunk_id(rel_path: Path, idx: int) -> str:
    slug = rel_path.as_posix().replace("/", "_").replace(" ", "_")
    return f"{slug}_chunk{idx:04d}"


def process_directory(
    directory: Path | str,
    cfg: PreprocessConfig,
    chunk_tokens: int = 120,
    chunk_overlap: int = 20,
    repo_config: Optional[Dict[str, Any]] = None,
) -> BatchPreprocessResult:
    base_dir = Path(directory).expanduser().resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"La ruta especificada no existe o no es una carpeta: {base_dir}")

    files = _discover_documents(base_dir)
    if not files:
        raise ValueError("No se encontraron documentos .txt o .md dentro de la carpeta indicada.")

    documents: List[BatchDocumentResult] = []
    chunk_records: List[Dict[str, Any]] = []
    for file_path in files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = file_path.read_text(encoding="latin-1", errors="ignore")

        cleaned = clean_en.normalize(raw_text or "", base_clean_dict(cfg))
        chunks_full = clean_en.chunk_by_budget([cleaned], max_tokens=chunk_tokens, overlap=chunk_overlap)
        preview = _build_chunk_preview(chunks_full)
        stats = summarize_text(raw_text, cleaned)
        rel_path = file_path.relative_to(base_dir)
        label = _detect_label(file_path, repo_config)

        chunk_payload: List[Dict[str, Any]] = []
        for idx, (chunk_text, n_tok) in enumerate(chunks_full):
            chunk_id = _build_chunk_id(rel_path, idx)
            record = {
                "document_id": chunk_id,
                "chunk_text": chunk_text,
                "label": label,
                "source_path": str(rel_path),
                "tokens": n_tok,
            }
            chunk_records.append(record)
            chunk_payload.append(record)

        documents.append(
            BatchDocumentResult(
                file_path=file_path,
                relative_path=rel_path,
                label=label,
                clean_text=cleaned,
                stats=stats,
                chunk_preview=preview,
                chunks=chunk_payload,
            )
        )

    return BatchPreprocessResult(base_dir=base_dir, documents=documents, chunk_records=chunk_records)


def persist_parquet_outputs(
    result: BatchPreprocessResult,
    repo_config: Optional[Dict[str, Any]],
) -> List[Path]:
    io_cfg = (repo_config or {}).get("io", {})
    id_col = io_cfg.get("id_col", "id")
    text_col = io_cfg.get("text_col", "chunk_text")
    label_col = io_cfg.get("label_col", "label")
    local_output = _resolve_output_dir(io_cfg.get("local_output_dir"))
    remote_output = _resolve_output_dir(io_cfg.get("s3_output_prefix"))

    df = result.to_dataframe(id_col=id_col, text_col=text_col, label_col=label_col)
    if df.empty:
        return []

    saved_paths: List[Path] = []
    for root in [local_output, remote_output]:
        if root is None:
            continue
        for label, group in df.groupby(label_col):
            target_dir = root / label
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / "data_final_chunks.parquet"
            group.to_parquet(out_path, index=False)
            saved_paths.append(out_path)
    return saved_paths


__all__ = [
    "PreprocessConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_BATCH_DIR",
    "apply_preprocessing",
    "load_default_text",
    "process_directory",
    "persist_parquet_outputs",
    "BatchPreprocessResult",
    "BatchDocumentResult",
]
