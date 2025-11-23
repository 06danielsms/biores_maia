"""Lightweight training utilities for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from .utils import REPO_ROOT
except ImportError:  # pragma: no cover - Streamlit script mode
    from utils import REPO_ROOT  # type: ignore

DEFAULT_PLS_CHUNKS = REPO_ROOT / "data_processed_en_local" / "PLS" / "data_final_chunks.parquet"
DEFAULT_NOPLS_CHUNKS = REPO_ROOT / "data_processed_en_local" / "NO_PLS" / "data_final_chunks.parquet"


@dataclass
class TrainingConfig:
    model_name: str = "BioClinical-SGD"
    dataset_name: str = "data_processed_en_local"
    pls_path: Path = DEFAULT_PLS_CHUNKS
    npls_path: Path = DEFAULT_NOPLS_CHUNKS
    subset_per_label: int = 400
    learning_rate: float = 0.001
    epochs: int = 5
    batch_size: int = 128
    max_features: int = 6000
    ngram_max: int = 2
    random_state: int = 13
    alpha: float = 1e-4


@dataclass
class TrainingHistoryRow:
    epoch: int
    train_acc: float
    val_acc: float
    train_f1: float
    val_f1: float
    train_loss: float
    val_loss: float


@dataclass
class TrainingResult:
    config: TrainingConfig
    history: List[TrainingHistoryRow]
    class_distribution: Dict[str, int]
    feature_space: int
    train_size: int
    val_size: int
    logs: List[str]

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.history])


TEXT_CANDIDATES = ["chunk_text", "clean_text", "text"]


def _pick_column(pf: pq.ParquetFile) -> str:
    for candidate in TEXT_CANDIDATES:
        if candidate in pf.schema_arrow.names:
            return candidate
    return pf.schema_arrow.names[0]


def _sample_parquet(path: Path, label: str, limit: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["chunk_text", "label"])
    pf = pq.ParquetFile(path)
    column = _pick_column(pf)
    frames: List[pd.DataFrame] = []
    collected = 0
    for row_group in range(pf.num_row_groups):
        table = pf.read_row_group(row_group, columns=[column])
        batch = table.to_pandas()
        batch = batch.rename(columns={column: "chunk_text"})
        batch["label"] = label
        frames.append(batch)
        collected += len(batch)
        if collected >= limit:
            break
    if not frames:
        return pd.DataFrame(columns=["chunk_text", "label"])
    df = pd.concat(frames, ignore_index=True).head(limit)
    df["chunk_text"] = df["chunk_text"].astype(str)
    return df


def load_training_dataframe(cfg: TrainingConfig) -> pd.DataFrame:
    pls_df = _sample_parquet(cfg.pls_path, "PLS", cfg.subset_per_label)
    npls_df = _sample_parquet(cfg.npls_path, "NO_PLS", cfg.subset_per_label)
    data = pd.concat([pls_df, npls_df], ignore_index=True)
    data = data.dropna(subset=["chunk_text"])
    data = data[data["chunk_text"].str.strip().astype(bool)]
    data = data.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    return data


def _batch_indices(n_samples: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield slice(start, end)


def run_training_job(cfg: TrainingConfig) -> TrainingResult:
    data = load_training_dataframe(cfg)
    if data.empty:
        raise RuntimeError("No se pudieron cargar muestras desde los parquet configurados.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["label"])

    X_train, X_val, y_train, y_val = train_test_split(
        data["chunk_text"],
        y,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=cfg.learning_rate,
        alpha=cfg.alpha,
        penalty="l2",
        random_state=cfg.random_state,
    )

    classes = np.unique(y)
    history: List[TrainingHistoryRow] = []
    logs: List[str] = []

    for epoch in range(1, cfg.epochs + 1):
        for slc in _batch_indices(X_train_vec.shape[0], cfg.batch_size):
            clf.partial_fit(X_train_vec[slc], y_train[slc], classes=classes)

        train_probs = clf.predict_proba(X_train_vec)
        val_probs = clf.predict_proba(X_val_vec)
        train_pred = np.argmax(train_probs, axis=1)
        val_pred = np.argmax(val_probs, axis=1)

        row = TrainingHistoryRow(
            epoch=epoch,
            train_acc=round(float(accuracy_score(y_train, train_pred)), 4),
            val_acc=round(float(accuracy_score(y_val, val_pred)), 4),
            train_f1=round(float(f1_score(y_train, train_pred, average="weighted")), 4),
            val_f1=round(float(f1_score(y_val, val_pred, average="weighted")), 4),
            train_loss=round(float(log_loss(y_train, train_probs, labels=classes)), 4),
            val_loss=round(float(log_loss(y_val, val_probs, labels=classes)), 4),
        )
        history.append(row)
        logs.append(
            f"[Epoch {epoch}/{cfg.epochs}] acc_train={row.train_acc} acc_val={row.val_acc} "
            f"f1_val={row.val_f1} loss_val={row.val_loss}"
        )

    distribution = data["label"].value_counts().to_dict()
    result = TrainingResult(
        config=cfg,
        history=history,
        class_distribution=distribution,
        feature_space=len(vectorizer.vocabulary_),
        train_size=len(y_train),
        val_size=len(y_val),
        logs=logs,
    )
    return result


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "run_training_job",
    "load_training_dataframe",
    "DEFAULT_PLS_CHUNKS",
    "DEFAULT_NOPLS_CHUNKS",
]
