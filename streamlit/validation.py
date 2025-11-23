"""Validation helpers for the Streamlit experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import math
from collections import Counter

import pandas as pd

try:
    from .utils import REPO_ROOT
except ImportError:  # pragma: no cover - Streamlit script mode
    from utils import REPO_ROOT  # type: ignore

METRICS_PATH = REPO_ROOT / "metrics.csv"


def load_metrics_dataset(max_rows: int = 5_000) -> pd.DataFrame | None:
    if not METRICS_PATH.exists():
        return None
    df = pd.read_csv(METRICS_PATH, nrows=max_rows)
    return df


def plot_metric(df: pd.DataFrame, metric: str) -> Any:
    try:
        import plotly.express as px  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("plotly no está instalado en el entorno actual.") from exc
    data = df[["label", metric]].dropna()
    fig = px.box(
        data,
        x="label",
        y=metric,
        color="label",
        title=f"Distribución de {metric}",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    return fig


def tokenize(text: str) -> List[str]:
    return [tok for tok in (text or "").lower().split() if tok]


def rouge_1(pred: Iterable[str], ref: Iterable[str]) -> Tuple[float, float, float]:
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum(min(count, ref_counts[tok]) for tok, count in pred_counts.items())
    precision = overlap / max(1, sum(pred_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def bleu_score(pred: List[str], ref: List[str], max_n: int = 4) -> float:
    if not pred or not ref:
        return 0.0
    weights = [1 / max_n] * max_n
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred[i : i + n]) for i in range(len(pred) - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append(overlap / total if total else 0.0)
    # geometric mean with smoothing
    log_sum = sum(w * math.log(p + 1e-9) for w, p in zip(weights, precisions))
    geo_mean = math.exp(log_sum)
    bp = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(1, len(pred)))
    return float(bp * geo_mean)


@dataclass
class ValidationScore:
    document_id: str
    rouge_precision: float
    rouge_recall: float
    rouge_f1: float
    bleu: float
    precision: float
    recall: float
    f1: float


def score_pair(document_id: str, prediction: str, reference: str) -> ValidationScore:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    r_p, r_r, r_f1 = rouge_1(pred_tokens, ref_tokens)

    overlap = len(set(pred_tokens) & set(ref_tokens))
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    bleu = bleu_score(pred_tokens, ref_tokens)
    return ValidationScore(
        document_id=document_id,
        rouge_precision=round(r_p, 4),
        rouge_recall=round(r_r, 4),
        rouge_f1=round(r_f1, 4),
        bleu=round(bleu, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def evaluate_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"prediction", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Las columnas obligatorias no existen: {', '.join(sorted(missing))}")

    records: List[ValidationScore] = []
    for idx, row in df.iterrows():
        doc_id = str(row.get("document_id", f"row_{idx}"))
        records.append(score_pair(doc_id, str(row["prediction"]), str(row["reference"])))
    return pd.DataFrame([vars(r) for r in records])


__all__ = [
    "load_metrics_dataset",
    "plot_metric",
    "score_pair",
    "evaluate_rows",
]
