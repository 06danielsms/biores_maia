"""Validation helpers for the Streamlit experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional
import math
from collections import Counter
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import torch

try:
    from .utils import REPO_ROOT
except ImportError:  # pragma: no cover - Streamlit script mode
    from utils import REPO_ROOT  # type: ignore

METRICS_PATH = REPO_ROOT / "metrics.csv"
CLASSIFIER_PATH = REPO_ROOT / "inference" / "clasificador_medico_sencillo.pkl"


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


def load_medical_classifier():
    """
    Carga el clasificador médico y el modelo BERT necesario para generar embeddings.
    
    Returns:
        dict con 'classifier', 'model', 'tokenizer', 'device'
    """
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Clasificador médico no encontrado en: {CLASSIFIER_PATH}. "
            "Asegúrate de que el archivo 'clasificador_medico_sencillo.pkl' existe."
        )
    
    # Cargar el clasificador LogisticRegression
    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = pickle.load(f)
    
    # Cargar modelo BERT (usar el mismo que se usó en entrenamiento)
    # Según el notebook, se puede usar bert-base-uncased o ncbi/MedCPT-Article-Encoder
    try:
        from transformers import BertTokenizer, BertModel
        
        # Intentar primero con MedCPT (mejor para textos médicos)
        try:
            tokenizer = BertTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
            model = BertModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
        except:
            # Fallback a BERT base
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained("bert-base-uncased")
        
        # Configurar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        return {
            'classifier': classifier,
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
    except ImportError:
        raise ImportError(
            "transformers no está instalado. Instálalo con: pip install transformers torch"
        )


def classify_text(text: str, classifier_dict=None) -> str:
    """
    Clasifica un texto médico como 'pls' o 'no_pls'.
    
    Args:
        text: Texto médico a clasificar
        classifier_dict: Dict con classifier, model, tokenizer, device (opcional).
                        Si no se proporciona, se carga automáticamente.
    
    Returns:
        'pls' o 'no_pls' según la clasificación
    """
    if classifier_dict is None:
        classifier_dict = load_medical_classifier()
    
    classifier = classifier_dict['classifier']
    model = classifier_dict['model']
    tokenizer = classifier_dict['tokenizer']
    device = classifier_dict['device']
    
    # Tokenizar el texto (igual que en el notebook)
    encoded_input = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=512
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Generar embedding usando BERT
    with torch.no_grad():
        output = model(**encoded_input).last_hidden_state[0, 0]
    
    # Normalizar el embedding
    output = output.cpu().numpy()
    normalized_output = output / np.linalg.norm(output)
    
    # Expandir dimensiones para que sea (1, n_features)
    embedding = normalized_output.reshape(1, -1)
    
    # Clasificar
    prediction = classifier.predict(embedding)
    
    # 0 = médico (no_pls), 1 = sencillo (pls)
    return 'pls' if prediction[0] == 1 else 'no_pls'


__all__ = [
    "load_metrics_dataset",
    "plot_metric",
    "score_pair",
    "evaluate_rows",
    "load_medical_classifier",
    "classify_text",
]
