import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    data_cfg = cfg["data"]
    pls_path = data_cfg["pls_parquet"]
    no_pls_path = data_cfg["no_pls_parquet"]
    text_col = data_cfg.get("text_column", "text")

    if not os.path.exists(pls_path) or not os.path.exists(no_pls_path):
        raise FileNotFoundError(
            f"No se encontraron los parquets:\n  {pls_path}\n  {no_pls_path}"
        )

    df_pls = pd.read_parquet(pls_path)
    df_no = pd.read_parquet(no_pls_path)

    if text_col not in df_pls.columns or text_col not in df_no.columns:
        raise KeyError(
            f"La columna de texto '{text_col}' no existe.\n"
            f"PLS cols: {df_pls.columns.tolist()}\n"
            f"NO_PLS cols: {df_no.columns.tolist()}"
        )

    return df_pls, df_no, text_col


def describe_lengths(lengths: pd.Series, prefix: str) -> Dict[str, float]:
    lengths = lengths.dropna().astype(float)
    if lengths.empty:
        return {f"{prefix}_count": 0.0}
    stats = {
        f"{prefix}_count": float(lengths.count()),
        f"{prefix}_sum": float(lengths.sum()),
        f"{prefix}_min": float(lengths.min()),
        f"{prefix}_max": float(lengths.max()),
        f"{prefix}_mean": float(lengths.mean()),
        f"{prefix}_std": float(lengths.std(ddof=1)),
        f"{prefix}_p10": float(lengths.quantile(0.10)),
        f"{prefix}_p25": float(lengths.quantile(0.25)),
        f"{prefix}_p50": float(lengths.quantile(0.50)),
        f"{prefix}_p75": float(lengths.quantile(0.75)),
        f"{prefix}_p90": float(lengths.quantile(0.90)),
        f"{prefix}_p95": float(lengths.quantile(0.95)),
    }
    total = float(lengths.count())
    if total > 0:
        stats[f"{prefix}_share_lt_50"] = float((lengths < 50).sum() / total)
        stats[f"{prefix}_share_50_100"] = float(((lengths >= 50) & (lengths < 100)).sum() / total)
        stats[f"{prefix}_share_100_200"] = float(((lengths >= 100) & (lengths < 200)).sum() / total)
        stats[f"{prefix}_share_ge_200"] = float((lengths >= 200).sum() / total)
    else:
        stats[f"{prefix}_share_lt_50"] = 0.0
        stats[f"{prefix}_share_50_100"] = 0.0
        stats[f"{prefix}_share_100_200"] = 0.0
        stats[f"{prefix}_share_ge_200"] = 0.0
    return stats


def compute_metrics_for_label(df: pd.DataFrame, text_col: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    text = df[text_col].fillna("")
    n_docs = float(len(text))
    n_chars = text.str.len()
    n_words = text.str.split().str.len()
    chars_per_word = n_chars / n_words.replace(0, np.nan)

    metrics["n_docs"] = n_docs
    metrics["n_nonempty_docs"] = float((n_words > 0).sum())
    metrics["avg_words_per_doc"] = float(n_words.mean())
    metrics["avg_chars_per_doc"] = float(n_chars.mean())

    metrics.update(describe_lengths(n_words, "words"))
    metrics.update(describe_lengths(n_chars, "chars"))

    cpw = chars_per_word.dropna()
    if not cpw.empty:
        metrics["chars_per_word_mean"] = float(cpw.mean())
        metrics["chars_per_word_std"] = float(cpw.std(ddof=1))
        metrics["chars_per_word_p25"] = float(cpw.quantile(0.25))
        metrics["chars_per_word_p50"] = float(cpw.quantile(0.50))
        metrics["chars_per_word_p75"] = float(cpw.quantile(0.75))
    else:
        metrics["chars_per_word_mean"] = 0.0
        metrics["chars_per_word_std"] = 0.0
        metrics["chars_per_word_p25"] = 0.0
        metrics["chars_per_word_p50"] = 0.0
        metrics["chars_per_word_p75"] = 0.0

    tokens = (
        text.str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.split()
    )
    vocab = set()
    total_tokens = 0
    for row in tokens:
        vocab.update(row)
        total_tokens += len(row)
    vocab_size = float(len(vocab))
    metrics["vocab_size"] = vocab_size
    metrics["total_tokens"] = float(total_tokens)
    metrics["type_token_ratio"] = float(vocab_size / total_tokens) if total_tokens > 0 else 0.0

    return metrics


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    df_pls, df_no, text_col = load_data(cfg)

    m_pls = compute_metrics_for_label(df_pls, text_col)
    m_no = compute_metrics_for_label(df_no, text_col)

    metric_names = sorted(set(m_pls.keys()) | set(m_no.keys()))
    rows = []
    for name in metric_names:
        rows.append(
            {
                "metric": name,
                "PLS": m_pls.get(name, np.nan),
                "NO_PLS": m_no.get(name, np.nan),
            }
        )

    df_metrics = pd.DataFrame(rows).set_index("metric")

    out_csv = cfg["metrics"]["output_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_metrics.to_csv(out_csv)
    print(f"Métricas guardadas en: {out_csv}")
    print(df_metrics.head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )
    args = parser.parse_args()
    main(args.config)

