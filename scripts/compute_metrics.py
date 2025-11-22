import os
import time
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import textstat


def compute_metrics_for_text(doc, text: str) -> dict:
    tokens = [t for t in doc if not t.is_space]
    words = [t.text for t in tokens if t.is_alpha]
    word_lens = [len(w) for w in words]
    n_tokens = len(tokens)
    n_words = len(words)
    n_chars = len(text)
    n_chars_nospace = len(text.replace(" ", ""))
    sents = list(doc.sents)
    n_sents = len(sents) if sents else 1

    if n_words == 0:
        return {
            "n_tokens": n_tokens,
            "n_words": 0,
            "n_chars": n_chars,
            "n_chars_nospace": n_chars_nospace,
            "n_sents": 0,
            "avg_word_len": 0.0,
            "median_word_len": 0.0,
            "std_word_len": 0.0,
            "avg_sent_len_words": 0.0,
            "std_sent_len_words": 0.0,
            "ttr": 0.0,
            "root_ttr": 0.0,
            "log_ttr": 0.0,
            "hapax_ratio": 0.0,
            "hapax_dis_ratio": 0.0,
            "prop_long_words_7": 0.0,
            "prop_long_words_10": 0.0,
            "prop_uppercase_words": 0.0,
            "prop_punct_tokens": 0.0,
            "prop_stop_tokens": 0.0,
            "lexical_density": 0.0,
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adj_ratio": 0.0,
            "adv_ratio": 0.0,
            "pron_ratio": 0.0,
            "propn_ratio": 0.0,
            "num_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "dale_chall": 0.0,
            "lix": 0.0,
            "rix": 0.0,
        }

    avg_word_len = float(np.mean(word_lens)) if word_lens else 0.0
    median_word_len = float(np.median(word_lens)) if word_lens else 0.0
    std_word_len = float(np.std(word_lens)) if len(word_lens) > 1 else 0.0

    sent_lens = [len([t for t in s if t.is_alpha]) for s in sents] if sents else [n_words]
    avg_sent_len_words = float(np.mean(sent_lens)) if sent_lens else 0.0
    std_sent_len_words = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0

    types = set(w.lower() for w in words)
    v = len(types)
    ttr = v / n_words if n_words > 0 else 0.0
    root_ttr = v / math.sqrt(n_words) if n_words > 0 else 0.0
    log_ttr = math.log(v + 1) / math.log(n_words + 1) if n_words > 1 else 0.0

    freqs = Counter(w.lower() for w in words)
    hapax = sum(1 for f in freqs.values() if f == 1)
    hapax_dis = sum(1 for f in freqs.values() if f == 2)
    hapax_ratio = hapax / n_words if n_words > 0 else 0.0
    hapax_dis_ratio = hapax_dis / n_words if n_words > 0 else 0.0

    long7 = sum(1 for w in words if len(w) >= 7)
    long10 = sum(1 for w in words if len(w) >= 10)
    prop_long_words_7 = long7 / n_words if n_words > 0 else 0.0
    prop_long_words_10 = long10 / n_words if n_words > 0 else 0.0

    upper_words = sum(1 for w in words if any(c.isupper() for c in w))
    prop_uppercase_words = upper_words / n_words if n_words > 0 else 0.0

    punct_tokens = sum(1 for t in tokens if t.is_punct)
    prop_punct_tokens = punct_tokens / n_tokens if n_tokens > 0 else 0.0

    stop_tokens = sum(1 for t in tokens if t.is_stop)
    prop_stop_tokens = stop_tokens / n_tokens if n_tokens > 0 else 0.0

    content_tokens = sum(1 for t in tokens if not t.is_stop and t.is_alpha)
    lexical_density = content_tokens / n_words if n_words > 0 else 0.0

    pos_counts = Counter(t.pos_ for t in tokens)
    noun_ratio = pos_counts.get("NOUN", 0) / n_tokens if n_tokens > 0 else 0.0
    verb_ratio = pos_counts.get("VERB", 0) / n_tokens if n_tokens > 0 else 0.0
    adj_ratio = pos_counts.get("ADJ", 0) / n_tokens if n_tokens > 0 else 0.0
    adv_ratio = pos_counts.get("ADV", 0) / n_tokens if n_tokens > 0 else 0.0
    pron_ratio = pos_counts.get("PRON", 0) / n_tokens if n_tokens > 0 else 0.0
    propn_ratio = pos_counts.get("PROPN", 0) / n_tokens if n_tokens > 0 else 0.0
    num_ratio = pos_counts.get("NUM", 0) / n_tokens if n_tokens > 0 else 0.0

    try:
        flesch = textstat.flesch_reading_ease(text)
        fk = textstat.flesch_kincaid_grade(text)
        fog = textstat.gunning_fog(text)
        smog = textstat.smog_index(text)
        coleman = textstat.coleman_liau_index(text)
        ari = textstat.automated_readability_index(text)
        dale = textstat.dale_chall_readability_score(text)
    except Exception:
        flesch = fk = fog = smog = coleman = ari = dale = 0.0

    long_words_lix = sum(1 for w in words if len(w) > 6)
    lix = (n_words / n_sents) + (100 * long_words_lix / n_words) if n_sents > 0 and n_words > 0 else 0.0
    rix = long_words_lix / n_sents if n_sents > 0 else 0.0

    return {
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_chars": n_chars,
        "n_chars_nospace": n_chars_nospace,
        "n_sents": n_sents,
        "avg_word_len": avg_word_len,
        "median_word_len": median_word_len,
        "std_word_len": std_word_len,
        "avg_sent_len_words": avg_sent_len_words,
        "std_sent_len_words": std_sent_len_words,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "log_ttr": log_ttr,
        "hapax_ratio": hapax_ratio,
        "hapax_dis_ratio": hapax_dis_ratio,
        "prop_long_words_7": prop_long_words_7,
        "prop_long_words_10": prop_long_words_10,
        "prop_uppercase_words": prop_uppercase_words,
        "prop_punct_tokens": prop_punct_tokens,
        "prop_stop_tokens": prop_stop_tokens,
        "lexical_density": lexical_density,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio,
        "pron_ratio": pron_ratio,
        "propn_ratio": propn_ratio,
        "num_ratio": num_ratio,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk,
        "gunning_fog": fog,
        "smog_index": smog,
        "coleman_liau_index": coleman,
        "automated_readability_index": ari,
        "dale_chall": dale,
        "lix": lix,
        "rix": rix,
    }


def process_file(path: str, nlp, n_process: int, batch_size: int, log_every: int, output_dir: str):
    print(f"\n=== Procesando archivo: {path} ===", flush=True)
    df = pd.read_parquet(path)
    if "text" not in df.columns:
        raise ValueError(f"El archivo {path} no tiene columna 'text'")

    total = len(df)
    print(f"Total filas: {total}", flush=True)
    texts = df["text"].astype(str).tolist()

    start_time = time.time()
    rows_metrics = []
    docs = nlp.pipe(texts, n_process=n_process, batch_size=batch_size)

    for i, (idx, doc) in enumerate(zip(df.index, docs), start=1):
        text = df.at[idx, "text"]
        metrics = compute_metrics_for_text(doc, text)
        metrics_row = {"index": idx}
        metrics_row.update(metrics)
        rows_metrics.append(metrics_row)

        if i % log_every == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = total - i
            eta = remaining / rate if rate > 0 else 0.0
            perc = (i / total) * 100 if total > 0 else 0.0
            print(
                f"[{os.path.basename(path)}] {i}/{total} ({perc:5.1f}%) "
                f"elapsed {elapsed/60:5.1f} min, ETA {eta/60:5.1f} min, "
                f"{rate:6.1f} docs/s",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows_metrics).set_index("index").sort_index()

    common_cols = [c for c in metrics_df.columns if c in df.columns]
    if common_cols:
        print(f"Eliminando columnas duplicadas del original: {common_cols}", flush=True)
        df = df.drop(columns=common_cols)

    out_df = df.join(metrics_df, how="left")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(path)
        name_no_ext = os.path.splitext(base_name)[0]
        out_path = os.path.join(output_dir, f"{name_no_ext}_metrics.parquet")
    else:
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        name_no_ext = os.path.splitext(base_name)[0]
        out_path = os.path.join(dir_name, f"{name_no_ext}_metrics.parquet")

    out_df.to_parquet(out_path, index=False)
    total_time = time.time() - start_time
    print(f"[{os.path.basename(path)}] terminado. Guardado en: {out_path}", flush=True)
    print(f"Tiempo total: {total_time/60:.2f} min ({total_time:.1f} s)\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Ruta a archivo .parquet de entrada (puede repetirse varias veces)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directorio de salida (si se omite, se usa el mismo de cada input)",
    )
    parser.add_argument(
        "--n-process",
        type=int,
        default=2,
        help="Número de procesos para spaCy (n_process en nlp.pipe)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para nlp.pipe",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Cada cuántos textos imprimir progreso/ETA",
    )
    args = parser.parse_args()

    print("Cargando modelo spaCy...", flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = 2_000_000

    for path in args.input:
        process_file(
            path=path,
            nlp=nlp,
            n_process=args.n_process,
            batch_size=args.batch_size,
            log_every=args.log_every,
            output_dir=args.output_dir,
        )

    print("Todos los archivos han sido procesados.", flush=True)


if __name__ == "__main__":
    main()

