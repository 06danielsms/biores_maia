"""Compute comprehensive text metrics for preprocessing analysis."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RE_WORD = re.compile(r"\b\w+\b", re.UNICODE)
RE_SENT = re.compile(r"[.!?]+")
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
VOWELS = set("aeiouAEIOUyY")

STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because been before being
below between both but by can cannot could couldn did didn do does doesn doing don down during each
few for from further had hadn has hasn have haven having he her here hers herself him himself his how
i if in into is isn it its itself let me more most mustn my myself no nor not of off on once only or
other our ours ourselves out over own same shan she should shouldn so some such than that the their
theirs them themselves then there these they this those through to too under until up very was wasn we
were weren what when where which while who whom why with won would wouldn you your yours yourself yourselves
""".split())


def safe_text(x: Any) -> str:
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x or "")


def count_syllables(w: str) -> int:
    w = w.lower()
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def readability_block(words: list[str], sents: list[str]) -> Dict[str, float]:
    n_words = len(words)
    n_sents = max(1, sum(1 for s in sents if s.strip()))
    if n_words == 0:
        return {}
    n_syll = sum(count_syllables(w) for w in words)
    words_per_sent = n_words / n_sents
    syll_per_word = n_syll / n_words
    long_words = sum(len(w) > 6 for w in words)
    poly_syll = sum(count_syllables(w) >= 3 for w in words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    flesch = 206.835 - 1.015 * words_per_sent - 84.6 * syll_per_word
    fk = 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59
    fog = 0.4 * (words_per_sent + 100 * long_words / max(1, n_words))
    smog = 1.0430 * math.sqrt(30 * poly_syll / max(1, n_sents)) + 3.1291 if poly_syll > 0 else 0.0
    cli = 5.89 * avg_word_len - 0.3 * (n_sents / max(1, n_words)) - 15.8
    ari = 4.71 * avg_word_len + 0.5 * words_per_sent - 21.43
    dale = 64 - 0.95 * (sum(len(w) > 2 for w in words) * 100 / max(1, n_words))
    chars_per_sent = sum(len(s) for s in sents) / n_sents if n_sents else 0.0
    return dict(
        flesch=flesch,
        fk=fk,
        fog=fog,
        smog=smog,
        cli=cli,
        ari=ari,
        dale=dale,
        words_per_sent=words_per_sent,
        chars_per_sent=chars_per_sent,
    )


def lexical_diversity(words: list[str]) -> Dict[str, float]:
    n = len(words)
    if n == 0:
        return {}
    vocab = Counter(words)
    V = len(vocab)
    V1 = sum(1 for v in vocab.values() if v == 1)
    V2 = sum(1 for v in vocab.values() if v == 2)
    freqs = np.array(list(vocab.values()), dtype=float)
    p = freqs / n
    entropy_words = float(-(p * np.log2(p)).sum())
    ttr = V / n
    hapax_ratio = V1 / n
    disleg_ratio = V2 / n
    guiraud_r = V / math.sqrt(n)
    herdan_c = (math.log(V) / math.log(n)) if n > 1 and V > 1 else 0.0
    dugast_u = ((math.log(n)) ** 2 / (math.log(n) - math.log(V))) if V < n and n > 1 and V > 1 else 0.0
    sichel_s = V2 / max(1, V)
    brunet_w = n ** (-(V ** -0.165)) if V > 0 else 0.0
    honore_r = (100 * math.log(n) / (1 - V1 / max(1, V))) if V1 < V else 0.0
    maas_ttr = ((math.log(n) - math.log(V)) / (math.log(n) ** 2)) if n > 1 and V > 1 else 0.0
    Vi = Counter(freqs).items()
    sum_i2Vi = sum((int(i) ** 2) * int(vi) for i, vi in Vi)
    yule_k = (1e4 * (sum_i2Vi - n) / (n ** 2)) if n > 0 else 0.0
    simpson_d = float((p ** 2).sum())
    stop_ratio = sum(1 for w in words if w in STOPWORDS) / n
    long_ratio = sum(1 for w in words if len(w) >= 7) / n
    short_ratio = sum(1 for w in words if len(w) <= 3) / n
    return dict(
        type_token_ratio=ttr,
        hapax_ratio=hapax_ratio,
        dislegomena_ratio=disleg_ratio,
        unique_words=V,
        guiraud_r=guiraud_r,
        herdan_c=herdan_c,
        dugast_u=dugast_u,
        sichel_s=sichel_s,
        brunet_w=brunet_w,
        honore_r=honore_r,
        maas_ttr=maas_ttr,
        yule_k=yule_k,
        simpson_d=simpson_d,
        entropy_words=entropy_words,
        stopword_ratio=stop_ratio,
        long_word_ratio=long_ratio,
        short_word_ratio=short_ratio,
    )


def char_distributions(text: str) -> Dict[str, float]:
    n_chars = len(text)
    letters = [c for c in text if c.isalpha()]
    n_letters = len(letters)
    n_upper = sum(c.isupper() for c in text)
    n_lower = sum(c.islower() for c in text)
    n_digits = sum(c.isdigit() for c in text)
    n_spaces = text.count(" ")
    n_tabs = text.count("\t")
    n_newl = text.count("\n")
    punct_set = ".,;:!?\"'()-[]{}"
    n_punct = sum(1 for c in text if c in punct_set)
    n_nonascii = sum(1 for c in text if ord(c) > 127)
    n_alpha = sum(c.isalpha() for c in text)
    n_alnum = sum(c.isalnum() for c in text)
    n_vowels = sum(c in VOWELS for c in letters)
    n_conson = max(0, n_letters - n_vowels)
    n_periods = text.count(".")
    n_commas = text.count(",")
    n_sc = text.count(";")
    n_colon = text.count(":")
    n_exc = text.count("!")
    n_q = text.count("?")
    n_hash = text.count("#")
    n_at = text.count("@")
    n_urls = len(RE_URL.findall(text))
    n_emails = len(RE_EMAIL.findall(text))
    vc = Counter(text)
    p = np.array(list(vc.values()), dtype=float) / max(1, n_chars)
    entropy_chars = float(-(p * np.log2(p)).sum())
    runs3 = len(re.findall(r"(.)\1\1+", text))
    return dict(
        n_chars=n_chars,
        n_letters=n_letters,
        n_upper=n_upper,
        n_lower=n_lower,
        n_digits=n_digits,
        n_spaces=n_spaces,
        n_tabs=n_tabs,
        n_newlines=n_newl,
        n_punct=n_punct,
        n_nonascii=n_nonascii,
        n_alpha=n_alpha,
        n_alnum=n_alnum,
        n_vowels=n_vowels,
        n_consonants=n_conson,
        n_periods=n_periods,
        n_commas=n_commas,
        n_semicolons=n_sc,
        n_colons=n_colon,
        n_exclaims=n_exc,
        n_questions=n_q,
        hashtags=n_hash,
        atsigns=n_at,
        urls=n_urls,
        emails=n_emails,
        entropy_chars=entropy_chars,
        repeat_char_runs=runs3,
        digit_ratio=(n_digits / max(1, n_chars)),
        punct_ratio=(n_punct / max(1, n_chars)),
        upper_ratio=(n_upper / max(1, n_chars)),
        lower_ratio=(n_lower / max(1, n_chars)),
        space_ratio=(n_spaces / max(1, n_chars)),
        non_ascii_ratio=(n_nonascii / max(1, n_chars)),
        vowel_ratio=(n_vowels / max(1, n_letters)) if n_letters else 0.0,
        consonant_ratio=(n_conson / max(1, n_letters)) if n_letters else 0.0,
        urls_ratio=(n_urls / max(1, n_chars)),
        emails_ratio=(n_emails / max(1, n_chars)),
    )


def metrics_for_row(row: Dict[str, Any], text_col: str = "chunk_text", label_col: str = "label") -> Dict[str, Any]:
    """Compute all metrics for a single row/chunk."""
    text = safe_text(row.get(text_col, ""))
    n_tokens = int(row.get("tokens", len(text.split())))
    words = RE_WORD.findall(text.lower())
    sents = RE_SENT.split(text)
    surf = char_distributions(text)
    avg_tok_len = (surf["n_chars"] / n_tokens) if n_tokens else 0.0
    avg_word_len = (np.mean([len(w) for w in words]) if words else 0.0)
    rb = readability_block(words, sents)
    lx = lexical_diversity(words)
    out = {
        "document_id": row.get("id", row.get("document_id")),
        label_col: row.get(label_col),
        "n_tokens": n_tokens,
        "avg_tok_len": avg_tok_len,
        "avg_word_len": avg_word_len,
        "n_words": len(words),
        "n_sents": max(1, sum(1 for s in sents if s.strip())),
    }
    out.update(rb)
    out.update(lx)
    out.update(surf)
    return out


def compute_metrics_batch(df: pd.DataFrame, text_col: str = "chunk_text", label_col: str = "label", n_jobs: int = 2) -> pd.DataFrame:
    """Compute metrics for a batch of chunks in parallel."""
    rows = df.to_dict("records")
    res = Parallel(n_jobs=n_jobs)(delayed(metrics_for_row)(r, text_col, label_col) for r in rows)
    return pd.DataFrame(res)
