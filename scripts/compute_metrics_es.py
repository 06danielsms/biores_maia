import os, re, json, gc, random
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import spacy
import textdescriptives as td
import mlflow
from tqdm import tqdm
from transformers import pipeline

DATA_ROOT = Path.home() / "Analisis_Datos" / "data" / "healthlit_data_sources"
OUT_DIR = Path("results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
ERR_LOG = OUT_DIR / "errors_log.csv"

SAMPLE_N    = int(os.getenv("SAMPLE_N", "0"))
SAMPLE_FRAC = float(os.getenv("SAMPLE_FRAC", "0.04"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "4"))
N_PLS  = int(os.getenv("N_PLS", "0"))
N_TECH = int(os.getenv("N_TECH", "0"))

MAX_BYTES = int(os.getenv("MAX_BYTES", "2000000"))
MAX_CHARS_PER_DOC = int(os.getenv("MAX_CHARS_PER_DOC", "200000"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "400"))
TRANSLATE_BATCH_SIZE = int(os.getenv("TRANSLATE_BATCH_SIZE", "4"))

SOURCE_TO_GROUP = {
    "ClinicalTrials.gov": "TECH",
    "Cochrane": "PLS",
    "Pfizer": "PLS",
    "Trial Summaries": "PLS",
}
TEXT_EXTS = {".txt",".text",".md",".csv",".tsv",".json",".jsonl",".html",".htm",".xhtml",".xml"}

def read_text(p: Path) -> str:
    try:
        try:
            if p.stat().st_size > MAX_BYTES:
                raise ValueError(f"skip: too big ({p.stat().st_size} bytes)")
        except Exception:
            pass
        raw = p.read_text(encoding="utf-8", errors="ignore")
        ext = p.suffix.lower()
        if ext in {".html",".htm",".xhtml"}:
            return BeautifulSoup(raw, "lxml").get_text(" ", strip=True)
        if ext in {".json",".jsonl"}:
            try:
                j = json.loads(raw)
                for k in ("text","body","content","abstract","summary"):
                    if isinstance(j.get(k), str): return j[k]
            except Exception:
                pass
        return raw
    except Exception as e:
        raise RuntimeError(f"read_text failed: {e}")

def collect_files(root: Path) -> List[Path]:
    files=[]
    for sub in SOURCE_TO_GROUP:
        d = root / sub
        if d.exists():
            files += [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]
    return files

def group_of_path(p: Path) -> str:
    s = str(p).lower()
    if "cochrane" in s:
        if "/non_pls/" in s or "\\non_pls\\" in s:
            return "TECH"
        if "/pls/" in s or "\\pls\\" in s:
            return "PLS"
        return "PLS"
    if "clinicaltrials.gov" in s:
        return "TECH"
    if "trial summaries" in s or "trial_summaries" in s or "pfizer" in s:
        return "PLS"
    for src, grp in SOURCE_TO_GROUP.items():
        if src.lower() in s:
            return grp
    return "UNKNOWN"

def sample_files(files: List[Path]) -> List[Path]:
    random.seed(RANDOM_SEED)
    if N_PLS > 0 or N_TECH > 0:
        pls_pool  = [f for f in files if group_of_path(f) == "PLS"]
        tech_pool = [f for f in files if group_of_path(f) == "TECH"]
        chosen=[]
        if N_PLS  > 0 and pls_pool:  chosen += random.sample(pls_pool,  min(N_PLS,  len(pls_pool)))
        if N_TECH > 0 and tech_pool: chosen += random.sample(tech_pool, min(N_TECH, len(tech_pool)))
        return chosen
    if SAMPLE_N > 0 and SAMPLE_N < len(files):
        return random.sample(files, SAMPLE_N)
    k = max(1, int(len(files) * SAMPLE_FRAC))
    return random.sample(files, k)

def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    def tok_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])
    chunks, cur = [], []
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    for s in sents:
        if not s: 
            continue
        if tok_len(s) > max_tokens:
            words, part = s.split(), []
            for w in words:
                cand = (" ".join(part + [w])).strip()
                if tok_len(cand) > max_tokens:
                    if part: chunks.append(" ".join(part).strip())
                    part = [w]
                else:
                    part.append(w)
            if part: chunks.append(" ".join(part).strip())
            continue
        cand = (" ".join(cur + [s])).strip()
        if tok_len(cand) <= max_tokens:
            cur.append(s)
        else:
            if cur: chunks.append(" ".join(cur).strip())
            cur = [s]
    if cur: chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]

def translate_piecewise(translator, pieces: list[str]) -> list[str]:
    out=[]
    for i in range(0, len(pieces), TRANSLATE_BATCH_SIZE):
        batch = pieces[i:i+TRANSLATE_BATCH_SIZE]
        try:
            res = translator(batch, max_length=512, clean_up_tokenization_spaces=True)
            out.extend([r["translation_text"] for r in res])
        except Exception:
            for p in batch:
                try:
                    r = translator([p], max_length=512, clean_up_tokenization_spaces=True)[0]["translation_text"]
                    out.append(r)
                except Exception:
                    mid = len(p)//2
                    sub = [p[:mid], p[mid:]]
                    r = translator(sub, max_length=512, clean_up_tokenization_spaces=True)
                    out.extend([x["translation_text"] for x in r])
    return out

def translate_en2es_offline(text: str, translator) -> str:
    max_tokens = max(64, min(450, getattr(translator.tokenizer, "model_max_length", 512) - 32))
    pieces = chunk_by_tokens(text, translator.tokenizer, max_tokens)
    if not pieces:
        return ""
    out = translate_piecewise(translator, pieces)
    es = " ".join(out)
    if len(es) > MAX_CHARS_PER_DOC:
        es = es[:MAX_CHARS_PER_DOC]
    return es

def log_error(row: dict):
    df = pd.DataFrame([row])
    if ERR_LOG.exists():
        df.to_csv(ERR_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(ERR_LOG, index=False)

def main():
    mlflow.set_tracking_uri(f"file://{Path.cwd().absolute()}/mlflow_work")
    mlflow.set_experiment("healthlit_metrics_es")

    files_all = collect_files(DATA_ROOT)
    files = sample_files(files_all)

    translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device=-1)

    try:
        nlp = spacy.load("es_core_news_sm", disable=["ner","lemmatizer"])
    except OSError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
        nlp = spacy.load("es_core_news_sm", disable=["ner","lemmatizer"])
    nlp.add_pipe("textdescriptives/all")
    nlp.max_length = max(nlp.max_length, MAX_CHARS_PER_DOC + 10000)

    rows=[]
    with mlflow.start_run(run_name="translate_offline_and_compute_es") as run:
        mlflow.log_params({
            "sample_frac": SAMPLE_FRAC, "sample_n": SAMPLE_N, "random_seed": RANDOM_SEED,
            "spacy_model": "es_core_news_sm", "translator": "Helsinki-NLP/opus-mt-en-es",
            "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK, "translate_batch_size": TRANSLATE_BATCH_SIZE,
            "max_chars_per_doc": MAX_CHARS_PER_DOC, "max_bytes": MAX_BYTES,
            "quota_pls": N_PLS, "quota_tech": N_TECH
        })

        for p in tqdm(files, desc="Traduciendo y calculando métricas (offline)"):
            try:
                raw = read_text(p)
                if len(raw.strip()) < 20:
                    raise ValueError("empty/short text")

                es = translate_en2es_offline(raw, translator)
                if len(es.strip()) < 20:
                    raise ValueError("empty/short after translate")

                doc = nlp(es)
                df_doc = td.extract_df(doc)
                metrics = df_doc.select_dtypes(include=[np.number]).iloc[0].to_dict()

                src = None
                for s in SOURCE_TO_GROUP:
                    if s in p.parts or s in str(p):
                        src = s; break
                if src is None:
                    src = "UNKNOWN"
                grp = group_of_path(p)

                rows.append({
                    "file": str(p), "source": src, "group": grp,
                    "len_src_chars": len(raw), "len_es_chars": len(es), **metrics
                })

            except Exception as e:
                log_error({"file": str(p), "error": repr(e)})
            finally:
                gc.collect()

        df = pd.DataFrame(rows)
        out_csv = OUT_DIR / "metrics_sample_es.csv"
        if not df.empty:
            df.to_csv(out_csv, index=False)
            mlflow.log_artifact(str(out_csv), artifact_path="tables")

            numeric_cols = [c for c in df.columns if c not in ("file","source","group","len_src_chars","len_es_chars")
                            and pd.api.types.is_numeric_dtype(df[c])]
            pd.Series(sorted(numeric_cols), name="metric").to_csv(OUT_DIR/"metrics_list_es.csv", index=False)
            mlflow.log_artifact(str(OUT_DIR/"metrics_list_es.csv"), artifact_path="tables")

            mlflow.log_metrics({
                "n_files_all": len(files_all), "n_files_sample": len(files),
                "n_rows_df": len(df), "n_numeric_metrics": len(numeric_cols)
            })

        if ERR_LOG.exists():
            mlflow.log_artifact(str(ERR_LOG), artifact_path="tables")

if __name__ == "__main__":
    main()
