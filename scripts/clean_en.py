import os, re, html, yaml, regex, argparse, hashlib
import pandas as pd
from typing import List, Tuple, Iterable
from bs4 import BeautifulSoup
from unidecode import unidecode
import ftfy
import fsspec
from smart_open import open as sopen
import spacy
from tqdm import tqdm

# --------- Regex & helpers ----------
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}\s?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b")
SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DOB_RE   = re.compile(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b")
MRN_RE   = re.compile(r"\bMRN[:\s]*\w+\b", re.I)
HAS_TAGS_RE = re.compile(r"<[a-zA-Z/][\s\S]*?>")

def strip_html(x: str) -> str:
    if not x:
        return ""
    if not HAS_TAGS_RE.search(x):
        return x
    return BeautifulSoup(x, "html.parser").get_text(" ")

def deidentify(t: str) -> str:
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    t = SSN_RE.sub("[SSN]", t)
    t = DOB_RE.sub("[DATE]", t)
    t = MRN_RE.sub("[MRN]", t)
    return t

def normalize(text: str, cfg: dict) -> str:
    if not text:
        return ""
    t = ftfy.fix_text(html.unescape(text))
    if cfg.get("normalize_unicode", True):
        t = unidecode(t)
    if cfg.get("strip_html", True):
        t = strip_html(t)
    if cfg.get("replace_urls", True):
        t = URL_RE.sub("[URL]", t)
    if cfg.get("replace_emails", True):
        t = EMAIL_RE.sub("[EMAIL]", t)
    if cfg.get("deidentify_phi", True):
        t = deidentify(t)
    rn = cfg.get("replace_numbers", "normalize")
    if rn == "mask":
        t = regex.sub(r"\p{N}+", "[NUM]", t)
    elif rn == "normalize":
        t = regex.sub(r"\p{N}+", "0", t)
    if cfg.get("lowercase", True):
        t = t.lower()
    if cfg.get("remove_punctuation", False):
        t = regex.sub(r"[^\p{L}\p{N}\s]", " ", t)
    if cfg.get("normalize_whitespace", True):
        t = regex.sub(r"\s+", " ", t).strip()
    return t

def load_abbrev_map(csv_path: str):
    if not os.path.exists(csv_path):
        return {}, None
    df = pd.read_csv(csv_path)
    m = {str(a).strip().lower(): str(e).strip().lower()
         for a, e in zip(df["abbr"], df["expanded"])
         if str(a).strip() and str(e).strip()}
    if not m:
        return {}, None
    patt = re.compile(r"\b(" + "|".join(map(re.escape, m.keys())) + r")\b", re.I)
    return m, patt

def expand_abbrev(t: str, m: dict, patt: re.Pattern | None) -> str:
    if not m or patt is None:
        return t
    return patt.sub(lambda k: m[k.group(0).lower()], t)

# --------- Sentence splitting & chunking ----------
def get_sent_splitter(model="en_core_web_sm", n_process=2):
    try:
        # desactiva componentes pesados; el 'parser' es lo más lento
        nlp = spacy.load(model, disable=["ner", "lemmatizer", "tagger", "parser"])
    except Exception:
        nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    nlp._n_process = n_process
    return nlp

def count_tokens_ws(text: str) -> int:
    return len(text.split())

def chunk_by_budget(sentences: List[str], max_tokens: int, overlap: int, joiner=" ") -> List[Tuple[str, int]]:
    chunks: List[Tuple[str, int]] = []
    buf, buf_tokens = [], 0
    for s in sentences:
        s_tok = count_tokens_ws(s)
        if s_tok > max_tokens:
            words = s.split()
            start = 0
            while start < len(words):
                end = min(start + max_tokens, len(words))
                part = joiner.join(words[start:end]).strip()
                if part:
                    chunks.append((part, end - start))
                start = max(end - overlap, end)
            continue
        if buf_tokens + s_tok <= max_tokens:
            buf.append(s); buf_tokens += s_tok
        else:
            chunk_text = joiner.join(buf).strip()
            if chunk_text:
                chunks.append((chunk_text, buf_tokens))
            tail = chunk_text.split()[-overlap:] if (overlap > 0 and chunk_text) else []
            buf = ([" ".join(tail)] if tail else []) + [s]
            buf_tokens = len(tail) + s_tok
    if buf:
        chunk_text = joiner.join(buf).strip()
        if chunk_text:
            chunks.append((chunk_text, buf_tokens))
    return chunks

# --------- Labeling ----------
def infer_label_from_path(path: str, pls_pats: list, npls_pats: list, default="NO_PLS") -> str:
    p = (path or "").lower()
    for pat in pls_pats:
        if pat.lower() in p:
            return "PLS"
    for pat in npls_pats:
        if pat.lower() in p:
            return "NO_PLS"
    return default

# --------- S3 listing & reading ----------
def list_all_files(prefix: str) -> list:
    fs = fsspec.filesystem("s3")
    keys = fs.find(prefix.rstrip("/"))  # incluye TODO (no filtramos extensiones)
    return [f"s3://{k}" if not str(k).startswith("s3://") else str(k) for k in keys]

def read_any(path: str, text_col="text") -> pd.DataFrame:
    """
    Intenta leer como parquet/csv/json(l)/texto.
    Si falla, levanta excepción (el caller creará un stub con read_ok=False).
    """
    lp = path.lower()
    if lp.endswith(".parquet"):
        return pd.read_parquet(path, storage_options={"anon": False})
    if lp.endswith(".csv"):
        return pd.read_csv(path, storage_options={"anon": False})
    if lp.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", storage_options={"anon": False})
    if lp.endswith(".jsonl"):
        return pd.read_json(path, lines=True, storage_options={"anon": False})
    if lp.endswith(".json"):
        obj = pd.read_json(path, lines=False, storage_options={"anon": False})
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame([{text_col: obj}])
    # fallback texto
    with sopen(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return pd.DataFrame([{text_col: txt}])

# --------- Main ----------
def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    io  = cfg["io"]
    cln = cfg["cleaning"]
    tok = cfg["tokenization"]
    ch  = cfg["chunking"]
    lab = cfg["labeling"]

    text_col  = io["text_col"]
    label_col = io.get("label_col", "label")
    id_col    = io.get("id_col")
    raw_prefix = io["s3_raw_prefix"].rstrip("/")
    out_root   = io["s3_output_prefix"].rstrip("/")
    local_out  = io["local_output_dir"]

    pls_pats   = lab.get("pls_patterns", ["/pls/"])
    npls_pats  = lab.get("no_pls_patterns", ["/non_pls/","/alx/","/med/","/not_taken_texts/","/tech/","/technical/"])
    default_lb = lab.get("default_label", "NO_PLS")

    limit = io.get("limit_files")

    # 0) listar todo
    paths = list_all_files(raw_prefix)
    if not paths:
        raise SystemExit(f"No se encontraron archivos en {raw_prefix}.")
    if limit:
        paths = paths[:int(limit)]
        print(f"[INFO] Procesando solo {limit} archivos (modo prueba).")
    print(f"[INFO] Encontrados {len(paths)} archivos bajo {raw_prefix} (muestra 10):")
    for p in paths[:10]:
        print("   -", p)

    # mapa de abreviaturas
    abbr_map, abbr_patt = load_abbrev_map("config/abbrev_map.csv") if cln.get("expand_abbrev", True) else ({}, None)

    rows_pls, rows_npls = [], []
    pbar = tqdm(paths, desc="🧹 Cleaning files", unit="file")

    for i, pth in enumerate(pbar):
        label = infer_label_from_path(pth, pls_pats, npls_pats, default_lb)

        try:
            dfp = read_any(pth, text_col=text_col)
            dfp[text_col] = dfp[text_col].astype(str)
            dfp["clean_text"] = dfp[text_col].map(lambda x: expand_abbrev(normalize(x, cln), abbr_map, abbr_patt))
            read_ok = True
        except Exception as e:
            # si no se puede leer, igual categorizamos: fila stub
            dfp = pd.DataFrame([{text_col: "", "clean_text": ""}])
            read_ok = False

        dfp[label_col] = label
        dfp["source_path"] = pth
        dfp["read_ok"] = read_ok

        if not id_col or id_col not in dfp.columns:
            dfp["doc_id"] = dfp[text_col].map(lambda x: hashlib.sha1(str(x).encode("utf-8")).hexdigest())
        else:
            dfp.rename(columns={id_col: "doc_id"}, inplace=True)

        keep_cols = [c for c in [text_col, "clean_text", "doc_id", label_col, "source_path", "read_ok"] if c in dfp.columns]
        (rows_pls if label == "PLS" else rows_npls).append(dfp[keep_cols])

        if (i + 1) % 1000 == 0:
            with open("clean_progress.txt", "w") as f:
                f.write(str(i + 1))

    # 1) concatenar por clase
    df_pls  = pd.concat(rows_pls,  ignore_index=True) if rows_pls  else pd.DataFrame(columns=[text_col,"clean_text","doc_id",label_col,"source_path","read_ok"])
    df_npls = pd.concat(rows_npls, ignore_index=True) if rows_npls else pd.DataFrame(columns=[text_col,"clean_text","doc_id",label_col,"source_path","read_ok"])

    print("[INFO] Docs por clase (incluye stubs no legibles):", {"PLS": len(df_pls), "NO_PLS": len(df_npls)})

    # 2) guardar CLEAN primero (local + S3)
    out_pls_local  = os.path.join(local_out, "PLS")
    out_npls_local = os.path.join(local_out, "NO_PLS")
    os.makedirs(out_pls_local, exist_ok=True)
    os.makedirs(out_npls_local, exist_ok=True)

    p_doc_pls_local  = os.path.join(out_pls_local,  "data_final_clean.parquet")
    p_doc_npls_local = os.path.join(out_npls_local, "data_final_clean.parquet")
    df_pls.to_parquet(p_doc_pls_local, index=False)
    df_npls.to_parquet(p_doc_npls_local, index=False)

    out_pls_s3  = f"{out_root}/PLS"
    out_npls_s3 = f"{out_root}/NO_PLS"
    p_doc_pls_s3  = f"{out_pls_s3}/data_final_clean.parquet"
    p_doc_npls_s3 = f"{out_npls_s3}/data_final_clean.parquet"
    df_pls.to_parquet(p_doc_pls_s3,  index=False, storage_options={"anon": False})
    df_npls.to_parquet(p_doc_npls_s3, index=False, storage_options={"anon": False})
    print("[OK] CLEAN escritos en local y S3.")

    # 3) chunking (solo para filas con texto; los stubs read_ok=False no generan chunks)
    if ch.get("enabled", True):
        nlp = get_sent_splitter(tok.get("spacy_model_en", "en_core_web_sm"),
                                int(tok.get("n_process", 2)))
        bs = int(tok.get("pipe_batch_size", 128))

        def gen_chunks(df_docs: pd.DataFrame) -> pd.DataFrame:
            if df_docs.empty:
                return pd.DataFrame(columns=["doc_id","chunk_id","chunk_index","n_tokens",label_col,"chunk_text"])
            df_docs = df_docs[df_docs["clean_text"].astype(str).str.len() > 0].copy()
            if df_docs.empty:
                return pd.DataFrame(columns=["doc_id","chunk_id","chunk_index","n_tokens",label_col,"chunk_text"])

            texts   = df_docs["clean_text"].astype(str).tolist()
            doc_ids = df_docs["doc_id"].tolist()
            labels  = df_docs[label_col].tolist()
            rows = []
            use_sents = (ch.get("method","spacy-sentences") == "spacy-sentences")

            for doc, doc_id, lab in tqdm(
                zip(nlp.pipe(texts, n_process=nlp._n_process, batch_size=bs), doc_ids, labels),
                total=len(texts), desc="✂️ Chunking", unit="doc"
            ):
                if use_sents:
                    sents = [s.text.strip() for s in doc.sents if s.text.strip()] or [doc.text]
                else:
                    sents = [doc.text]

                chunks = chunk_by_budget(
                    sents,
                    max_tokens=ch.get("max_tokens", 800),
                    overlap=ch.get("overlap_tokens", 50),
                    joiner=ch.get("joiner", " ")
                )
                for j, (ct, n_tok) in enumerate(chunks):
                    rows.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{j:04d}",
                        "chunk_index": j,
                        "n_tokens": n_tok,
                        label_col: lab,
                        "chunk_text": ct
                    })
            return pd.DataFrame(rows)

        df_chunks_pls  = gen_chunks(df_pls)
        df_chunks_npls = gen_chunks(df_npls)
    else:
        df_chunks_pls  = pd.DataFrame(columns=["doc_id","chunk_id","chunk_index","n_tokens",label_col,"chunk_text"])
        df_chunks_npls = pd.DataFrame(columns=["doc_id","chunk_id","chunk_index","n_tokens",label_col,"chunk_text"])

    # 4) escribir CHUNKS
    p_chk_pls_local  = os.path.join(out_pls_local,  "data_final_chunks.parquet")
    p_chk_npls_local = os.path.join(out_npls_local, "data_final_chunks.parquet")
    df_chunks_pls.to_parquet(p_chk_pls_local, index=False)
    df_chunks_npls.to_parquet(p_chk_npls_local, index=False)

    p_chk_pls_s3  = f"{out_pls_s3}/data_final_chunks.parquet"
    p_chk_npls_s3 = f"{out_npls_s3}/data_final_chunks.parquet"
    df_chunks_pls.to_parquet(p_chk_pls_s3,  index=False, storage_options={"anon": False})
    df_chunks_npls.to_parquet(p_chk_npls_s3, index=False, storage_options={"anon": False})

    print("[OK] CHUNKS escritos en local y S3.")
    print("[INFO] Finalizado.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    main(args.config)

