import os
import re
import unicodedata
import boto3
import pandas as pd

BUCKET = "biores-maia-data-clean"
PREFIX_NOPLS = "textos_final/no_pls/"
PREFIX_PLS = "textos_final/pls/"
OUTPUT_LOCAL_DIR = "data_final"
S3_OUTPUT_PREFIX = "data_final"
TEST_SIZE = 0.2
MIN_WORDS_PARAGRAPH = 300
RANDOM_STATE = 42

os.makedirs(OUTPUT_LOCAL_DIR, exist_ok=True)
s3 = boto3.client("s3")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
PHONE_RE = re.compile(r"\b\+?\d[\d\s\-\(\)]{6,}\d\b")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"\s+")
MULTI_DOT_RE = re.compile(r"\.{3,}")
MULTI_PUNCT_RE = re.compile(r"([!?]){2,}")

def list_txt_keys(prefix: str):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".txt"):
                keys.append(key)
    return keys

def basename_without_ext(key: str) -> str:
    return os.path.splitext(os.path.basename(key))[0]

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = re.sub(r"^[\s>*\-•·◦]+", "", text, flags=re.MULTILINE)
    text = MULTI_DOT_RE.sub("...", text)
    text = MULTI_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()
    return text

def get_source_from_key(key: str, base_prefix: str) -> str:
    rest = key[len(base_prefix):]
    parts = rest.split("/")
    if len(parts) > 1:
        return parts[0]
    return "unknown"

def build_df_from_prefix(prefix: str, label: str) -> pd.DataFrame:
    keys = list_txt_keys(prefix)
    print(f"[{label}] {len(keys)} archivos .txt en s3://{BUCKET}/{prefix}")
    rows = []
    for i, key in enumerate(keys, start=1):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        raw = obj["Body"].read().decode("utf-8", errors="ignore")
        text = clean_text(raw)
        if not text or len(text.split()) < 10:
            continue
        source = get_source_from_key(key, prefix)
        rows.append({
            "doc_id": basename_without_ext(key),
            "text": text,
            "label": label,
            "source": source,
            "n_chars": len(text),
            "n_words": len(text.split()),
            "s3_key": key,
        })
        if i % 500 == 0:
            print(f"[{label}] procesados {i} archivos...")
    df = pd.DataFrame(rows)
    print(f"[{label}] total filas útiles: {df.shape[0]}")
    if not df.empty:
        print(f"[{label}] distribución por fuente:\n{df['source'].value_counts()}")
    return df

def drop_exact_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates(subset=["text"])
    after = df.shape[0]
    print(f"[{label}] duplicados exactos eliminados: {before - after}")
    return df

def save_df(df: pd.DataFrame, local_name: str, s3_name: str):
    local_path = os.path.join(OUTPUT_LOCAL_DIR, local_name)
    df.to_parquet(local_path, index=False)
    print(f"Guardado local: {local_path}")
    s3_key = f"{S3_OUTPUT_PREFIX}/{s3_name}"
    s3.upload_file(local_path, BUCKET, s3_key)
    print(f"Subido a S3: s3://{BUCKET}/{s3_key}")

def stratified_train_test_split(df: pd.DataFrame, test_size: float, random_state: int):
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    dfs_train = []
    dfs_test = []
    for (_, _), df_grp in df.groupby(["label", "source"]):
        df_grp = df_grp.sample(frac=1.0, random_state=random_state)
        n = len(df_grp)
        n_test = int(n * test_size)
        if n_test == 0 and n > 1:
            n_test = 1
        if n_test >= n:
            n_test = n - 1
        if n_test <= 0:
            dfs_train.append(df_grp)
            continue
        df_test_grp = df_grp.iloc[:n_test]
        df_train_grp = df_grp.iloc[n_test:]
        dfs_train.append(df_train_grp)
        dfs_test.append(df_test_grp)
    df_train = pd.concat(dfs_train).sample(frac=1.0, random_state=random_state)
    if dfs_test:
        df_test = pd.concat(dfs_test).sample(frac=1.0, random_state=random_state)
    else:
        df_test = pd.DataFrame(columns=df_train.columns)
    df_train = df_train.assign(split="train")
    df_test = df_test.assign(split="test")
    return df_train, df_test

def split_into_paragraphs(text: str):
    paras = re.split(r"\n{2,}", text)
    paras = [p.strip() for p in paras if p.strip()]
    if not paras:
        paras = [text]
    return paras

def paragraphs_min_words(text: str, min_words: int):
    paras = split_into_paragraphs(text)
    paras = [p for p in paras if len(p.split()) >= min_words]
    return paras

def df_to_paragraph_units(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        paras = paragraphs_min_words(row["text"], min_words=min_words)
        for i, p in enumerate(paras):
            rows.append({
                "doc_id": row["doc_id"],
                "unit_id": f"{row['doc_id']}_p{i}",
                "text": p,
                "label": row["label"],
                "source": row["source"],
                "split": row.get("split", "unknown"),
                "n_chars": len(p),
                "n_words": len(p.split()),
                "s3_key": row["s3_key"],
            })
    df_units = pd.DataFrame(rows)
    return df_units

def print_stats_main(df_train: pd.DataFrame, df_test: pd.DataFrame):
    print("\nDataset principal (documentos completos):")
    print("Train total:", len(df_train))
    print(df_train.groupby(["label", "source"]).size())
    print("Test total:", len(df_test))
    print(df_test.groupby(["label", "source"]).size())

def print_stats_augmented(df_train_units: pd.DataFrame, df_test_units: pd.DataFrame):
    print("\nDataset aumentado (párrafos >= palabras mínimas):")
    print("Train total:", len(df_train_units))
    print(df_train_units.groupby(["label", "source"]).size())
    print("Test total:", len(df_test_units))
    print(df_test_units.groupby(["label", "source"]).size())

def main():
    df_no = build_df_from_prefix(PREFIX_NOPLS, "no_pls")
    if not df_no.empty:
        df_no = drop_exact_duplicates(df_no, "no_pls")
        save_df(df_no, "no_pls_clean.parquet", "no_pls_clean.parquet")
    else:
        print("[no_pls] sin filas útiles después de la limpieza.")
    df_pl = build_df_from_prefix(PREFIX_PLS, "pls")
    if not df_pl.empty:
        df_pl = drop_exact_duplicates(df_pl, "pls")
        save_df(df_pl, "pls_clean.parquet", "pls_clean.parquet")
    else:
        print("[pls] sin filas útiles después de la limpieza.")

    df_all = pd.concat([df_no, df_pl], ignore_index=True)
    if df_all.empty:
        print("No hay datos para construir datasets.")
        return

    df_train_docs, df_test_docs = stratified_train_test_split(df_all, TEST_SIZE, RANDOM_STATE)
    save_df(df_train_docs, "main_train.parquet", "main_train.parquet")
    save_df(df_test_docs, "main_test.parquet", "main_test.parquet")

    df_train_units = df_to_paragraph_units(df_train_docs, MIN_WORDS_PARAGRAPH)
    df_test_units = df_to_paragraph_units(df_test_docs, MIN_WORDS_PARAGRAPH)

    if not df_train_units.empty:
        save_df(df_train_units, "augmented_train.parquet", "augmented_train.parquet")
    else:
        print("Sin párrafos de entrenamiento >= mín palabras.")
    if not df_test_units.empty:
        save_df(df_test_units, "augmented_test.parquet", "augmented_test.parquet")
    else:
        print("Sin párrafos de prueba >= mín palabras.")

    print_stats_main(df_train_docs, df_test_docs)
    print_stats_augmented(df_train_units, df_test_units)
    print("\nPipeline completado.")

if __name__ == "__main__":
    main()

