import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

plt.rcParams["figure.dpi"] = 160
plt.rcParams["axes.grid"] = True

def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    io = cfg.get("io", {})
    return {
        "out_dir": io.get("out_dir", "metrics"),
        "label_col": io.get("label_col", "label"),
    }

def read_parquet_any(path):
    if str(path).startswith("s3://"):
        import fsspec
        fs = fsspec.filesystem("s3")
        with fs.open(path, "rb") as f:
            return pd.read_parquet(f)
    return pd.read_parquet(path)

def safe_save(fig, outdir, fname):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {fname}")

def plot_histograms(df, label_col, outdir, bins=40):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["n_tokens"]]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        if label_col in df.columns:
            for lab, subset in df.groupby(label_col):
                vals = subset[col].dropna().values
                if len(vals) > 0:
                    ax.hist(vals, bins=bins, alpha=0.5, density=True, label=str(lab))
            ax.legend()
        else:
            ax.hist(df[col].dropna(), bins=bins, alpha=0.7, density=True)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col); ax.set_ylabel("Densidad")
        safe_save(fig, outdir, f"hist_{col}.png")

def plot_boxplots(df, label_col, outdir):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["n_tokens"]]
    if label_col not in df.columns:
        return
    for col in num_cols:
        vals = [g[col].dropna().values for _, g in df.groupby(label_col)]
        if any(len(v) > 0 for v in vals):
            fig, ax = plt.subplots(figsize=(6,4))
            ax.boxplot(vals, labels=[str(l) for l in df[label_col].unique()], showfliers=False)
            ax.set_title(f"Boxplot {col} por {label_col}")
            safe_save(fig, outdir, f"box_{col}_por_label.png")

def plot_correlation(df, outdir):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 3:
        return
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(10, 0.5*len(corr.columns)), 8))
    cax = ax.imshow(corr, interpolation="nearest", cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(cax)
    ax.set_title("Matriz de correlación de métricas")
    safe_save(fig, outdir, "heatmap_correlacion_metricas.png")

def plot_medians(df, label_col, outdir):
    if label_col not in df.columns:
        return
    num_df = df.select_dtypes(include=[np.number])
    med = num_df.join(df[label_col]).groupby(label_col).median(numeric_only=True)
    if med.empty: return
    fig, ax = plt.subplots(figsize=(10,5))
    med.T.plot(kind="bar", ax=ax)
    plt.title("Medianas por label")
    plt.xlabel("Métrica"); plt.ylabel("Valor mediano")
    plt.xticks(rotation=90)
    plt.legend(title=label_col)
    safe_save(fig, outdir, "barras_medianas_por_label.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics_path", default="metrics/metrics_all_chunks.parquet")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    outdir = os.path.join(cfg["out_dir"], "figs")
    label_col = cfg["label_col"]

    print(f"→ leyendo métricas desde {args.metrics_path}")
    df = read_parquet_any(args.metrics_path)
    print(f"shape: {df.shape}")
    print(f"Columnas numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")

    plot_histograms(df, label_col, outdir)
    plot_boxplots(df, label_col, outdir)
    plot_correlation(df, outdir)
    plot_medians(df, label_col, outdir)

    print(f"🎨 Gráficos guardados en: {outdir}")

if __name__ == "__main__":
    main()

