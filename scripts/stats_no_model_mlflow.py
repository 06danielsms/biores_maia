from pathlib import Path
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde
import mlflow

IN_FULL = Path("data/metrics_es_full.csv")        # si existe, usa el combinado
IN_ONE  = Path("results/metrics_sample_es.csv")   # si no, usa el último
OUT_DIR = Path("results/plots_stats"); OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP_THRESHOLD = float(os.getenv("OVERLAP_THRESHOLD", "0.60"))
MIN_PER_GROUP = int(os.getenv("MIN_PER_GROUP", "5"))

def load_df():
    if IN_FULL.exists():
        df = pd.read_csv(IN_FULL); src = str(IN_FULL)
    elif IN_ONE.exists():
        df = pd.read_csv(IN_ONE);  src = str(IN_ONE)
    else:
        raise SystemExit("No encuentro data/metrics_es_full.csv ni results/metrics_sample_es.csv")
    return df, src

def kde_overlap(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < MIN_PER_GROUP or len(b) < MIN_PER_GROUP: return np.nan
    xs = np.linspace(np.nanpercentile(np.r_[a,b],1), np.nanpercentile(np.r_[a,b],99), 256)
    try:
        fa = gaussian_kde(a)(xs); fb = gaussian_kde(b)(xs)
        area = np.trapz(np.minimum(fa, fb), xs)
        denom = max(np.trapz(fa, xs), np.trapz(fb, xs))
        return float(area/denom) if denom>0 else np.nan
    except Exception:
        return np.nan

def cliffs_delta_from_u(u, nx, ny):
    # delta = 2*U/(nx*ny) - 1
    return float(2.0*u/(nx*ny) - 1.0)

def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    m1, m2 = np.nanmean(a), np.nanmean(b)
    s1, s2 = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    sp = np.sqrt(((len(a)-1)*s1**2 + (len(b)-1)*s2**2) / (len(a)+len(b)-2))
    return float((m1 - m2)/sp) if sp>0 else np.nan

def bh_fdr(pvals):
    # Benjamini–Hochberg: devuelve p-ajustados en el mismo orden
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(m, float)
    prev = 1.0
    for i, idx in enumerate(order, start=1):
        q = pvals[idx] * m / i
        prev = min(prev, q)
        ranked[idx] = prev
    return ranked

def main():
    mlflow.set_tracking_uri(f"file://{Path.cwd().absolute()}/mlflow_work")
    mlflow.set_experiment("healthlit_stats_es")

    df, dataset_name = load_df()
    if "group" not in df.columns:
        raise SystemExit("El CSV no tiene columna 'group' (PLS/TECH).")

    # columnas numéricas (métricas)
    drop = {"file","source","group","len_src_chars","len_es_chars"}
    metrics = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # conteos por grupo
    counts = df["group"].value_counts().to_dict()
    n_pls, n_tech = counts.get("PLS",0), counts.get("TECH",0)

    with mlflow.start_run(run_name="stats_no_model") as run:
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("n_total", len(df))
        mlflow.log_param("n_PLS", n_pls)
        mlflow.log_param("n_TECH", n_tech)
        mlflow.log_param("overlap_threshold", OVERLAP_THRESHOLD)
        mlflow.log_param("min_per_group", MIN_PER_GROUP)

        rows = []
        pvals = []

        for m in metrics:
            a = df[df["group"]=="PLS"][m].astype(float).dropna().values
            b = df[df["group"]=="TECH"][m].astype(float).dropna().values

            if len(a) < MIN_PER_GROUP or len(b) < MIN_PER_GROUP:
                rows.append({"metric": m, "n_pls": len(a), "n_tech": len(b),
                             "overlap": np.nan, "u": np.nan, "p": np.nan,
                             "cliffs_delta": np.nan, "cohens_d": np.nan,
                             "mean_pls": np.nan, "mean_tech": np.nan,
                             "median_pls": np.nan, "median_tech": np.nan})
                pvals.append(1.0)
                continue

            # estadísticas
            overlap = kde_overlap(a, b)
            res = mannwhitneyu(a, b, alternative="two-sided")
            u, p = float(res.statistic), float(res.pvalue)
            cd = cliffs_delta_from_u(u, len(a), len(b))
            d  = cohens_d(a, b)

            rows.append({
                "metric": m,
                "n_pls": len(a), "n_tech": len(b),
                "overlap": overlap,
                "u": u, "p": p,
                "cliffs_delta": cd, "cohens_d": d,
                "mean_pls": float(np.nanmean(a)), "mean_tech": float(np.nanmean(b)),
                "median_pls": float(np.nanmedian(a)), "median_tech": float(np.nanmedian(b)),
            })
            pvals.append(p)

            # figura (histos superpuestos)
            plt.figure(figsize=(7,4))
            plt.hist(a, bins=40, density=True, alpha=0.45, label="PLS")
            plt.hist(b, bins=40, density=True, alpha=0.45, label="TECH")
            t = f"{m} | overlap≈{overlap:.2f} | p={p:.3g}"
            plt.title(t); plt.xlabel(m); plt.ylabel("densidad"); plt.legend()
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            img = OUT_DIR / "all" / f"{m.replace('/','_')}.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout(); plt.savefig(img, dpi=150); plt.close()
            mlflow.log_artifact(str(img), artifact_path="plots_stats/all")

            if overlap >= OVERLAP_THRESHOLD:
                mlflow.log_artifact(str(img), artifact_path="plots_stats/overlapping")

        stats_df = pd.DataFrame(rows)

        # FDR
        stats_df["p_adj_fdr"] = bh_fdr(np.array(pvals, float))
        stats_df = stats_df.sort_values(["p_adj_fdr","overlap"])

        # guardar tablas
        Path("results").mkdir(exist_ok=True)
        stats_csv = Path("results/stats_summary.csv")
        stats_df.to_csv(stats_csv, index=False)
        mlflow.log_artifact(str(stats_csv), artifact_path="tables")

        overlap_csv = Path("results/top_overlaps.csv")
        stats_df[stats_df["overlap"]>=OVERLAP_THRESHOLD]\
            .sort_values("overlap", ascending=False)\
            .to_csv(overlap_csv, index=False)
        mlflow.log_artifact(str(overlap_csv), artifact_path="tables")

        sig_csv = Path("results/significant_differences.csv")
        stats_df[stats_df["p_adj_fdr"]<=0.05]\
            .sort_values("p_adj_fdr")\
            .to_csv(sig_csv, index=False)
        mlflow.log_artifact(str(sig_csv), artifact_path="tables")

        # métricas globales para la tarjeta de MLflow
        mlflow.log_metrics({
            "n_metrics_tested": int(stats_df.shape[0]),
            "n_significant_fdr05": int((stats_df["p_adj_fdr"]<=0.05).sum()),
            "n_overlapping": int((stats_df["overlap"]>=OVERLAP_THRESHOLD).sum()),
            "mean_overlap": float(stats_df["overlap"].dropna().mean() if "overlap" in stats_df else np.nan)
        })

if __name__ == "__main__":
    main()
