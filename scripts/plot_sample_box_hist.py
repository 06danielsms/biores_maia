from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

# ---- Config ----
IN_CSV = Path(os.getenv("IN_CSV", "results/metrics_sample_es.csv"))  # la muestra
PLOTS_DIR = Path("results/plots_sample")
HIST_DIR = PLOTS_DIR / "hist"
BOX_DIR  = PLOTS_DIR / "box"

MIN_PER_GROUP = int(os.getenv("MIN_PER_GROUP", "3"))
HIST_BINS = int(os.getenv("HIST_BINS", "40"))

DROP = {"file","source","group","len_src_chars","len_es_chars"}

def main():
    if not IN_CSV.exists():
        raise SystemExit(f"No existe {IN_CSV}. Corre primero compute_metrics_es.py (ES offline).")

    df = pd.read_csv(IN_CSV)
    num_cols = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    BOX_DIR.mkdir(parents=True, exist_ok=True)

    # --- MLflow ---
    mlflow.set_tracking_uri(f"file://{Path.cwd().absolute()}/mlflow_work")
    mlflow.set_experiment("healthlit_sample_plots_es")
    with mlflow.start_run(run_name="box_hist_sample") as run:
        mlflow.log_param("input_csv", str(IN_CSV))
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_metrics", len(num_cols))
        mlflow.log_param("hist_bins", HIST_BINS)
        mlflow.log_param("min_per_group", MIN_PER_GROUP)

        plotted = []

        for m in num_cols:
            a = df[df["group"]=="PLS"][m].astype(float).replace([np.inf,-np.inf], np.nan).dropna().values
            b = df[df["group"]=="TECH"][m].astype(float).replace([np.inf,-np.inf], np.nan).dropna().values

            # Histograma (si hay algo en al menos 1 grupo)
            if len(a) >= 1 or len(b) >= 1:
                plt.figure(figsize=(7,4))
                if len(a) >= 1:
                    plt.hist(a, bins=HIST_BINS, density=True, alpha=0.45, label="PLS")
                if len(b) >= 1:
                    plt.hist(b, bins=HIST_BINS, density=True, alpha=0.45, label="TECH")
                plt.title(f"{m} — hist")
                plt.xlabel(m); plt.ylabel("densidad"); plt.legend()
                out_h = HIST_DIR / f"{m.replace('/','_')}.png"
                plt.tight_layout(); plt.savefig(out_h, dpi=150); plt.close()
                mlflow.log_artifact(str(out_h), artifact_path="plots_sample/hist")

            # Boxplot (requiere mínimo en cada grupo para comparar)
            if len(a) >= MIN_PER_GROUP and len(b) >= MIN_PER_GROUP:
                plt.figure(figsize=(6,4))
                plt.boxplot([a, b], labels=["PLS","TECH"], showfliers=False)
                plt.title(f"{m} — boxplot (n≥{MIN_PER_GROUP} por grupo)")
                plt.ylabel(m)
                out_b = BOX_DIR / f"{m.replace('/','_')}.png"
                plt.tight_layout(); plt.savefig(out_b, dpi=150); plt.close()
                mlflow.log_artifact(str(out_b), artifact_path="plots_sample/box")
                plotted.append(m)

        # Resumen de las métricas con boxplot válido
        pd.Series(plotted, name="metric")\
          .to_csv(PLOTS_DIR/"metrics_with_boxplot.csv", index=False)
        mlflow.log_artifact(str(PLOTS_DIR/"metrics_with_boxplot.csv"), artifact_path="tables")

        # También loguea los directorios completos (por si miras desde la UI)
        if HIST_DIR.exists():
            mlflow.log_artifacts(str(HIST_DIR), artifact_path="plots_sample/hist")
        if BOX_DIR.exists():
            mlflow.log_artifacts(str(BOX_DIR), artifact_path="plots_sample/box")

        mlflow.log_metrics({
            "n_hist_plots": len(list(HIST_DIR.glob("*.png"))),
            "n_box_plots": len(list(BOX_DIR.glob("*.png")))
        })

if __name__ == "__main__":
    main()
