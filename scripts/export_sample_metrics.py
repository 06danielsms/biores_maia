from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import os

IN_CSV  = Path("results/metrics_sample_es.csv")   # salida de compute_metrics_es.py
OUT_DIR = Path("results/export"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not IN_CSV.exists():
        raise SystemExit(f"No existe {IN_CSV}. Corre primero compute_metrics_es.py")

    df = pd.read_csv(IN_CSV)
    drop = {"file","source","group","len_src_chars","len_es_chars"}
    num_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # 1) solo métricas de la muestra (sin columnas administrativas)
    only_metrics = df[num_cols]
    out_only = OUT_DIR / "only_metrics_sample.csv"
    only_metrics.to_csv(out_only, index=False)

    # 2) lista de métricas (nombres, por si te piden “las 64”)
    pd.Series(sorted(num_cols), name="metric").to_csv(OUT_DIR/"metrics_list_sample.csv", index=False)

    # 3) resumen por grupo (PLS vs TECH) y global
    stats = []
    for m in num_cols:
        g = df.groupby("group")[m].agg(["count","mean","median","std","min","max"]).rename_axis(None)
        row = {"metric": m}
        for grp in ["PLS","TECH"]:
            if grp in g.index:
                row.update({
                    f"{grp}_n": int(g.loc[grp,"count"]),
                    f"{grp}_mean": float(g.loc[grp,"mean"]),
                    f"{grp}_median": float(g.loc[grp,"median"]),
                    f"{grp}_std": float(g.loc[grp,"std"]) if pd.notna(g.loc[grp,"std"]) else np.nan,
                    f"{grp}_min": float(g.loc[grp,"min"]),
                    f"{grp}_max": float(g.loc[grp,"max"]),
                })
            else:
                row.update({f"{grp}_n":0, f"{grp}_mean":np.nan, f"{grp}_median":np.nan,
                            f"{grp}_std":np.nan, f"{grp}_min":np.nan, f"{grp}_max":np.nan})
        # global
        g_all = df[m].agg(["count","mean","median","std","min","max"])
        row.update({
            "ALL_n": int(g_all["count"]),
            "ALL_mean": float(g_all["mean"]),
            "ALL_median": float(g_all["median"]),
            "ALL_std": float(g_all["std"]) if pd.notna(g_all["std"]) else np.nan,
            "ALL_min": float(g_all["min"]),
            "ALL_max": float(g_all["max"]),
        })
        stats.append(row)
    summary = pd.DataFrame(stats).sort_values("metric")
    out_summary = OUT_DIR / "summary_by_group_sample.csv"
    summary.to_csv(out_summary, index=False)

    # 4) pequeño overview de la muestra
    overview = {
        "n_rows": len(df),
        "n_metrics": len(num_cols),
        "n_PLS": int((df["group"]=="PLS").sum()) if "group" in df.columns else 0,
        "n_TECH": int((df["group"]=="TECH").sum()) if "group" in df.columns else 0,
    }
    pd.DataFrame([overview]).to_csv(OUT_DIR/"overview_sample.csv", index=False)

    # ----- MLflow (opcional pero útil) -----
    mlflow.set_tracking_uri(f"file://{Path.cwd().absolute()}/mlflow_work")
    mlflow.set_experiment("healthlit_export_es")
    with mlflow.start_run(run_name="export_from_sample") as run:
        mlflow.log_params({"source_csv": str(IN_CSV), **overview})
        mlflow.log_artifact(str(out_only),    artifact_path="tables")
        mlflow.log_artifact(str(out_summary), artifact_path="tables")
        mlflow.log_artifact(str(OUT_DIR/"metrics_list_sample.csv"), artifact_path="tables")
        mlflow.log_artifact(str(OUT_DIR/"overview_sample.csv"),     artifact_path="tables")
    print(f"Listo:\n- {out_only}\n- {out_summary}\n- {OUT_DIR/'metrics_list_sample.csv'}\n- {OUT_DIR/'overview_sample.csv'}")

if __name__ == "__main__":
    main()
