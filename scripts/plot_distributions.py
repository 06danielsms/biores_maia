import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_hist(df, metric, output_dir, bins=50):
    if metric not in df.columns:
        print(f"[WARN] {metric} no está en el dataset, skip.")
        return

    plt.figure(figsize=(8, 5))

    data_no = df[df["label"] == "no_pls"][metric].dropna()
    data_pl = df[df["label"] == "pls"][metric].dropna()

    plt.hist(data_no, bins=bins, alpha=0.5, label="NO_PLS", density=True)
    plt.hist(data_pl, bins=bins, alpha=0.5, label="PLS", density=True)

    plt.title(f"Distribución: {metric}")
    plt.xlabel(metric)
    plt.ylabel("Densidad")
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Guardado: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet con métricas (train o test)")
    parser.add_argument("--output", required=True, help="Carpeta donde guardar PNGs")
    parser.add_argument("--metrics", nargs="*", default=[], help="Lista de métricas a graficar (opcional)")
    args = parser.parse_args()

    print(f"Cargando {args.input} ...")
    df = pd.read_parquet(args.input)

    if "label" not in df.columns:
        raise ValueError("El dataset no tiene columna 'label'")

    # seleccionar métricas numéricas automáticamente si no se dan
    if not args.metrics:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["doc_id", "unit_id", "n_chars", "n_words"]
        metrics = [c for c in numeric_cols if c not in exclude]
    else:
        metrics = args.metrics

    print(f"Generando histogramas para {len(metrics)} métricas...")

    for m in metrics:
        plot_hist(df, m, args.output)

    print("Finalizado.")

if __name__ == "__main__":
    main()

