import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics(cfg: dict) -> pd.DataFrame:
    metrics_path = cfg["metrics"]["output_csv"]
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No existe: {metrics_path}")
    df = pd.read_csv(metrics_path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(",", "_")
    )


def plot_metric(metric_name: str, value_pls: float, value_no: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    labels = ["PLS", "NO_PLS"]
    values = [value_pls, value_no]
    plt.bar(labels, values)
    plt.title(metric_name)
    plt.ylabel("Valor")
    plt.tight_layout()
    fname = safe_name(metric_name) + ".png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Guardado: {out_path}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    df = load_metrics(cfg)

    out_dir = "metrics/plots_metrics"

    for metric in df.index:
        v_pls = df.loc[metric, "PLS"]
        v_no = df.loc[metric, "NO_PLS"]
        plot_metric(metric, v_pls, v_no, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)

