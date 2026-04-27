from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, load_json


METRIC_KEYS = [
    "overall_accuracy",
    "known_accuracy",
    "macro_f1",
    "auroc",
    "fpr95",
    "unknown_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def _openness_from_counts(num_known: int, num_unknown: int) -> float:
    return 1.0 - math.sqrt((2.0 * num_known) / (2.0 * num_known + num_unknown))


def _load_split_meta(config: dict) -> dict:
    split_file = config.get("prep", {}).get("split_file")
    if split_file and Path(split_file).exists():
        return load_json(split_file)
    return {}


def _plot_openness_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["openness"])
    openness = [row["openness"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for key, label, color in [
        ("overall_accuracy", "Overall Acc", "#1f77b4"),
        ("known_accuracy", "Known Acc", "#2ca02c"),
        ("macro_f1", "Macro-F1", "#ff7f0e"),
        ("auroc", "AUROC", "#9467bd"),
        ("unknown_recall", "Unknown Recall", "#d62728"),
    ]:
        axes[0].plot(openness, [row[key] for row in rows], marker="o", linewidth=2, label=label, color=color)
    axes[0].set_xlabel("Openness")
    axes[0].set_ylabel("Metric Value")
    axes[0].set_title("Openness-Metric Curve")
    axes[0].legend()

    axes[1].plot(openness, [row["fpr95"] for row in rows], marker="o", linewidth=2, color="#8c564b")
    axes[1].set_xlabel("Openness")
    axes[1].set_ylabel("FPR95")
    axes[1].set_title("FPR95 vs Openness")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_accuracy_f1_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["openness"])
    openness = [row["openness"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(openness, [row["overall_accuracy"] for row in rows], marker="o", linewidth=2.2, color="#1f4e79", label="Overall Accuracy")
    ax.plot(openness, [row["macro_f1"] for row in rows], marker="s", linewidth=2.2, color="#e67e22", label="Macro-F1")
    ax.set_xlabel("Openness")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and Macro-F1 vs Openness")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_accuracy_f1_summary(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: row["openness"])
    lines = [
        "# 开放度变化下的准确率与 Macro-F1",
        "",
        "| 实验 | Openness | Overall Accuracy | Macro-F1 | Known Accuracy | Unknown Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['openness']:.4f} | {row['overall_accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['known_accuracy']:.4f} | {row['unknown_recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows = []
    for config_path in args.configs:
        config = load_config(config_path)
        metrics_path = Path(config["project"]["output_dir"]) / "open_set_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        split_meta = _load_split_meta(config)
        known_classes = split_meta.get("known_classes", config.get("prep", {}).get("known_classes", []))
        unknown_classes = split_meta.get("unknown_classes", config.get("prep", {}).get("unknown_classes", []))
        openness = split_meta.get("openness")
        if openness is None and known_classes and unknown_classes:
            openness = _openness_from_counts(len(known_classes), len(unknown_classes))
        row = {
            "experiment": config["project"]["name"],
            "config_path": str(Path(config_path).resolve()),
            "output_dir": str(Path(config["project"]["output_dir"]).resolve()),
            "split_file": str(split_meta.get("split_name", "")),
            "num_known": len(known_classes),
            "num_unknown": len(unknown_classes),
            "openness": float(openness or 0.0),
            "known_classes": ",".join(known_classes),
            "unknown_classes": ",".join(unknown_classes),
        }
        for key in METRIC_KEYS:
            row[key] = float(metrics.get(key, 0.0))
        rows.append(row)

    rows = sorted(rows, key=lambda row: row["openness"])
    if not rows:
        raise RuntimeError("No finished experiment metrics were found for the provided configs.")

    csv_path = output_dir / "osr_results_table.csv"
    md_path = output_dir / "osr_results_table.md"
    png_path = output_dir / "openness_metric_curve.png"
    acc_f1_png = output_dir / "openness_accuracy_f1_curve.png"
    acc_f1_md = output_dir / "openness_accuracy_f1_summary.md"

    import csv

    fieldnames = [
        "experiment",
        "num_known",
        "num_unknown",
        "openness",
        *METRIC_KEYS,
        "config_path",
        "output_dir",
        "split_file",
        "known_classes",
        "unknown_classes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OSR 实验结果表",
        "",
        "| 实验名 | K | U | Openness | Overall Accuracy | Known-class Accuracy | Macro-F1 | AUROC | FPR95 | Unknown Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['num_known']} | {row['num_unknown']} | {row['openness']:.4f} | "
            f"{row['overall_accuracy']:.4f} | {row['known_accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['auroc']:.4f} | {row['fpr95']:.4f} | {row['unknown_recall']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _plot_openness_curve(png_path, rows)
    _plot_accuracy_f1_curve(acc_f1_png, rows)
    _write_accuracy_f1_summary(acc_f1_md, rows)


if __name__ == "__main__":
    main()
