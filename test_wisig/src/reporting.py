from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("无可用结果。")
    else:
        lines.extend(dataframe_to_markdown(df))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["无可用结果。"]
    cols = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for values in rows:
        lines.append("| " + " | ".join(values) + " |")
    return lines


def summarize_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = ["nmi", "ari", "purity", "hungarian_accuracy"]
    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        row["valid_runs"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _method_order(methods: list[str]) -> list[str]:
    preferred = ["Raw IQ + PCA + K-Means", "Raw IQ + PCA + GMM", "FFT Magnitude + PCA + K-Means"]
    return [m for m in preferred if m in methods] + [m for m in methods if m not in preferred]


def plot_metric_bar(summary: pd.DataFrame, path: Path, title: str, metric: str, category_col: str | None = None) -> None:
    if summary.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(summary) * 0.9), 4.8))
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if category_col and category_col in summary.columns:
        categories = list(summary[category_col].drop_duplicates())
        methods = _method_order(list(summary["method"].drop_duplicates()))
        x = np.arange(len(categories))
        width = 0.8 / max(len(methods), 1)
        for idx, method in enumerate(methods):
            sub = summary[summary["method"] == method].set_index(category_col)
            means = [float(sub.loc[c, mean_col]) if c in sub.index else np.nan for c in categories]
            stds = [float(sub.loc[c, std_col]) if c in sub.index else 0.0 for c in categories]
            ax.bar(x + (idx - (len(methods) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.legend(fontsize=8)
    else:
        ordered = summary.copy()
        if "method" in ordered.columns:
            ordered["method"] = pd.Categorical(ordered["method"], categories=_method_order(list(ordered["method"].unique())), ordered=True)
            ordered = ordered.sort_values("method")
            labels = ordered["method"].astype(str).tolist()
        else:
            labels = [str(i) for i in range(len(ordered))]
        ax.bar(labels, ordered[mean_col], yerr=ordered[std_col], capsize=4)
        ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, true_labels: list[str], pred_labels: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(pred_labels) * 0.55), max(5, len(true_labels) * 0.45)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Hungarian 对齐后的聚类簇")
    ax.set_ylabel("真实 Tx")
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(true_labels)))
    ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    ax.set_yticklabels(true_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
