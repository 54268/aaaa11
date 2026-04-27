from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from .io import ensure_dir, load_json, save_json
from .metrics import evaluate_open_set


def _load_split_payload(config: dict) -> dict:
    split_file = config.get("prep", {}).get("split_file")
    if split_file and Path(split_file).exists():
        return load_json(split_file)
    return {}


def _load_test_arrays(data_root: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    root = Path(data_root)
    known = np.load(root / "test_known.npz", allow_pickle=True)
    unknown = np.load(root / "test_unknown.npz", allow_pickle=True)
    label_names = None
    if "label_name" in known.files and "label_name" in unknown.files:
        label_names = np.concatenate([known["label_name"], unknown["label_name"]], axis=0)
    known_names = known["label_name"] if "label_name" in known.files else None
    return label_names, known_names


def _raw_known_labels(config: dict, known_names: np.ndarray | None) -> list[str]:
    split_payload = _load_split_payload(config)
    known = list(split_payload.get("known_classes", []))
    if not known:
        known = list(config.get("prep", {}).get("known_classes", []))
    if not known and known_names is not None:
        known = [str(name) for name in np.unique(known_names).tolist()]
    return [str(item) for item in known]


def _confusion_label_pack(config: dict, unknown_label: int, known_names: np.ndarray | None) -> tuple[list[str], list[dict[str, str]]]:
    raw_known = _raw_known_labels(config, known_names)
    if not raw_known:
        raw_known = [f"class_{idx}" for idx in range(unknown_label)]

    display = [f"K{idx + 1}" for idx in range(len(raw_known))] + ["Unknown"]
    mapping = [{"display_label": disp, "raw_label": raw} for disp, raw in zip(display[:-1], raw_known)]
    mapping.append({"display_label": "Unknown", "raw_label": "unknown"})
    return display, mapping


def _save_confusion_label_map(path: Path, mapping: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["display_label", "raw_label"])
        writer.writeheader()
        writer.writerows(mapping)


def _paper_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)


def _plot_roc(path: Path, y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> None:
    binary = (y_true == unknown_label).astype(np.int32)
    fpr, tpr, _ = roc_curve(binary, unknown_score)
    score_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(fpr, tpr, label=f"AUROC = {score_auc:.4f}", linewidth=2.3, color="#1f4e79")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#b0b0b0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(frameon=False, loc="lower right")
    _paper_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_pr(path: Path, y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> None:
    binary = (y_true == unknown_label).astype(np.int32)
    precision, recall, _ = precision_recall_curve(binary, unknown_score)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2.3, color="#2b8c56")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.set_ylim(max(0.0, float(np.nanmin(precision)) - 0.03), 1.01)
    ax.legend(frameon=False, loc="lower left")
    _paper_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str], unknown_label: int) -> None:
    labels = list(range(unknown_label + 1))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, np.maximum(row_sum, 1), where=row_sum > 0)

    n = len(class_labels)
    fig_size = max(7.5, 0.6 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized by True Class")
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix")
    ax.grid(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ratio = float(norm[i, j])
            count = int(matrix[i, j])
            if count == 0:
                continue
            if ratio < 0.015 and i != j:
                continue
            text = f"{ratio * 100:.1f}%"
            if i == j or ratio >= 0.08:
                text = f"{count}\n{ratio * 100:.1f}%"
            color = "white" if ratio >= 0.45 else "#1a1a1a"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_score_histogram(
    path: Path,
    y_true: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
    threshold: float | None = None,
) -> None:
    known = unknown_score[y_true != unknown_label]
    unknown = unknown_score[y_true == unknown_label]

    lo = float(min(known.min(initial=0.0), unknown.min(initial=0.0)))
    hi = float(max(known.max(initial=1.0), unknown.max(initial=1.0)))
    bins = np.linspace(lo, hi, 41) if hi > lo else np.linspace(0.0, 1.0, 41)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [2.3, 1.2]})

    axes[0].hist(known, bins=bins, alpha=0.72, color="#4c78a8", label=f"Known ({len(known)})", edgecolor="white")
    if np.std(unknown) < 1e-6:
        x0 = float(unknown[0]) if len(unknown) else 1.0
        axes[0].axvline(x0, color="#f58518", linewidth=3.0, label=f"Unknown ({len(unknown)})")
        axes[0].annotate(
            f"Unknown scores collapse at {x0:.3f}",
            xy=(x0, axes[0].get_ylim()[1] * 0.85),
            xytext=(max(lo, x0 - 0.22), axes[0].get_ylim()[1] * 0.92),
            arrowprops={"arrowstyle": "->", "color": "#f58518"},
            fontsize=9,
            color="#b15928",
        )
    else:
        axes[0].hist(unknown, bins=bins, alpha=0.68, color="#f58518", label=f"Unknown ({len(unknown)})", edgecolor="white")
    if threshold is not None:
        axes[0].axvline(float(threshold), color="#54a24b", linestyle="--", linewidth=2.0, label=f"Threshold = {threshold:.3f}")
    axes[0].set_xlabel("Unknown Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Unknown Score Distribution")
    axes[0].legend(frameon=False)

    zoom_lo = max(lo, hi - max(0.18, (hi - lo) * 0.35))
    zoom_hi = hi + 1e-6
    zoom_bins = np.linspace(zoom_lo, zoom_hi, 31) if zoom_hi > zoom_lo else np.linspace(0.85, 1.0, 31)
    axes[1].hist(known, bins=zoom_bins, alpha=0.72, color="#4c78a8", edgecolor="white")
    if np.std(unknown) < 1e-6:
        x0 = float(unknown[0]) if len(unknown) else 1.0
        axes[1].axvline(x0, color="#f58518", linewidth=3.0)
    else:
        axes[1].hist(unknown, bins=zoom_bins, alpha=0.68, color="#f58518", edgecolor="white")
    if threshold is not None:
        axes[1].axvline(float(threshold), color="#54a24b", linestyle="--", linewidth=2.0)
    axes[1].set_xlim(zoom_lo, zoom_hi)
    axes[1].set_xlabel("Unknown Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("High-score Zoom")

    for ax in axes:
        _paper_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _build_openness_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    label_names: np.ndarray | None,
    unknown_label: int,
    num_known_classes: int,
) -> list[Dict[str, float]]:
    if label_names is None:
        return []
    unknown_names = label_names[y_true == unknown_label]
    if len(unknown_names) == 0:
        return []
    unique_unknown, counts = np.unique(unknown_names, return_counts=True)
    order = np.argsort(-counts)
    ordered_unknown = unique_unknown[order]

    rows: list[Dict[str, float]] = []
    for m in range(1, len(ordered_unknown) + 1):
        chosen = set(ordered_unknown[:m].tolist())
        mask = (y_true != unknown_label) | np.isin(label_names, list(chosen))
        subset_metrics = evaluate_open_set(y_true[mask], y_pred[mask], unknown_score[mask], unknown_label)
        openness = 1.0 - np.sqrt((2.0 * num_known_classes) / (2.0 * num_known_classes + m))
        rows.append(
            {
                "num_unknown_classes": float(m),
                "openness": float(openness),
                "overall_accuracy": float(subset_metrics["overall_accuracy"]),
                "known_accuracy": float(subset_metrics["known_accuracy"]),
                "macro_f1": float(subset_metrics["macro_f1"]),
                "unknown_recall": float(subset_metrics["unknown_recall"]),
                "auroc": float(subset_metrics["auroc"]),
                "fpr95": float(subset_metrics["fpr95"]),
            }
        )
    return rows


def _plot_openness_metric_curve(path: Path, curve_rows: list[Dict[str, float]]) -> None:
    if not curve_rows:
        return
    openness = [row["openness"] for row in curve_rows]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for key, label, color in [
        ("overall_accuracy", "Overall Accuracy", "#1f4e79"),
        ("known_accuracy", "Known Accuracy", "#2b8c56"),
        ("macro_f1", "Macro-F1", "#e67e22"),
        ("unknown_recall", "Unknown Recall", "#c0392b"),
        ("auroc", "AUROC", "#7f3c8d"),
    ]:
        ax.plot(openness, [row[key] for row in curve_rows], marker="o", linewidth=2.1, label=label, color=color)
    ax.set_xlabel("Openness")
    ax.set_ylabel("Metric Value")
    ax.set_title("Openness-Metric Curve")
    ax.legend(frameon=False, loc="best")
    _paper_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_accuracy_f1_curve(path: Path, curve_rows: list[Dict[str, float]]) -> None:
    if not curve_rows:
        return
    openness = [row["openness"] for row in curve_rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(openness, [row["overall_accuracy"] for row in curve_rows], marker="o", linewidth=2.2, color="#1f4e79", label="Overall Accuracy")
    ax.plot(openness, [row["macro_f1"] for row in curve_rows], marker="s", linewidth=2.2, color="#e67e22", label="Macro-F1")
    ax.set_xlabel("Openness")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and Macro-F1 vs Openness")
    ax.legend(frameon=False, loc="best")
    _paper_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_openness_text(path: Path, curve_rows: list[Dict[str, float]]) -> None:
    if not curve_rows:
        return
    rows = sorted(curve_rows, key=lambda row: row["openness"])
    lines = [
        "# 开放度变化下的准确率与 Macro-F1",
        "",
        "| Openness | Overall Accuracy | Macro-F1 | Known Accuracy | Unknown Recall |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['openness']:.4f} | {row['overall_accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['known_accuracy']:.4f} | {row['unknown_recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_open_set_figures(
    config: dict,
    output_dir: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
    threshold: float | None = None,
) -> Path:
    output_dir = Path(output_dir)
    project_root = Path(config["_project_root"])
    if bool(config.get("reporting", {}).get("write_figures_inside_output_dir", False)):
        figure_dir = ensure_dir(output_dir / "figures")
    else:
        figure_dir = ensure_dir(project_root / "figures" / output_dir.name)

    label_names, known_names = _load_test_arrays(config["data"]["root"])
    class_labels, label_mapping = _confusion_label_pack(config, unknown_label, known_names)

    _plot_roc(figure_dir / "roc_curve.png", y_true, unknown_score, unknown_label)
    _plot_pr(figure_dir / "pr_curve.png", y_true, unknown_score, unknown_label)
    _plot_confusion(figure_dir / "confusion_matrix.png", y_true, y_pred, class_labels, unknown_label)
    _save_confusion_label_map(figure_dir / "confusion_label_map.csv", label_mapping)
    _plot_score_histogram(figure_dir / "unknown_score_histogram.png", y_true, unknown_score, unknown_label, threshold)

    curve_rows = _build_openness_curve(
        y_true=y_true,
        y_pred=y_pred,
        unknown_score=unknown_score,
        label_names=label_names,
        unknown_label=unknown_label,
        num_known_classes=unknown_label,
    )
    if curve_rows:
        save_json(figure_dir / "openness_curve.json", curve_rows)
        _plot_openness_metric_curve(figure_dir / "openness_metric_curve.png", curve_rows)
        _plot_accuracy_f1_curve(figure_dir / "openness_accuracy_f1_curve.png", curve_rows)
        _write_openness_text(figure_dir / "openness_accuracy_f1_summary.md", curve_rows)

    index = {
        "figure_dir": str(figure_dir),
        "files": sorted([path.name for path in figure_dir.iterdir() if path.is_file()]),
        "notes": {"confusion_label_map": "confusion_label_map.csv stores display label to raw class label mapping."},
    }
    save_json(figure_dir / "figures_index.json", index)
    return figure_dir
