from __future__ import annotations

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


def _class_axis_labels(config: dict, unknown_label: int, known_names: np.ndarray | None) -> list[str]:
    split_payload = _load_split_payload(config)
    known = list(split_payload.get("known_classes", []))
    if not known:
        known = list(config.get("prep", {}).get("known_classes", []))
    if not known and known_names is not None:
        known = [str(name) for name in np.unique(known_names).tolist()]
    if not known:
        known = [f"class_{idx}" for idx in range(unknown_label)]
    return known + ["Unknown"]


def _plot_roc(path: Path, y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> None:
    binary = (y_true == unknown_label).astype(np.int32)
    fpr, tpr, _ = roc_curve(binary, unknown_score)
    score_auc = auc(fpr, tpr)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(fpr, tpr, label=f"AUROC = {score_auc:.4f}", linewidth=2.4, color="#1f77b4")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#ff7f0e", alpha=0.85)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_pr(path: Path, y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> None:
    binary = (y_true == unknown_label).astype(np.int32)
    precision, recall, _ = precision_recall_curve(binary, unknown_score)
    pr_auc = auc(recall, precision)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2.4, color="#2ca02c")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.set_ylim(max(0.0, float(np.nanmin(precision)) - 0.03), 1.01)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str], unknown_label: int) -> None:
    labels = list(range(unknown_label + 1))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, np.maximum(row_sum, 1), where=row_sum > 0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-wise Ratio")
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Row-normalized)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] <= 0:
                continue
            text = f"{int(matrix[i, j])}\n{norm[i, j] * 100:.1f}%"
            color = "white" if norm[i, j] >= 0.55 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)

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

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [2.2, 1.2]})

    axes[0].hist(known, bins=bins, alpha=0.68, color="#1f77b4", label=f"Known ({len(known)})")
    if np.std(unknown) < 1e-6:
        x0 = float(unknown[0]) if len(unknown) else 1.0
        axes[0].axvline(x0, color="#ff7f0e", linewidth=3.0, label=f"Unknown ({len(unknown)})")
        axes[0].annotate(
            f"Unknown scores collapse at {x0:.3f}",
            xy=(x0, axes[0].get_ylim()[1] * 0.85),
            xytext=(max(lo, x0 - 0.22), axes[0].get_ylim()[1] * 0.92),
            arrowprops={"arrowstyle": "->", "color": "#ff7f0e"},
            fontsize=9,
            color="#cc5500",
        )
    else:
        axes[0].hist(unknown, bins=bins, alpha=0.68, color="#ff7f0e", label=f"Unknown ({len(unknown)})")
    if threshold is not None:
        axes[0].axvline(float(threshold), color="#2ca02c", linestyle="--", linewidth=2.0, label=f"Threshold = {threshold:.3f}")
    axes[0].set_xlabel("Unknown Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Unknown Score Distribution")
    axes[0].legend()

    zoom_lo = max(lo, 0.85)
    zoom_hi = hi + 1e-6
    zoom_bins = np.linspace(zoom_lo, zoom_hi, 31) if zoom_hi > zoom_lo else np.linspace(0.85, 1.0, 31)
    axes[1].hist(known, bins=zoom_bins, alpha=0.68, color="#1f77b4")
    if np.std(unknown) < 1e-6:
        x0 = float(unknown[0]) if len(unknown) else 1.0
        axes[1].axvline(x0, color="#ff7f0e", linewidth=3.0)
    else:
        axes[1].hist(unknown, bins=zoom_bins, alpha=0.68, color="#ff7f0e")
    if threshold is not None:
        axes[1].axvline(float(threshold), color="#2ca02c", linestyle="--", linewidth=2.0)
    axes[1].set_xlim(zoom_lo, zoom_hi)
    axes[1].set_xlabel("Unknown Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("High-score Zoom")

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
                "known_accuracy": float(subset_metrics["known_accuracy"]),
                "macro_f1": float(subset_metrics["macro_f1"]),
                "auroc": float(subset_metrics["auroc"]),
                "known_fpr_as_unknown": float(subset_metrics["known_fpr_as_unknown"]),
            }
        )
    return rows


def _plot_openness_curve(path: Path, curve_rows: list[Dict[str, float]]) -> None:
    if not curve_rows:
        return
    openness = [row["openness"] for row in curve_rows]
    series_specs = [
        ("known_accuracy", "Known Accuracy", "#1f77b4"),
        ("macro_f1", "Macro-F1", "#ff7f0e"),
        ("auroc", "AUROC", "#2ca02c"),
        ("known_fpr_as_unknown", "Known FPR as Unknown", "#d62728"),
    ]
    active_specs = []
    for key, label, color in series_specs:
        values = np.asarray([row[key] for row in curve_rows], dtype=np.float64)
        if np.max(values) - np.min(values) > 1e-4:
            active_specs.append((key, label, color, values))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    if active_specs:
        for _, label, color, values in active_specs:
            ax.plot(openness, values, marker="o", linewidth=2.2, label=label, color=color)
        stacked = np.concatenate([values for _, _, _, values in active_specs])
        ymin = max(0.0, float(stacked.min()) - 0.05)
        ymax = min(1.02, float(stacked.max()) + 0.05)
        ax.set_ylim(ymin, ymax)
    else:
        ax.text(
            0.5,
            0.5,
            "Metrics are nearly unchanged within this fixed-known split.\nUse the cross-config openness curve for the main comparison.",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Openness")
    ax.set_ylabel("Metric")
    ax.set_title("Within-split Unknown Accumulation Curve")
    if active_specs:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
    figure_dir = ensure_dir(project_root / "figures" / output_dir.name)

    label_names, known_names = _load_test_arrays(config["data"]["root"])
    class_labels = _class_axis_labels(config, unknown_label, known_names)

    _plot_roc(figure_dir / "roc_curve.png", y_true, unknown_score, unknown_label)
    _plot_pr(figure_dir / "pr_curve.png", y_true, unknown_score, unknown_label)
    _plot_confusion(figure_dir / "confusion_matrix.png", y_true, y_pred, class_labels, unknown_label)
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
        _plot_openness_curve(figure_dir / "openness_metric_curve.png", curve_rows)

    index = {
        "figure_dir": str(figure_dir),
        "files": sorted([path.name for path in figure_dir.iterdir() if path.is_file()]),
    }
    save_json(figure_dir / "figures_index.json", index)
    return figure_dir
