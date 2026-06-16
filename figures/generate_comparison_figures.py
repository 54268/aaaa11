from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parent
COMPARISON_ROOT = PROJECT_ROOT / "Comparison method" / "adapted_results" / "formal"

METHOD_SPECS = [
    ("Softmax", "softmax", "#4E79A7"),
    ("OpenMax", "openmax", "#F28E2B"),
    ("HyperRSI", "hyperrsi", "#59A14F"),
    ("HyDRA", "hydra", "#E15759"),
    ("OpenRFI", "openrfi", "#B07AA1"),
    ("ARPL", "arpl_evt", "#76B7B2"),
    ("PCBM (ours)", None, "#111111"),
]

DATASETS = {
    "oracle": {
        "display": "Oracle",
        "comparison_name": "oracle_kri16_demod",
        "ours": PROJECT_ROOT / "outputs" / "oracle_kri16_demod_known_first" / "open_set_predictions.csv",
        "unknown_label": 10,
    },
    "wisig": {
        "display": "WiSig",
        "comparison_name": "wisig_singleday_osr_k16_u12",
        "ours": PROJECT_ROOT / "outputs" / "wisig_singleday_osr_k16_u12" / "open_set_predictions.csv",
        "unknown_label": 16,
    },
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


@dataclass(frozen=True)
class PredictionData:
    y_true: np.ndarray
    y_pred: np.ndarray
    unknown_score: np.ndarray
    unknown_label: int


def load_prediction_csv(path: str | Path, fallback_unknown_label: int | None = None) -> PredictionData:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")

    unknown_label_text = rows[0].get("unknown_label", "")
    if unknown_label_text != "":
        unknown_label = int(float(unknown_label_text))
    elif fallback_unknown_label is not None:
        unknown_label = int(fallback_unknown_label)
    else:
        raise ValueError(f"Unknown label is not available in {path}")

    return PredictionData(
        y_true=np.asarray([int(float(row["y_true"])) for row in rows], dtype=np.int64),
        y_pred=np.asarray([int(float(row["y_pred"])) for row in rows], dtype=np.int64),
        unknown_score=np.asarray([float(row["unknown_score"]) for row in rows], dtype=np.float64),
        unknown_label=unknown_label,
    )


def _baseline_prediction_path(dataset_name: str, method_name: str, subdir: str) -> Path:
    directory = COMPARISON_ROOT / dataset_name / subdir
    matches = sorted(directory.glob(f"{method_name}_{dataset_name}_seed*_predictions.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Missing per-sample predictions for {method_name} on {dataset_name}. "
            "Run Comparison method/adapted_baselines/run_comparison.py first."
        )
    return matches[-1]


def _prediction_sets(dataset_key: str) -> list[tuple[str, str, PredictionData]]:
    spec = DATASETS[dataset_key]
    rows: list[tuple[str, str, PredictionData]] = []
    for method_name, subdir, color in METHOD_SPECS:
        if subdir is None:
            path = Path(spec["ours"])
        else:
            path = _baseline_prediction_path(str(spec["comparison_name"]), method_name, subdir)
        rows.append(
            (
                method_name,
                color,
                load_prediction_csv(path, fallback_unknown_label=int(spec["unknown_label"])),
            )
        )
    return rows


def confusion_output_paths(output_dir: str | Path = FIGURE_ROOT) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        dataset_key: output_dir / f"confusion_matrix_{dataset_key}.png"
        for dataset_key in DATASETS
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)


def plot_roc_comparison(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)
    for panel_index, (dataset_key, ax) in enumerate(zip(["oracle", "wisig"], axes)):
        for method_name, color, data in _prediction_sets(dataset_key):
            binary = (data.y_true == data.unknown_label).astype(np.int32)
            fpr, tpr, _ = roc_curve(binary, data.unknown_score)
            score_auc = auc(fpr, tpr)
            is_ours = method_name == "PCBM (ours)"
            ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2.8 if is_ours else 1.75,
                linestyle="-" if is_ours else None,
                label=f"{method_name} ({score_auc:.3f})",
                zorder=5 if is_ours else 2,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#9E9E9E", zorder=1)
        ax.set_title(f"({chr(97 + panel_index)}) {DATASETS[dataset_key]['display']}", fontsize=13, pad=9)
        ax.set_xlabel("False Positive Rate")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.legend(loc="lower right", frameon=False, fontsize=8.7, handlelength=2.5)
        _style_axis(ax)
    axes[0].set_ylabel("True Positive Rate")
    fig.tight_layout(w_pad=2.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _draw_confusion(ax: plt.Axes, data: PredictionData, title: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = list(range(data.unknown_label + 1))
    matrix = confusion_matrix(data.y_true, data.y_pred, labels=labels)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        np.maximum(row_sum, 1),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sum > 0,
    )
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    display_labels = [f"K{i + 1}" for i in range(data.unknown_label)] + ["U"]
    compact = len(display_labels) <= 12
    tick_fontsize = 9.0 if compact else 7.5
    diagonal_fontsize = 8.7 if compact else 7.7
    off_diagonal_fontsize = 8.0 if compact else 7.0
    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(display_labels, fontsize=tick_fontsize)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = float(normalized[row, col])
            count = int(matrix[row, col])
            if count <= 0:
                continue
            if row == col or value >= 0.01:
                is_diag = row == col
                text = f"{count}\n{100.0 * value:.1f}%"
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=diagonal_fontsize if is_diag else off_diagonal_fontsize,
                    color="white" if is_diag and value >= 0.52 else ("#b22222" if not is_diag else "#222222"),
                    fontweight="bold" if is_diag else "normal",
                )
    return image, matrix, display_labels


def plot_confusion_matrix(dataset_key: str, output_path: str | Path) -> Path:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    output_path = Path(output_path)
    spec = DATASETS[dataset_key]
    data = load_prediction_csv(
        spec["ours"],
        fallback_unknown_label=int(spec["unknown_label"]),
    )
    class_count = data.unknown_label + 1
    figsize = (7.4, 6.5) if class_count <= 12 else (9.4, 8.2)
    fig, ax = plt.subplots(figsize=figsize)
    image, _, _ = _draw_confusion(ax, data, f"{spec['display']} confusion matrix")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized rate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(output_dir: str | Path = FIGURE_ROOT) -> dict[str, Path]:
    paths = confusion_output_paths(output_dir)
    return {
        dataset_key: plot_confusion_matrix(dataset_key, output_path)
        for dataset_key, output_path in paths.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready comparison figures.")
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=FIGURE_ROOT / "roc_comparison_oracle_wisig.png",
    )
    parser.add_argument(
        "--confusion-dir",
        type=Path,
        default=FIGURE_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"ROC figure: {plot_roc_comparison(args.roc_output)}")
    for dataset_key, output_path in plot_confusion_matrices(args.confusion_dir).items():
        print(f"{DATASETS[dataset_key]['display']} confusion figure: {output_path}")


if __name__ == "__main__":
    main()
