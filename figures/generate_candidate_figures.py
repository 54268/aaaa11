from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures"

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9.5


OPEN_SET = {
    "Oracle": {
        "labels": ["Softmax", "OpenMax", "HyperRSI", "HyDRA", "OpenRFI", "ARPL", "PCBM"],
        "unknown_recall": [0.536417, 0.886708, 0.895458, 0.805417, 0.945667, 0.882875, 0.967042],
        "known_accuracy": [0.934125, 0.795875, 0.891500, 0.882500, 0.899625, 0.893750, 0.963625],
        "macro_f1": [0.721127, 0.804256, 0.866924, 0.826454, 0.905725, 0.854296, 0.936905],
        "auroc": [0.900906, 0.900686, 0.951477, 0.904814, 0.969528, 0.932765, 0.986836],
    },
    "WiSig": {
        "labels": ["Softmax", "OpenMax", "HyperRSI", "HyDRA", "OpenRFI", "ARPL", "PCBM"],
        "unknown_recall": [0.906250, 0.997917, 1.000000, 1.000000, 0.924479, 1.000000, 1.000000],
        "known_accuracy": [0.952734, 0.899219, 0.894141, 0.900391, 0.900391, 0.899609, 0.986719],
        "macro_f1": [0.916164, 0.928624, 0.939272, 0.946136, 0.891645, 0.948643, 0.993554],
        "auroc": [0.940071, 0.988474, 0.998671, 0.998989, 0.978104, 0.992654, 0.995952],
    },
}

MODULE_ABLATION = {
    "Oracle": {
        "labels": ["Base", "+OpenMax", "+ProtoDist", "Full"],
        "known_accuracy": [0.842625, 0.874875, 0.902250, 0.963625],
        "unknown_recall": [0.976792, 0.995250, 0.991958, 0.967042],
        "macro_f1": [0.892250, 0.929271, 0.936424, 0.936905],
        "auroc": [0.957690, 0.978594, 0.986702, 0.986836],
    },
    "WiSig": {
        "labels": ["Base", "+OpenMax", "+ProtoDist", "Full"],
        "known_accuracy": [0.853906, 0.888672, 0.897266, 0.986719],
        "unknown_recall": [0.967500, 1.000000, 1.000000, 1.000000],
        "macro_f1": [0.838442, 0.943172, 0.947862, 0.993554],
        "auroc": [0.983290, 0.987252, 0.995949, 0.995952],
    },
}

KM_SENSITIVITY = {
    "Oracle": {
        "labels": ["m=0", "m=1", "m=2", "m=3", "Auto"],
        "adjusted_quality": [0.745862, 0.841655, 0.885741, 0.864713, 0.885741],
        "coverage": [0.914250, 0.909000, 0.886292, 0.895542, 0.886292],
        "resolved_k": [5, 6, 6, 7, 6],
        "selected": "Auto -> m=2",
    },
    "WiSig": {
        "labels": ["m=0", "m=1", "m=2", "m=3", "Auto"],
        "adjusted_quality": [0.998878, 0.975676, 0.954992, 0.914294, 0.998878],
        "coverage": [1.000000, 0.993437, 0.991354, 0.949271, 1.000000],
        "resolved_k": [12, 13, 14, 14, 12],
        "selected": "Auto -> m=0",
    },
}

FLOW_ABLATION = {
    "Oracle": {
        "labels": ["Embedding", "I/Q", "Fusion", "Full"],
        "nmi": [0.853374, 0.716079, 0.904883, 0.999540],
        "ari": [0.751999, 0.593146, 0.787661, 0.999758],
        "hungarian_accuracy": [0.796252, 0.661353, 0.801637, 0.999904],
        "coverage": [0.967083, 0.967083, 0.967083, 0.863833],
    },
    "WiSig": {
        "labels": ["Embedding", "I/Q", "Fusion", "Full"],
        "nmi": [0.997803, 0.884918, 0.998125, 0.998125],
        "ari": [0.998409, 0.746998, 0.998637, 0.998637],
        "hungarian_accuracy": [0.999271, 0.734583, 0.999375, 0.999375],
        "coverage": [1.000000, 1.000000, 1.000000, 1.000000],
    },
}

BACKEND_COMPARISON = {
    "Oracle": {
        "labels": ["K-means", "HDBSCAN", "OpenRFI", "PCBM (ours)"],
        "nmi": [0.852434, 0.684820, 0.931928, 0.998816],
        "ari": [0.832207, 0.425885, 0.909938, 0.999263],
        "hungarian_accuracy": [0.921844, 0.487485, 0.930196, 0.999718],
        "coverage": [0.967083, 0.699167, 0.850000, 0.886292],
    },
}

METRIC_TITLES = {
    "unknown_recall": "Unknown Recall",
    "known_accuracy": "Known Acc.",
    "macro_f1": "Macro F1",
    "auroc": "AUROC",
    "nmi": "NMI",
    "ari": "ARI",
    "hungarian_accuracy": "Hungarian Acc.",
    "coverage": "Coverage",
}

METHOD_COLORS = {
    "Softmax": "#8FAFD3",
    "OpenMax": "#E7BE8B",
    "HyperRSI": "#A7C88D",
    "HyDRA": "#D9A6A1",
    "OpenRFI": "#B8A8D6",
    "ARPL": "#8EC3C7",
    "PCBM": "#9FB2C7",
    "PCBM (ours)": "#9FB2C7",
    "K-means": "#8FAFD3",
    "HDBSCAN": "#E7BE8B",
}

LINE_COLOR = "#1F4E79"
ACCENT_COLOR = "#C43C39"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def _ylim_for(values: list[float], floor_zero: bool = False) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 0.015)
    pad = max(span * 0.18, 0.015)
    bottom = 0.0 if floor_zero else max(0.0, lo - pad)
    top = min(1.05, hi + pad)
    if top - bottom < 0.04:
        mid = (top + bottom) / 2.0
        bottom = max(0.0, mid - 0.025)
        top = min(1.05, mid + 0.025)
    return bottom, top


def _annotate_values(ax: plt.Axes, xs: np.ndarray, values: list[float], y_offset: float = 0.006) -> None:
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * y_offset
    for x, value in zip(xs, values):
        ax.text(
            x,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#222222",
        )


def plot_bar_grid(
    data: dict[str, dict[str, list[float] | list[str]]],
    metrics: list[str],
    title: str,
    output_name: str,
) -> Path:
    dataset_names = list(data.keys())
    fig, axes = plt.subplots(
        len(dataset_names),
        len(metrics),
        figsize=(4.1 * len(metrics), 3.15 * len(dataset_names)),
        squeeze=False,
    )
    for col, metric in enumerate(metrics):
        all_values: list[float] = []
        for dataset in dataset_names:
            all_values.extend(data[dataset][metric])  # type: ignore[arg-type]
        y_limits = _ylim_for(all_values)
        for row, dataset in enumerate(dataset_names):
            ax = axes[row, col]
            labels = list(data[dataset]["labels"])  # type: ignore[arg-type]
            values = list(data[dataset][metric])  # type: ignore[arg-type]
            xs = np.arange(len(labels))
            colors = [METHOD_COLORS.get(label, "#6B7280") for label in labels]
            ax.bar(xs, values, color=colors, width=0.72)
            ax.set_ylim(*y_limits)
            ax.set_title(METRIC_TITLES[metric], fontsize=11, pad=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.0)
            if col == 0:
                ax.set_ylabel(dataset, fontsize=12, fontweight="bold")
            _style_axis(ax)
            _annotate_values(ax, xs, values)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.015)
    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_open_set_comparison(dataset: str) -> Path:
    metrics = ["known_accuracy", "unknown_recall", "macro_f1", "auroc"]
    data = OPEN_SET[dataset]
    labels = list(data["labels"])
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.7), squeeze=False)
    xs = np.arange(len(labels))
    colors = [METHOD_COLORS.get(label, "#6B7280") for label in labels]
    all_values = [value for metric in metrics for value in data[metric]]
    y_limits = _ylim_for(all_values)

    for ax, metric in zip(axes.flat, metrics):
        values = list(data[metric])
        ax.bar(xs, values, color=colors, width=0.72)
        ax.set_ylim(*y_limits)
        ax.set_title(METRIC_TITLES[metric], fontsize=11, pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8.6)
        _style_axis(ax)
        _annotate_values(ax, xs, values, y_offset=0.008)

    axes[0, 0].set_ylabel(dataset, fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel(dataset, fontsize=12, fontweight="bold")
    fig.suptitle(f"{dataset} 开放集对比实验", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=2.0, w_pad=1.7)
    output_path = OUTPUT_DIR / f"open_set_comparison_{dataset.lower()}.png"
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_line_grid(
    data: dict[str, dict[str, list[float] | list[str]]],
    metrics: list[str],
    title: str,
    output_name: str,
) -> Path:
    dataset_names = list(data.keys())
    fig, axes = plt.subplots(
        len(dataset_names),
        len(metrics),
        figsize=(4.1 * len(metrics), 3.2 * len(dataset_names)),
        squeeze=False,
    )
    for col, metric in enumerate(metrics):
        all_values: list[float] = []
        for dataset in dataset_names:
            all_values.extend(data[dataset][metric])  # type: ignore[arg-type]
        y_limits = _ylim_for(all_values)
        for row, dataset in enumerate(dataset_names):
            ax = axes[row, col]
            labels = list(data[dataset]["labels"])  # type: ignore[arg-type]
            values = list(data[dataset][metric])  # type: ignore[arg-type]
            xs = np.arange(len(labels))
            ax.plot(
                xs,
                values,
                color=LINE_COLOR,
                linewidth=2.4,
                marker="o",
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=2.0,
            )
            ax.set_ylim(*y_limits)
            ax.set_title(METRIC_TITLES[metric], fontsize=11, pad=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.2)
            if col == 0:
                ax.set_ylabel(dataset, fontsize=12, fontweight="bold")
            _style_axis(ax)
            _annotate_values(ax, xs, values, y_offset=0.012)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.015)
    fig.tight_layout(h_pad=2.4, w_pad=1.6)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_km_sensitivity() -> Path:
    dataset_names = list(KM_SENSITIVITY.keys())
    fig, axes = plt.subplots(len(dataset_names), 1, figsize=(8.2, 6.2), squeeze=False)
    all_values = [
        value
        for dataset in dataset_names
        for value in KM_SENSITIVITY[dataset]["adjusted_quality"]  # type: ignore[index]
    ]
    y_limits = _ylim_for(all_values)
    for row, dataset in enumerate(dataset_names):
        ax = axes[row, 0]
        labels = list(KM_SENSITIVITY[dataset]["labels"])  # type: ignore[arg-type]
        values = list(KM_SENSITIVITY[dataset]["adjusted_quality"])  # type: ignore[arg-type]
        coverage = list(KM_SENSITIVITY[dataset]["coverage"])  # type: ignore[arg-type]
        resolved_k = list(KM_SENSITIVITY[dataset]["resolved_k"])  # type: ignore[arg-type]
        xs = np.arange(len(labels))
        ax.plot(
            xs,
            values,
            color=LINE_COLOR,
            linewidth=2.6,
            marker="o",
            markersize=7.0,
            markerfacecolor="white",
            markeredgewidth=2.1,
        )
        best_idx = int(np.argmax(values))
        ax.scatter([best_idx], [values[best_idx]], s=95, color=ACCENT_COLOR, zorder=5)
        ax.set_ylim(*y_limits)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel(f"{dataset}\nAdjusted Quality", fontsize=11, fontweight="bold")
        ax.set_title(str(KM_SENSITIVITY[dataset]["selected"]), fontsize=11, pad=8)
        _style_axis(ax)
        y0, y1 = ax.get_ylim()
        for x, value, cov, k in zip(xs, values, coverage, resolved_k):
            ax.text(
                x,
                value + (y1 - y0) * 0.025,
                f"{value:.3f}\nK={k}, C={cov:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#222222",
            )
    fig.suptitle("K+M 缓冲分量敏感性", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=2.0)
    output_path = OUTPUT_DIR / "km_sensitivity.png"
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_backend_comparison() -> Path:
    dataset = "Oracle"
    metrics = ["nmi", "ari", "hungarian_accuracy", "coverage"]
    labels = list(BACKEND_COMPARISON[dataset]["labels"])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), squeeze=False)
    xs = np.arange(len(labels))
    colors = [METHOD_COLORS.get(label, "#6B7280") for label in labels]
    all_values = [
        value
        for metric in metrics
        for value in BACKEND_COMPARISON[dataset][metric]
    ]
    y_limits = _ylim_for(all_values)

    for ax, metric in zip(axes.flat, metrics):
        values = list(BACKEND_COMPARISON[dataset][metric])
        ax.bar(xs, values, color=colors, width=0.68)
        ax.set_ylim(*y_limits)
        ax.set_title(METRIC_TITLES[metric], fontsize=11, pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8.6)
        _style_axis(ax)
        _annotate_values(ax, xs, values, y_offset=0.008)

    axes[0, 0].set_ylabel(dataset, fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel(dataset, fontsize=12, fontweight="bold")
    fig.suptitle("未知类细分方法对比", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=2.0, w_pad=1.7)
    output_path = OUTPUT_DIR / "backend_comparison.png"
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_index(paths: list[Path]) -> Path:
    lines = [
        "# 论文候选图",
        "",
        "本目录保存当前整理后的论文候选图。",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    index_path = OUTPUT_DIR / "README.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legacy_open_set = OUTPUT_DIR / "open_set_comparison.png"
    if legacy_open_set.exists():
        legacy_open_set.unlink()
    paths = [
        plot_open_set_comparison("Oracle"),
        plot_open_set_comparison("WiSig"),
        plot_km_sensitivity(),
        plot_backend_comparison(),
    ]
    index_path = write_index(paths)
    for path in paths:
        print(path)
    print(index_path)


if __name__ == "__main__":
    main()
