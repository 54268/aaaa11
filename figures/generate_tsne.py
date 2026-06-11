from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.common.io import load_json, load_pickle
from functions.common.seed import set_seed
from functions.data.data_build import build_data_module
from functions.methods.fusion import (
    apply_known_rescue,
    apply_score_calibration,
    apply_unknown_rejection,
    fuse_unknown_score,
    prototype_distance_unknown_score,
)
from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes
from functions.model.closed_set import ClosedSetTrainer


RANDOM_STATE = 42
PCA_DIM = 50
TSNE_PERPLEXITY = 35
MAX_CORRECT_PER_KNOWN_CLASS = 350
MAX_CORRECT_UNKNOWN = 1500
FIG_SIZE = (11.4, 7.6)
FIG_DPI = 300
POINT_SIZE = 9

plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paper-ready open-set t-SNE figure.")
    parser.add_argument("--dataset", choices=["oracle", "wisig"], default="oracle")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def build_experiment_config(dataset: str) -> dict:
    if dataset == "oracle":
        from run_oracle import build_config
    elif dataset == "wisig":
        from run_wisig import build_config
    else:
        raise ValueError("dataset must be oracle or wisig")
    return build_config()


def resolve_checkpoint_path(config: dict, checkpoint_arg: str | None) -> Path:
    checkpoint = Path(checkpoint_arg) if checkpoint_arg else Path(config["project"]["output_dir"]) / "best_closed_set.pt"
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return checkpoint


def check_open_set_artifacts(output_dir: Path) -> None:
    required_files = [output_dir / "openmax.pkl", output_dir / "fusion.json", output_dir / "distance_stats.npz"]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing open-set artifacts:\n" + "\n".join(str(path) for path in missing))


def extract_open_set_results(config: dict, checkpoint_path: Path) -> dict:
    output_dir = Path(config["project"]["output_dir"]).resolve()
    check_open_set_artifacts(output_dir)

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(
        config=config,
        num_classes=datamodule.bundle.num_known_classes,
        signal_length=datamodule.bundle.signal_length,
    )
    trainer.load_checkpoint(checkpoint_path)

    known_payload = trainer.extract_embeddings(datamodule.test_known_dataloader())
    unknown_payload = trainer.extract_embeddings(datamodule.test_unknown_dataloader())

    embeddings = np.concatenate([known_payload["embeddings"], unknown_payload["embeddings"]], axis=0)
    unknown_label = datamodule.bundle.num_known_classes
    y_true = np.concatenate(
        [
            known_payload["labels"],
            np.full(len(unknown_payload["labels"]), unknown_label, dtype=np.int64),
        ],
        axis=0,
    )
    prototypes = known_payload["prototypes"]

    known_pred, _, distances = predict_with_prototypes(embeddings, prototypes, float(config["model"]["temperature"]))

    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    fusion_config = load_json(output_dir / "fusion.json")
    distance_stats = np.load(output_dir / "distance_stats.npz")

    q_om = openmax.predict(activations_from_distances(distances))["unknown_prob"]
    q_pd = prototype_distance_unknown_score(distances, known_pred, distance_stats["mu"], distance_stats["sigma"])
    q_u = fuse_unknown_score(
        q_om,
        q_pd,
        float(fusion_config["fusion_lambda"]),
        mode=str(fusion_config.get("fusion_mode", config["fusion"].get("mode", "linear"))),
    )
    q_u = apply_score_calibration(q_u, known_pred, fusion_config.get("score_calibration"))
    y_pred = apply_unknown_rejection(
        known_pred=known_pred,
        q_u=q_u,
        unknown_label=unknown_label,
        threshold=fusion_config.get("threshold"),
        thresholds_per_class=fusion_config.get("thresholds_per_class"),
    )
    y_pred = apply_known_rescue(
        y_pred=y_pred,
        known_pred=known_pred,
        q_u=q_u,
        distances=distances,
        unknown_label=unknown_label,
        rescue_config=fusion_config.get("known_rescue", config["fusion"].get("known_rescue")),
    )
    return {
        "embeddings": embeddings,
        "prototypes": prototypes,
        "y_true": y_true,
        "y_pred": y_pred,
        "unknown_label": unknown_label,
        "output_dir": output_dir,
    }


def balanced_sample_indices(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    selected: list[np.ndarray] = []

    for class_id in range(unknown_label):
        class_indices = np.flatnonzero(y_true == class_id)
        wrong_indices = class_indices[y_pred[class_indices] != class_id]
        correct_indices = class_indices[y_pred[class_indices] == class_id]
        selected.append(wrong_indices)

        quota = max(0, MAX_CORRECT_PER_KNOWN_CLASS - len(wrong_indices))
        if len(correct_indices) <= quota:
            selected.append(correct_indices)
        elif quota > 0:
            selected.append(rng.choice(correct_indices, size=quota, replace=False))

    unknown_indices = np.flatnonzero(y_true == unknown_label)
    unknown_wrong = unknown_indices[y_pred[unknown_indices] != unknown_label]
    unknown_correct = unknown_indices[y_pred[unknown_indices] == unknown_label]
    selected.append(unknown_wrong)
    if len(unknown_correct) <= MAX_CORRECT_UNKNOWN:
        selected.append(unknown_correct)
    elif len(unknown_correct) > 0:
        selected.append(rng.choice(unknown_correct, size=MAX_CORRECT_UNKNOWN, replace=False))

    sampled = np.concatenate(selected)
    sampled.sort()
    return sampled


def reduce_with_tsne(sampled_embeddings: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint_features = np.vstack([sampled_embeddings, prototypes])
    pca_dim = min(PCA_DIM, joint_features.shape[1], joint_features.shape[0] - 1)
    if pca_dim >= 2 and joint_features.shape[1] > pca_dim:
        joint_features = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit_transform(joint_features)

    perplexity = min(TSNE_PERPLEXITY, max(5, (len(joint_features) - 1) // 3))
    reduced = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=RANDOM_STATE,
    ).fit_transform(joint_features)
    num_samples = len(sampled_embeddings)
    return reduced[:num_samples], reduced[num_samples:]


def plot_global_tsne(
    points_2d: np.ndarray,
    prototypes_2d: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_label: int,
    save_path: Path,
) -> None:
    class_names = [f"K{i + 1}" for i in range(unknown_label)]
    class_colors = [plt.get_cmap("tab20")(i % 20) for i in range(unknown_label)]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.16)
    ax.set_facecolor("#FAFAFA")

    unknown_correct = (y_true == unknown_label) & (y_pred == unknown_label)
    if np.any(unknown_correct):
        ax.scatter(
            points_2d[unknown_correct, 0],
            points_2d[unknown_correct, 1],
            s=8,
            c="lightgray",
            alpha=0.34,
            marker="o",
            linewidths=0,
            zorder=1,
            rasterized=True,
        )

    for class_id in range(unknown_label):
        color = class_colors[class_id]
        class_mask = y_true == class_id
        correct_mask = class_mask & (y_pred == class_id)
        rejected_mask = class_mask & (y_pred == unknown_label)
        confused_mask = class_mask & (y_pred != class_id) & (y_pred != unknown_label)

        if np.any(correct_mask):
            ax.scatter(
                points_2d[correct_mask, 0],
                points_2d[correct_mask, 1],
                s=POINT_SIZE,
                c=[color],
                alpha=0.66,
                marker="o",
                linewidths=0,
                zorder=2,
                rasterized=True,
            )

        if np.any(rejected_mask):
            ax.scatter(
                points_2d[rejected_mask, 0],
                points_2d[rejected_mask, 1],
                s=34,
                c=[color],
                alpha=0.95,
                marker="x",
                linewidths=1.25,
                zorder=4,
                rasterized=True,
            )

        if np.any(confused_mask):
            ax.scatter(
                points_2d[confused_mask, 0],
                points_2d[confused_mask, 1],
                s=35,
                facecolors="none",
                edgecolors=[color],
                alpha=0.95,
                marker="s",
                linewidths=1.15,
                zorder=4,
                rasterized=True,
            )

    unknown_accepted = (y_true == unknown_label) & (y_pred != unknown_label)
    if np.any(unknown_accepted):
        ax.scatter(
            points_2d[unknown_accepted, 0],
            points_2d[unknown_accepted, 1],
            s=34,
            c="black",
            alpha=0.88,
            marker="^",
            linewidths=0,
            zorder=5,
            rasterized=True,
        )

    for class_id in range(unknown_label):
        x, y = prototypes_2d[class_id]
        ax.scatter(
            x,
            y,
            s=230,
            c=[class_colors[class_id]],
            marker="*",
            edgecolors="white",
            linewidths=1.4,
            zorder=6,
        )
        label = ax.annotate(
            class_names[class_id],
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=7,
        )
        label.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground="white")])

    dataset_label = "Oracle" if "oracle" in save_path.stem.lower() else "WiSig"
    ax.set_title(
        f"Open-set embedding distribution on {dataset_label}",
        fontsize=14,
        pad=10,
        fontfamily="Arial",
    )
    ax.set_xlabel("t-SNE 1", fontsize=10, labelpad=7)
    ax.set_ylabel("t-SNE 2", fontsize=10, labelpad=7)
    ax.grid(alpha=0.16, linestyle="-", linewidth=0.6)
    ax.tick_params(labelsize=8, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="gray", markeredgecolor="none", markersize=7, label="Known correct"),
        Line2D([0], [0], marker="x", linestyle="None", color="gray", markersize=7, label="Known rejected"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="gray", markersize=7, label="Known confused"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="lightgray", markeredgecolor="none", markersize=7, label="Unknown correct"),
        Line2D([0], [0], marker="^", linestyle="None", color="black", markersize=7, label="Unknown accepted"),
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=12, label="Known prototype"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0,
    )

    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def print_summary(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> None:
    print("\n===== Open-set evaluation summary =====")
    for class_id in range(unknown_label):
        mask = y_true == class_id
        total = int(mask.sum())
        correct = int(np.sum(y_pred[mask] == class_id))
        rejected = int(np.sum(y_pred[mask] == unknown_label))
        confused = total - correct - rejected
        print(f"K{class_id + 1}: correct={correct}/{total}, rejected={rejected}, confused={confused}")

    unknown_mask = y_true == unknown_label
    unknown_total = int(unknown_mask.sum())
    unknown_correct = int(np.sum(y_pred[unknown_mask] == unknown_label))
    print(f"Unknown: correct reject={unknown_correct}/{unknown_total}, accepted={unknown_total - unknown_correct}")


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_STATE)

    config = build_experiment_config(args.dataset)
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    payload = extract_open_set_results(config, checkpoint_path)

    embeddings = payload["embeddings"]
    prototypes = payload["prototypes"]
    y_true = payload["y_true"]
    y_pred = payload["y_pred"]
    unknown_label = payload["unknown_label"]

    output_name = args.output or f"tsne_{args.dataset}.png"
    save_path = FIGURE_ROOT / output_name

    print(f"Experiment dir: {payload['output_dir']}")
    print(f"Checkpoint: {checkpoint_path}")
    print_summary(y_true, y_pred, unknown_label)

    sampled_indices = balanced_sample_indices(y_true, y_pred, unknown_label)
    sampled_embeddings = embeddings[sampled_indices]
    sampled_true = y_true[sampled_indices]
    sampled_pred = y_pred[sampled_indices]

    print(f"\nTotal test samples: {len(embeddings)}")
    print(f"t-SNE sample count: {len(sampled_embeddings)}")

    points_2d, prototypes_2d = reduce_with_tsne(sampled_embeddings, prototypes)
    plot_global_tsne(points_2d, prototypes_2d, sampled_true, sampled_pred, unknown_label, save_path)

    print(f"\nSaved figure: {save_path}")


if __name__ == "__main__":
    main()
