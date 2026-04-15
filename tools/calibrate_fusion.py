from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.calibrators import OpenMaxCalibrator, prototype_distance_unknown_score, search_fusion_params
from sei_osr.datasets import build_data_module
from sei_osr.modules.prototype_utils import activations_from_distances, predict_with_prototypes
from sei_osr.trainers import ClosedSetTrainer
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, load_pickle, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--ckpt", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(config["project"]["output_dir"])
    ckpt_path = Path(args.ckpt) if args.ckpt else output_dir / "best_closed_set.pt"

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(ckpt_path)

    val_payload = trainer.extract_embeddings(datamodule.val_known_dataloader())
    pseudo_file = np.load(output_dir / "pseudo_unknown.npz", allow_pickle=True)
    stats_file = np.load(output_dir / "distance_stats.npz")
    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    unknown_label = datamodule.bundle.num_known_classes

    val_pred_proto, _, val_dist = predict_with_prototypes(
        val_payload["embeddings"],
        val_payload["prototypes"],
        float(config["model"]["temperature"]),
    )
    val_openmax = openmax.predict(activations_from_distances(val_dist))
    val_q_om = val_openmax["unknown_prob"]
    val_q_pd = prototype_distance_unknown_score(
        val_dist,
        val_pred_proto,
        stats_file["mu"],
        stats_file["sigma"],
    )

    pseudo_pred_proto, _, pseudo_dist = predict_with_prototypes(
        pseudo_file["pseudo_embeddings"],
        val_payload["prototypes"],
        float(config["model"]["temperature"]),
    )
    pseudo_openmax = openmax.predict(activations_from_distances(pseudo_dist))
    pseudo_q_om = pseudo_openmax["unknown_prob"]
    pseudo_q_pd = prototype_distance_unknown_score(
        pseudo_dist,
        pseudo_pred_proto,
        stats_file["mu"],
        stats_file["sigma"],
    )

    y_true = np.concatenate(
        [
            val_payload["labels"],
            np.full(len(pseudo_file["pseudo_embeddings"]), unknown_label, dtype=np.int64),
        ]
    )
    known_pred = np.concatenate([val_pred_proto, pseudo_pred_proto])
    q_om = np.concatenate([val_q_om, pseudo_q_om])
    q_pd = np.concatenate([val_q_pd, pseudo_q_pd])

    result = search_fusion_params(
        y_true=y_true,
        known_pred=known_pred,
        q_om=q_om,
        q_pd=q_pd,
        unknown_label=unknown_label,
        lambda_grid=config["fusion"]["lambda_grid"],
        threshold_grid=config["fusion"]["threshold_grid"],
        selection_weights=config["fusion"].get("selection_weights"),
        known_quantile_floor=config["fusion"].get("known_quantile_floor"),
        min_known_accuracy=config["fusion"].get("min_known_accuracy"),
        threshold_mode=str(config["fusion"].get("threshold_mode", "global")),
        classwise_quantile_grid=config["fusion"].get("classwise_quantile_grid"),
    )
    save_json(
        output_dir / "fusion.json",
        {
            "fusion_lambda": result.fusion_lambda,
            "threshold": result.threshold,
            "thresholds_per_class": result.thresholds_per_class,
            "threshold_mode": result.threshold_mode,
            "threshold_quantile": result.threshold_quantile,
            "metrics": result.metrics,
        },
    )


if __name__ == "__main__":
    main()
