from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.calibrators import OpenMaxCalibrator, apply_unknown_rejection, fuse_unknown_score, prototype_distance_unknown_score
from sei_osr.datasets import build_data_module
from sei_osr.modules.prototype_utils import activations_from_distances, predict_with_prototypes
from sei_osr.trainers import ClosedSetTrainer
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, load_json, load_pickle, save_json
from sei_osr.utils.metrics import evaluate_open_set, save_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=str, required=True)
    parser.add_argument("--target-config", type=str, required=True)
    parser.add_argument("--source-ckpt", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config = load_config(args.source_config)
    target_config = load_config(args.target_config)

    source_output = Path(source_config["project"]["output_dir"])
    target_tag = target_config["project"]["name"]
    output_dir = ensure_dir(source_output / f"cross_to_{target_tag}")
    ckpt_path = Path(args.source_ckpt) if args.source_ckpt else source_output / "best_closed_set.pt"

    source_dm = build_data_module(source_config)
    target_dm = build_data_module(target_config)

    trainer = ClosedSetTrainer(
        source_config,
        source_dm.bundle.num_known_classes,
        source_dm.bundle.signal_length,
    )
    trainer.load_checkpoint(ckpt_path)

    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(source_output / "openmax.pkl"))
    fusion_cfg = load_json(source_output / "fusion.json")
    stats_file = np.load(source_output / "distance_stats.npz")
    unknown_label = source_dm.bundle.num_known_classes

    source_known = trainer.extract_embeddings(source_dm.test_known_dataloader())
    target_known = trainer.extract_embeddings(target_dm.test_known_dataloader())
    target_unknown = trainer.extract_embeddings(target_dm.test_unknown_dataloader())

    target_all_embeddings = np.concatenate([target_known["embeddings"], target_unknown["embeddings"]], axis=0)
    target_all_unknown = np.full(len(target_all_embeddings), unknown_label, dtype=np.int64)

    all_embeddings = np.concatenate([source_known["embeddings"], target_all_embeddings], axis=0)
    y_true = np.concatenate([source_known["labels"], target_all_unknown], axis=0)

    pred_proto, _, distances = predict_with_prototypes(
        all_embeddings,
        source_known["prototypes"],
        float(source_config["model"]["temperature"]),
    )
    openmax_out = openmax.predict(activations_from_distances(distances))
    q_om = openmax_out["unknown_prob"]
    q_pd = prototype_distance_unknown_score(distances, pred_proto, stats_file["mu"], stats_file["sigma"])
    q_u = fuse_unknown_score(q_om, q_pd, float(fusion_cfg["fusion_lambda"]))
    y_pred = apply_unknown_rejection(
        known_pred=pred_proto,
        q_u=q_u,
        unknown_label=unknown_label,
        threshold=fusion_cfg.get("threshold"),
        thresholds_per_class=fusion_cfg.get("thresholds_per_class"),
    )

    metrics = evaluate_open_set(y_true, y_pred, q_u, unknown_label)
    d_min = distances[np.arange(len(distances)), pred_proto]
    save_json(
        output_dir / "cross_open_set_metrics.json",
        {
            **metrics,
            "source_config": str(Path(args.source_config).resolve()),
            "target_config": str(Path(args.target_config).resolve()),
            "protocol": "source known-test + target all-samples-as-unknown",
        },
    )
    save_prediction_csv(output_dir / "cross_open_set_predictions.csv", y_true, y_pred, q_u, q_om, q_pd, d_min)


if __name__ == "__main__":
    main()
