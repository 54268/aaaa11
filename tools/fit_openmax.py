from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.calibrators import OpenMaxCalibrator
from sei_osr.datasets import build_data_module
from sei_osr.modules.prototype_utils import activations_from_distances, collect_distance_stats
from sei_osr.trainers import ClosedSetTrainer
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, save_json, save_npz, save_pickle


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
    payload = trainer.extract_embeddings(datamodule.val_known_dataloader())

    activations = activations_from_distances(payload["distances"])
    openmax = OpenMaxCalibrator(
        alpha_rank=int(config["openmax"]["alpha_rank"]),
        tail_size=int(config["openmax"]["tail_size"]),
        backend=str(config["openmax"].get("backend", "native")),
        distance_type=str(config["openmax"].get("distance_type", "eucl")),
        euclid_weight=float(config["openmax"].get("euclid_weight", 1.0)),
    )
    openmax.fit(activations, payload["labels"], payload["pred"])
    save_pickle(output_dir / "openmax.pkl", openmax.state_dict())

    stats = collect_distance_stats(
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        prototypes=payload["prototypes"],
        temperature=float(config["model"]["temperature"]),
    )
    save_npz(
        output_dir / "distance_stats.npz",
        mu=stats["mu"],
        sigma=stats["sigma"],
        prototypes=payload["prototypes"],
    )
    save_json(
        output_dir / "openmax_summary.json",
        {
            "alpha_rank": int(config["openmax"]["alpha_rank"]),
            "tail_size": int(config["openmax"]["tail_size"]),
            "backend": str(config["openmax"].get("backend", "native")),
            "num_val_samples": int(len(payload["labels"])),
        },
    )


if __name__ == "__main__":
    main()
