from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.datasets import build_data_module
from sei_osr.modules import mine_boundary_samples
from sei_osr.trainers import ClosedSetTrainer
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, save_json, save_npz


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

    payload = trainer.extract_embeddings(datamodule.train_dataloader())
    boundary = mine_boundary_samples(
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        prototypes=payload["prototypes"],
        k=int(config["boundary"]["k"]),
        alpha=float(config["boundary"]["alpha"]),
        top_m=int(config["boundary"]["top_m"]),
        ordinary_edge_ratio=float(config["boundary"]["ordinary_edge_ratio"]),
    )
    save_npz(
        output_dir / "boundary_mining.npz",
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        prototypes=payload["prototypes"],
        scores=boundary["scores"],
        local_edge=boundary["local_edge"],
        gap=boundary["gap"],
        local_scale=boundary["local_scale"],
        nearest_foreign=boundary["nearest_foreign"],
        critical_mask=boundary["critical_mask"].astype("int64"),
        ordinary_edge_mask=boundary["ordinary_edge_mask"].astype("int64"),
    )
    save_json(output_dir / "boundary_summary.json", boundary["summary"])


if __name__ == "__main__":
    main()
