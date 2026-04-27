from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.datasets import build_data_module
from sei_osr.trainers import ClosedSetTrainer
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, save_json
from sei_osr.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "oracle_osr_main.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["train"]["seed"]))
    output_dir = ensure_dir(config["project"]["output_dir"])

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(
        config=config,
        num_classes=datamodule.bundle.num_known_classes,
        signal_length=datamodule.bundle.signal_length,
    )
    artifacts = trainer.fit(datamodule.train_dataloader(), datamodule.val_known_dataloader(), output_dir)
    save_json(
        output_dir / "train_summary.json",
        {
            "checkpoint_path": artifacts.checkpoint_path,
            "best_val_acc": artifacts.best_val_acc,
            "num_known_classes": datamodule.bundle.num_known_classes,
            "signal_length": datamodule.bundle.signal_length,
        },
    )


if __name__ == "__main__":
    main()
