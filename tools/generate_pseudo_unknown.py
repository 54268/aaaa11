from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.modules import generate_hybrid_pseudo_unknown
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, save_json, save_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "base.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(config["project"]["output_dir"])
    boundary_file = np.load(output_dir / "boundary_mining.npz", allow_pickle=True)

    boundary = {
        "local_scale": boundary_file["local_scale"],
        "nearest_foreign": boundary_file["nearest_foreign"],
        "critical_mask": boundary_file["critical_mask"].astype(bool),
        "ordinary_edge_mask": boundary_file["ordinary_edge_mask"].astype(bool),
    }
    pseudo = generate_hybrid_pseudo_unknown(
        embeddings=boundary_file["embeddings"],
        labels=boundary_file["labels"],
        prototypes=boundary_file["prototypes"],
        boundary_result=boundary,
        ordinary_eta=float(config["pseudo_unknown"]["ordinary_eta"]),
        critical_eta=float(config["pseudo_unknown"]["critical_eta"]),
        critical_beta=float(config["pseudo_unknown"]["critical_beta"]),
        ordinary_variations=int(config["pseudo_unknown"]["ordinary_variations"]),
        critical_variations=int(config["pseudo_unknown"]["critical_variations"]),
        jitter=float(config["pseudo_unknown"]["jitter"]),
        seed=int(config["train"]["seed"]),
    )
    save_npz(output_dir / "pseudo_unknown.npz", **pseudo)
    save_json(output_dir / "pseudo_unknown_summary.json", pseudo["summary"])


if __name__ == "__main__":
    main()
