from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.datasets.prep_oracle import _extract_label, _label_sort_key
from sei_osr.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(ROOT / "data" / "raw" / "oracle" / "KRI-16IQImbalances-DemodulatedData"),
    )
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "splits" / "oracle"))
    parser.add_argument("--known-count", type=int, default=10)
    parser.add_argument("--unknown-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-regex", type=str, default=r"IQ#(\d+)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-segments-per-label", type=int, default=4000)
    return parser.parse_args()


def _openness(num_known: int, num_unknown: int) -> float:
    return 1.0 - math.sqrt((2.0 * num_known) / (2.0 * num_known + num_unknown))


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = ensure_dir(args.output_dir)
    meta_files = sorted(raw_root.rglob("*.sigmf-meta"))
    labels = sorted(
        {
            _extract_label(meta_path, args.label_regex)
            for meta_path in meta_files
        },
        key=_label_sort_key,
    )
    total_needed = int(args.known_count) + int(args.unknown_count)
    if len(labels) < total_needed:
        raise RuntimeError(f"Need {total_needed} Oracle labels, but only found {len(labels)}.")

    import numpy as np

    rng = np.random.default_rng(int(args.seed))
    order = labels.copy()
    rng.shuffle(order)
    chosen = order[:total_needed]
    known_classes = chosen[: int(args.known_count)]
    unknown_classes = chosen[int(args.known_count) :]

    split_name = f"oracle_k{args.known_count}_u{args.unknown_count}_seed{args.seed}"
    payload = {
        "dataset": "oracle_sigmf",
        "split_name": split_name,
        "raw_root": str(raw_root.resolve()),
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "known_count": int(args.known_count),
        "unknown_count": int(args.unknown_count),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "segment_length": int(args.segment_length),
        "stride": int(args.stride),
        "max_segments_per_label": int(args.max_segments_per_label),
        "label_regex": args.label_regex,
        "seed": int(args.seed),
        "openness": _openness(int(args.known_count), int(args.unknown_count)),
        "selected_pool": chosen,
    }
    save_json(output_dir / f"{split_name}.json", payload)


if __name__ == "__main__":
    main()
