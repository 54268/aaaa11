from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.datasets.prep_wisig import _load_wisig_payload
from sei_osr.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=str, default=str(ROOT / "data" / "raw" / "wisig" / "SingleDay.pkl"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "splits" / "wisig" / "single_day_rx1_eq0"))
    parser.add_argument("--rx", nargs="+", default=["1-1"])
    parser.add_argument("--capture-dates", nargs="+", default=["2021_03_23"])
    parser.add_argument("--equalized", nargs="+", type=int, default=[0])
    parser.add_argument("--known-count", type=int, default=16)
    parser.add_argument("--unknown-counts", type=str, default="4,8,12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--candidate-classes", type=str, default="")
    return parser.parse_args()


def _count_samples(payload: dict, tx_name: str, rx_set: set[str], date_set: set[str], eq_set: set[int]) -> int:
    tx_idx = payload["tx_list"].index(tx_name)
    count = 0
    for rx_idx, rx_name in enumerate(payload["rx_list"]):
        if rx_name not in rx_set:
            continue
        for date_idx, capture_date in enumerate(payload["capture_date_list"]):
            if capture_date not in date_set:
                continue
            for eq_idx, eq_value in enumerate(payload["equalized_list"]):
                if eq_value not in eq_set:
                    continue
                count += int(len(payload["data"][tx_idx][rx_idx][date_idx][eq_idx]))
    return count


def _openness(num_known: int, num_unknown: int) -> float:
    return 1.0 - math.sqrt((2.0 * num_known) / (2.0 * num_known + num_unknown))


def main() -> None:
    args = parse_args()
    payload = _load_wisig_payload(args.raw_path)
    output_dir = ensure_dir(args.output_dir)

    rx_set = set(args.rx)
    date_set = set(args.capture_dates)
    eq_set = set(args.equalized)

    counts = {
        tx_name: _count_samples(payload, tx_name, rx_set, date_set, eq_set)
        for tx_name in payload["tx_list"]
    }
    candidate_classes = [tx_name for tx_name, count in counts.items() if count > 0]
    if args.candidate_classes:
        requested = [item.strip() for item in args.candidate_classes.split(",") if item.strip()]
        candidate_classes = [tx for tx in requested if counts.get(tx, 0) > 0]
    candidate_classes = sorted(candidate_classes)

    unknown_counts = [int(item) for item in args.unknown_counts.split(",") if item.strip()]
    max_unknown = max(unknown_counts)
    total_needed = int(args.known_count) + max_unknown
    if len(candidate_classes) < total_needed:
        raise RuntimeError(
            f"Not enough candidate transmitter classes under controlled filters: "
            f"need {total_needed}, got {len(candidate_classes)}."
        )

    import numpy as np

    rng = np.random.default_rng(int(args.seed))
    shuffled = candidate_classes.copy()
    rng.shuffle(shuffled)
    selected_pool = shuffled[:total_needed]
    known_classes = selected_pool[: int(args.known_count)]
    unknown_pool = selected_pool[int(args.known_count) :]

    chosen_counts = [counts[class_name] for class_name in selected_pool]
    samples_per_class = int(args.samples_per_class) if int(args.samples_per_class) > 0 else int(min(chosen_counts))

    with (output_dir / "candidate_class_counts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "available_samples"])
        for class_name in candidate_classes:
            writer.writerow([class_name, counts[class_name]])

    split_rows = []
    for unknown_count in unknown_counts:
        unknown_classes = unknown_pool[:unknown_count]
        split_name = f"wisig_single_day_rx1_eq0_k{args.known_count}_u{unknown_count}_seed{args.seed}"
        split_payload = {
            "dataset": "wisig_controlled_pool",
            "pool_name": "SingleDay_fixed_rx_eq",
            "split_name": split_name,
            "raw_path": str(Path(args.raw_path).resolve()),
            "include_rx": sorted(rx_set),
            "include_capture_dates": sorted(date_set),
            "include_equalized": sorted(eq_set),
            "candidate_classes": candidate_classes,
            "selected_pool": selected_pool,
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "known_count": int(args.known_count),
            "unknown_count": int(unknown_count),
            "samples_per_class": samples_per_class,
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "openness": _openness(int(args.known_count), int(unknown_count)),
            "available_samples_per_class": {class_name: int(counts[class_name]) for class_name in selected_pool},
        }
        split_path = output_dir / f"{split_name}.json"
        save_json(split_path, split_payload)
        split_rows.append(
            {
                "split_name": split_name,
                "split_file": str(split_path.resolve()),
                "known_count": int(args.known_count),
                "unknown_count": int(unknown_count),
                "openness": split_payload["openness"],
                "samples_per_class": samples_per_class,
            }
        )

    save_json(
        output_dir / "split_index.json",
        {
            "pool_name": "SingleDay_fixed_rx_eq",
            "raw_path": str(Path(args.raw_path).resolve()),
            "include_rx": sorted(rx_set),
            "include_capture_dates": sorted(date_set),
            "include_equalized": sorted(eq_set),
            "seed": int(args.seed),
            "candidate_classes": candidate_classes,
            "selected_pool": selected_pool,
            "known_base_classes": known_classes,
            "unknown_pool": unknown_pool,
            "samples_per_class": samples_per_class,
            "splits": split_rows,
        },
    )


if __name__ == "__main__":
    main()
