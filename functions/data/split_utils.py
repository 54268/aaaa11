from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Tuple

from functions.common.io import ensure_dir, load_json


SPLIT_OVERRIDE_KEYS = {
    "raw_path",
    "raw_paths",
    "raw_root",
    "known_classes",
    "unknown_classes",
    "num_known_classes",
    "include_rx",
    "include_capture_dates",
    "include_equalized",
    "split_mode",
    "fixed_rx",
    "samples_per_class",
    "max_samples_per_tx",
    "train_ratio",
    "val_ratio",
    "segment_length",
    "stride",
    "max_segments_per_label",
    "label_regex",
}

PATH_OVERRIDE_KEYS = {"raw_path", "raw_paths", "raw_root"}


def _path_exists(path_value: Any) -> bool:
    if isinstance(path_value, list):
        return all(Path(item).expanduser().exists() for item in path_value)
    if isinstance(path_value, str):
        return Path(path_value).expanduser().exists()
    return False


def merge_prep_with_split(prep_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    merged = dict(prep_cfg)
    split_file = merged.get("split_file")
    if not split_file:
        return merged, None
    split_payload = load_json(split_file)
    for key in SPLIT_OVERRIDE_KEYS:
        if key in split_payload:
            if key in PATH_OVERRIDE_KEYS and key in merged and not _path_exists(split_payload[key]):
                continue
            merged[key] = split_payload[key]
    merged["_split_payload"] = split_payload
    return merged, split_payload


def write_class_split_csv(path: str | Path, known_classes: list[str], unknown_classes: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "role"])
        for class_name in known_classes:
            writer.writerow([class_name, "known"])
        for class_name in unknown_classes:
            writer.writerow([class_name, "unknown"])



