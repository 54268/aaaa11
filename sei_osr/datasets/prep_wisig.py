from __future__ import annotations

import hashlib
import pickle
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np

from sei_osr.datasets.split_utils import merge_prep_with_split, write_class_split_csv
from sei_osr.utils.io import ensure_dir, save_json, save_npz


def _resolve_wisig_path(raw_path: str | Path) -> Path:
    raw_path = Path(raw_path)
    if raw_path.exists():
        return raw_path
    unpacked_candidate = raw_path.parent / f"{raw_path.stem}_unpacked" / raw_path.name
    if unpacked_candidate.exists():
        return unpacked_candidate
    raise FileNotFoundError(f"WiSig raw file not found: {raw_path}")


def _load_wisig_payload(raw_path: str | Path) -> Dict:
    raw_path = _resolve_wisig_path(raw_path)
    if zipfile.is_zipfile(raw_path):
        with zipfile.ZipFile(raw_path) as zf:
            inner_name = zf.namelist()[0]
            extract_dir = raw_path.parent / f"{raw_path.stem}_unpacked"
            ensure_dir(extract_dir)
            inner_path = extract_dir / inner_name
            if not inner_path.exists():
                zf.extractall(extract_dir)
        raw_path = inner_path
    with raw_path.open("rb") as f:
        return pickle.load(f)


def _stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, np.ndarray]:
    train_idx = []
    val_idx = []
    test_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train : n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val :])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def _deduplicate_samples(samples: np.ndarray) -> np.ndarray:
    seen = set()
    keep = []
    for idx, sample in enumerate(samples):
        key = hashlib.sha1(np.ascontiguousarray(sample).view(np.uint8)).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    return samples[np.asarray(keep, dtype=np.int64)]


def prepare_wisig_compact(config: dict) -> Dict[str, object]:
    prep_cfg, split_payload = merge_prep_with_split(config["prep"])
    raw_paths = prep_cfg.get("raw_paths")
    if raw_paths:
        raw_paths = [str(Path(p)) for p in raw_paths]
    else:
        raw_paths = [str(prep_cfg["raw_path"])]
    output_root = Path(prep_cfg["processed_root"])
    ensure_dir(output_root)

    rng = np.random.default_rng(int(config["train"]["seed"]))
    max_per_tx = int(prep_cfg.get("max_samples_per_tx", 0))
    samples_per_class = int(prep_cfg.get("samples_per_class", 0))
    deduplicate_exact = bool(prep_cfg.get("deduplicate_exact", len(raw_paths) > 1))

    payloads = [_load_wisig_payload(raw_path) for raw_path in raw_paths]
    tx_list = []
    for payload in payloads:
        for tx_name in payload["tx_list"]:
            if tx_name not in tx_list:
                tx_list.append(tx_name)

    include_equalized = prep_cfg.get("include_equalized")
    include_rx = prep_cfg.get("include_rx")
    include_capture_dates = prep_cfg.get("include_capture_dates")

    known_classes = prep_cfg.get("known_classes")
    unknown_classes = prep_cfg.get("unknown_classes")
    if not known_classes or not unknown_classes:
        split_idx = int(prep_cfg.get("num_known_classes", max(len(tx_list) - 2, 1)))
        known_classes = tx_list[:split_idx]
        unknown_classes = tx_list[split_idx:]

    class_to_idx = {name: idx for idx, name in enumerate(known_classes)}
    all_known_x = []
    all_known_y = []
    all_known_name = []
    all_unknown_x = []
    all_unknown_y = []
    all_unknown_name = []

    for tx_name in tx_list:
        samples_acc = []
        for payload in payloads:
            if tx_name not in payload["tx_list"]:
                continue
            payload_tx_idx = payload["tx_list"].index(tx_name)
            payload_rx = payload["rx_list"]
            payload_dates = payload["capture_date_list"]
            payload_equalized = payload["equalized_list"]
            use_equalized = set(include_equalized if include_equalized is not None else payload_equalized)
            use_rx = set(include_rx if include_rx is not None else payload_rx)
            use_dates = set(include_capture_dates if include_capture_dates is not None else payload_dates)

            for rx_idx, rx_name in enumerate(payload_rx):
                if rx_name not in use_rx:
                    continue
                for date_idx, capture_date in enumerate(payload_dates):
                    if capture_date not in use_dates:
                        continue
                    for eq_idx, eq_value in enumerate(payload_equalized):
                        if eq_value not in use_equalized:
                            continue
                        block = payload["data"][payload_tx_idx][rx_idx][date_idx][eq_idx]
                        block = np.asarray(block, dtype=np.float32)
                        block = np.transpose(block, (0, 2, 1))
                        samples_acc.append(block)
        if not samples_acc:
            continue
        tx_samples = np.concatenate(samples_acc, axis=0)
        if deduplicate_exact:
            tx_samples = _deduplicate_samples(tx_samples)
        if samples_per_class > 0:
            if len(tx_samples) < samples_per_class:
                continue
            choice = rng.choice(len(tx_samples), size=samples_per_class, replace=False)
            tx_samples = tx_samples[choice]
        if max_per_tx > 0 and len(tx_samples) > max_per_tx:
            choice = rng.choice(len(tx_samples), size=max_per_tx, replace=False)
            tx_samples = tx_samples[choice]

        if tx_name in class_to_idx:
            all_known_x.append(tx_samples)
            all_known_y.append(np.full(len(tx_samples), class_to_idx[tx_name], dtype=np.int64))
            all_known_name.append(np.full(len(tx_samples), tx_name))
        elif tx_name in unknown_classes:
            all_unknown_x.append(tx_samples)
            all_unknown_y.append(np.full(len(tx_samples), -1, dtype=np.int64))
            all_unknown_name.append(np.full(len(tx_samples), tx_name))

    if not all_known_x:
        raise RuntimeError("No known-class WiSig samples were selected. Check prep filters.")

    x_known = np.concatenate(all_known_x, axis=0)
    y_known = np.concatenate(all_known_y, axis=0)
    name_known = np.concatenate(all_known_name, axis=0)
    x_unknown = np.concatenate(all_unknown_x, axis=0) if all_unknown_x else np.zeros((0, 2, 256), dtype=np.float32)
    y_unknown = np.concatenate(all_unknown_y, axis=0) if all_unknown_y else np.zeros((0,), dtype=np.int64)
    name_unknown = np.concatenate(all_unknown_name, axis=0) if all_unknown_name else np.zeros((0,), dtype="<U1")

    splits = _stratified_split(
        x_known,
        y_known,
        rng=rng,
        train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
        val_ratio=float(prep_cfg.get("val_ratio", 0.1)),
    )
    save_npz(
        output_root / "train_known.npz",
        x=x_known[splits["train_idx"]],
        y=y_known[splits["train_idx"]],
        label_name=name_known[splits["train_idx"]],
    )
    save_npz(
        output_root / "val_known.npz",
        x=x_known[splits["val_idx"]],
        y=y_known[splits["val_idx"]],
        label_name=name_known[splits["val_idx"]],
    )
    save_npz(
        output_root / "test_known.npz",
        x=x_known[splits["test_idx"]],
        y=y_known[splits["test_idx"]],
        label_name=name_known[splits["test_idx"]],
    )
    save_npz(output_root / "test_unknown.npz", x=x_unknown, y=y_unknown, label_name=name_unknown)

    summary = {
        "dataset": "wisig_compact",
        "raw_paths": [str(Path(p).resolve()) for p in raw_paths],
        "processed_root": str(output_root.resolve()),
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "split_file": str(Path(prep_cfg["split_file"]).resolve()) if prep_cfg.get("split_file") else "",
        "num_train_known": int(len(splits["train_idx"])),
        "num_val_known": int(len(splits["val_idx"])),
        "num_test_known": int(len(splits["test_idx"])),
        "num_test_unknown": int(len(y_unknown)),
        "deduplicate_exact": deduplicate_exact,
        "samples_per_class": samples_per_class,
        "signal_shape": list(x_known[splits["train_idx"]].shape[1:]),
    }
    save_json(output_root / "dataset_summary.json", summary)
    write_class_split_csv(output_root / "class_split.csv", known_classes, unknown_classes)
    if split_payload is not None:
        save_json(output_root / "split_manifest.json", split_payload)
    return summary
