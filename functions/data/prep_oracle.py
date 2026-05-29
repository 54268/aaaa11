from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import numpy as np

from functions.data.split_utils import merge_prep_with_split, write_class_split_csv
from functions.common.io import ensure_dir, save_json, save_npz


SIGMF_DTYPES = {
    "cf32": np.dtype(np.complex64),
    "cf32_le": np.dtype("<c8"),
    "cf64": np.dtype(np.complex128),
    "cf64_le": np.dtype("<c16"),
    "ci16": np.dtype([("r", "<i2"), ("i", "<i2")]),
    "ci16_le": np.dtype([("r", "<i2"), ("i", "<i2")]),
    "ci8_le": np.dtype([("r", "i1"), ("i", "i1")]),
    "ci8": np.dtype([("r", "i1"), ("i", "i1")]),
}


def _read_sigmf_complex(path: Path, datatype: str) -> np.ndarray:
    if datatype not in SIGMF_DTYPES:
        raise ValueError(f"Unsupported SigMF datatype: {datatype}")
    raw = np.fromfile(path, dtype=SIGMF_DTYPES[datatype])
    if np.iscomplexobj(raw):
        complex_arr = raw.astype(np.complex64)
    else:
        complex_arr = raw["r"].astype(np.float32) + 1j * raw["i"].astype(np.float32)
    return complex_arr


def _get_sigmf_global(meta: dict) -> dict:
    if "global" in meta:
        return meta["global"]
    return meta.get("_metadata", {}).get("global", {})


def _get_sigmf_annotations(meta: dict) -> list[dict]:
    if "annotations" in meta:
        return meta["annotations"]
    return meta.get("_metadata", {}).get("annotations", [])


def _label_sort_key(label: str) -> tuple[int, int | str]:
    return (0, int(label)) if label.isdigit() else (1, label)


def _resolve_sigmf_datatype(path: Path, declared: str, sample_count: int | None) -> str:
    declared = declared.lower()
    if sample_count and sample_count > 0:
        bytes_per_sample = path.stat().st_size / sample_count
        inferred_map = {
            2: "ci8",
            4: "ci16_le",
            8: "cf32_le",
            16: "cf64_le",
        }
        rounded = int(round(bytes_per_sample))
        if abs(bytes_per_sample - rounded) < 1e-6 and rounded in inferred_map:
            return inferred_map[rounded]
    alias_map = {
        "cf32": "cf32_le",
        "cf64": "cf64_le",
        "ci16": "ci16_le",
        "ci8_le": "ci8",
    }
    return alias_map.get(declared, declared)


def _extract_label(path: Path, label_regex: str | None) -> str:
    stem = path.stem
    if label_regex:
        match = re.search(label_regex, stem)
        if not match:
            raise ValueError(f"Could not extract label from {stem} with regex {label_regex}")
        return match.group(1) if match.groups() else match.group(0)
    iq_match = re.search(r"IQ#(\d+)", stem)
    if iq_match:
        return iq_match.group(1)
    parts = stem.split("_")
    return parts[0]


def _window_complex_signal(signal: np.ndarray, segment_length: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, max(len(signal) - segment_length + 1, 0), stride):
        seg = signal[start : start + segment_length]
        iq = np.stack([seg.real, seg.imag], axis=0)
        windows.append(iq.astype(np.float32))
    return np.asarray(windows, dtype=np.float32)


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
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64)
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def prepare_oracle_sigmf(config: dict) -> Dict[str, object]:
    prep_cfg, split_payload = merge_prep_with_split(config["prep"])
    raw_root = Path(prep_cfg["raw_root"])
    output_root = Path(prep_cfg["processed_root"])
    ensure_dir(output_root)
    rng = np.random.default_rng(int(config["train"]["seed"]))

    meta_files = sorted(raw_root.rglob("*.sigmf-meta"))
    if not meta_files:
        raise FileNotFoundError(f"No .sigmf-meta files found under {raw_root}")

    samples_by_label: Dict[str, list[np.ndarray]] = {}
    records = []
    label_regex = prep_cfg.get("label_regex")
    segment_length = int(prep_cfg.get("segment_length", config["data"].get("signal_length", 256)))
    stride = int(prep_cfg.get("stride", segment_length))
    max_segments_per_label = int(prep_cfg.get("max_segments_per_label", 0))

    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        global_meta = _get_sigmf_global(meta)
        annotations = _get_sigmf_annotations(meta)
        data_path = meta_path.with_suffix(".sigmf-data")
        if not data_path.exists():
            continue
        sample_count = None
        if annotations:
            sample_count = int(annotations[0].get("core:sample_count", 0) or 0)
        declared_datatype = str(global_meta.get("core:datatype", "cf32"))
        datatype = _resolve_sigmf_datatype(data_path, declared_datatype, sample_count)
        label = _extract_label(meta_path, label_regex)
        signal = _read_sigmf_complex(data_path, datatype)
        windows = _window_complex_signal(signal, segment_length=segment_length, stride=stride)
        if len(windows) == 0:
            continue
        samples_by_label.setdefault(label, []).append(windows)
        records.append(
            {
                "label": label,
                "meta_path": str(meta_path),
                "data_path": str(data_path),
                "num_windows": int(len(windows)),
                "datatype_declared": declared_datatype,
                "datatype_used": datatype,
            }
        )

    if not samples_by_label:
        raise RuntimeError(f"No valid Oracle SigMF samples were loaded from {raw_root}")

    labels_sorted = sorted(samples_by_label, key=_label_sort_key)
    num_known_classes = int(prep_cfg.get("num_known_classes", max(len(labels_sorted) - 2, 1)))
    known_labels = prep_cfg.get("known_classes", labels_sorted[:num_known_classes])
    unknown_labels = prep_cfg.get("unknown_classes", labels_sorted[num_known_classes:])
    known_map = {name: idx for idx, name in enumerate(known_labels)}

    known_x = []
    known_y = []
    known_name = []
    unknown_x = []
    unknown_y = []
    unknown_name = []
    for label, chunks in samples_by_label.items():
        arr = np.concatenate(chunks, axis=0)
        if max_segments_per_label > 0 and len(arr) > max_segments_per_label:
            choice = rng.choice(len(arr), size=max_segments_per_label, replace=False)
            arr = arr[choice]
        if label in known_map:
            known_x.append(arr)
            known_y.append(np.full(len(arr), known_map[label], dtype=np.int64))
            known_name.append(np.full(len(arr), label))
        elif label in unknown_labels:
            unknown_x.append(arr)
            unknown_y.append(np.full(len(arr), -1, dtype=np.int64))
            unknown_name.append(np.full(len(arr), label))

    if not known_x:
        raise RuntimeError("No known-class Oracle samples were generated. Check known_classes/label_regex.")

    x_known = np.concatenate(known_x, axis=0)
    y_known = np.concatenate(known_y, axis=0)
    name_known = np.concatenate(known_name, axis=0)
    x_unknown = np.concatenate(unknown_x, axis=0) if unknown_x else np.zeros((0, 2, segment_length), dtype=np.float32)
    y_unknown = np.concatenate(unknown_y, axis=0) if unknown_y else np.zeros((0,), dtype=np.int64)
    name_unknown = np.concatenate(unknown_name, axis=0) if unknown_name else np.zeros((0,), dtype="<U1")

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
        "dataset": "oracle_sigmf",
        "raw_root": str(raw_root.resolve()),
        "processed_root": str(output_root.resolve()),
        "known_classes": known_labels,
        "unknown_classes": unknown_labels,
        "split_file": str(Path(prep_cfg["split_file"]).resolve()) if prep_cfg.get("split_file") else "",
        "num_train_known": int(len(splits["train_idx"])),
        "num_val_known": int(len(splits["val_idx"])),
        "num_test_known": int(len(splits["test_idx"])),
        "num_test_unknown": int(len(y_unknown)),
        "num_source_records": int(len(records)),
        "num_total_labels": int(len(labels_sorted)),
        "signal_shape": list(x_known[splits["train_idx"]].shape[1:]),
    }
    save_json(output_root / "dataset_summary.json", summary)
    save_json(output_root / "record_manifest.json", {"records": records})
    write_class_split_csv(output_root / "class_split.csv", known_labels, unknown_labels)
    if split_payload is not None:
        save_json(output_root / "split_manifest.json", split_payload)
    return summary



