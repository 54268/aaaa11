from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUBSET_NAMES = ["SingleDay", "ManySig", "ManyRx", "ManyTx"]


@dataclass
class LoadedSubset:
    name: str
    path: Path
    file_format: str
    payload: dict[str, Any] | None
    error: str | None = None


def find_subset_candidates(raw_root: Path) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {name: None for name in SUBSET_NAMES}
    all_paths = list(raw_root.rglob("*"))
    for subset in SUBSET_NAMES:
        lower = subset.lower()
        preferred = [
            path for path in all_paths
            if lower in path.name.lower() and path.suffix.lower() in {".pkl", ".zip"}
        ]
        if preferred:
            result[subset] = sorted(preferred, key=lambda p: (p.suffix.lower() != ".pkl", len(str(p))))[0]
            continue
        dirs = [path for path in all_paths if path.is_dir() and lower in path.name.lower()]
        if dirs:
            result[subset] = sorted(dirs, key=lambda p: len(str(p)))[0]
    return result


def _load_pickle_path(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_zip_path(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        pkl_names = [name for name in zf.namelist() if name.lower().endswith(".pkl")]
        if not pkl_names:
            raise FileNotFoundError(f"No .pkl found inside {path}")
        with zf.open(sorted(pkl_names)[0]) as f:
            return pickle.load(f)


def load_subset(name: str, path: Path | None) -> LoadedSubset:
    if path is None:
        return LoadedSubset(name=name, path=Path(""), file_format="missing", payload=None, error="未找到该 subset")
    try:
        if path.is_dir():
            pkl_files = sorted(path.rglob("*.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No .pkl found under {path}")
            return LoadedSubset(name, pkl_files[0], "unpacked_pkl", _load_pickle_path(pkl_files[0]))
        if path.suffix.lower() == ".pkl":
            return LoadedSubset(name, path, "pkl", _load_pickle_path(path))
        if path.suffix.lower() == ".zip":
            return LoadedSubset(name, path, "zip", _load_zip_path(path))
        raise ValueError(f"Unsupported file format: {path}")
    except Exception as exc:
        return LoadedSubset(name=name, path=path, file_format=path.suffix.lower() or "directory", payload=None, error=repr(exc))


def signal_block_to_iq(block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected block with 3 dims, got {arr.shape}")
    if arr.shape[1:] and arr.shape[-1] == 2:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)
    if arr.shape[1] == 2:
        return arr.astype(np.float32)
    raise ValueError(f"Cannot infer I/Q axes from shape {arr.shape}")


def tx_samples(
    payload: dict[str, Any],
    tx_ids: list[str] | None = None,
    include_rx: list[str] | None = None,
    include_days: list[str] | None = None,
    include_equalized: list[int] | None = None,
) -> dict[str, np.ndarray]:
    tx_list = list(payload["tx_list"])
    rx_list = list(payload["rx_list"])
    day_list = list(payload["capture_date_list"])
    eq_list = list(payload["equalized_list"])
    use_tx = tx_ids if tx_ids is not None else tx_list
    use_rx = set(include_rx) if include_rx is not None else set(rx_list)
    use_days = set(include_days) if include_days is not None else set(day_list)
    use_eq = set(include_equalized) if include_equalized is not None else set(eq_list)

    out: dict[str, np.ndarray] = {}
    for tx in use_tx:
        if tx not in tx_list:
            continue
        tx_idx = tx_list.index(tx)
        chunks = []
        for rx_idx, rx in enumerate(rx_list):
            if rx not in use_rx:
                continue
            for day_idx, day in enumerate(day_list):
                if day not in use_days:
                    continue
                for eq_idx, eq in enumerate(eq_list):
                    if eq not in use_eq:
                        continue
                    block = payload["data"][tx_idx][rx_idx][day_idx][eq_idx]
                    arr = signal_block_to_iq(block)
                    if len(arr):
                        chunks.append(arr)
        if chunks:
            out[tx] = np.concatenate(chunks, axis=0)
    return out


def subset_inventory_row(loaded: LoadedSubset) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subset": loaded.name,
        "path": str(loaded.path),
        "file_format": loaded.file_format,
        "loaded": loaded.payload is not None,
        "error": loaded.error or "",
    }
    payload = loaded.payload
    if payload is None:
        return row
    tx_list = list(payload.get("tx_list", []))
    rx_list = list(payload.get("rx_list", []))
    day_list = list(payload.get("capture_date_list", []))
    eq_list = list(payload.get("equalized_list", []))
    counts = []
    sample_shape = ""
    for tx_idx in range(len(tx_list)):
        total = 0
        for rx_idx in range(len(rx_list)):
            for day_idx in range(len(day_list)):
                for eq_idx in range(len(eq_list)):
                    block = np.asarray(payload["data"][tx_idx][rx_idx][day_idx][eq_idx])
                    total += int(len(block))
                    if not sample_shape and len(block):
                        sample_shape = str(signal_block_to_iq(block).shape[1:])
        counts.append(total)
    counts_arr = np.asarray(counts, dtype=np.float64) if counts else np.asarray([], dtype=np.float64)
    row.update(
        {
            "object_type": type(payload).__name__,
            "keys": ",".join(payload.keys()),
            "num_tx": len(tx_list),
            "num_rx": len(rx_list),
            "num_days": len(day_list),
            "equalized_values": ",".join(str(x) for x in eq_list),
            "signal_shape": sample_shape,
            "tx_sample_min": float(counts_arr.min()) if len(counts_arr) else 0.0,
            "tx_sample_max": float(counts_arr.max()) if len(counts_arr) else 0.0,
            "tx_sample_mean": float(counts_arr.mean()) if len(counts_arr) else 0.0,
            "tx_sample_median": float(np.median(counts_arr)) if len(counts_arr) else 0.0,
            "can_run_p1": len(tx_list) >= 6,
            "tx_list": tx_list,
            "rx_list": rx_list,
            "day_list": day_list,
        }
    )
    return row


def balanced_sample(
    samples_by_tx: dict[str, np.ndarray],
    tx_ids: list[str],
    max_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    available = {tx: len(samples_by_tx.get(tx, [])) for tx in tx_ids}
    usable = [tx for tx in tx_ids if available.get(tx, 0) > 0]
    if len(usable) != len(tx_ids):
        missing = [tx for tx in tx_ids if tx not in usable]
        raise ValueError(f"Missing samples for tx: {missing}")
    n_per_class = min(min(available.values()), int(max_per_class))
    if n_per_class <= 0:
        raise ValueError("No samples available for balanced sampling")
    x_parts = []
    y_parts = []
    index_summary = {}
    for label_id, tx in enumerate(tx_ids):
        choice = rng.choice(len(samples_by_tx[tx]), size=n_per_class, replace=False)
        x_parts.append(samples_by_tx[tx][choice])
        y_parts.append(np.full(n_per_class, tx, dtype=object))
        index_summary[tx] = {
            "count": int(n_per_class),
            "first_indices": [int(x) for x in choice[:20].tolist()],
        }
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order], {"samples_per_class": int(n_per_class), "selected_tx_ids": tx_ids, "index_summary": index_summary}

