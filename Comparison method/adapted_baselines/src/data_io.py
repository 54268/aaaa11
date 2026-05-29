from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetProtocol:
    name: str
    processed_root: Path
    normalize: str = "per_sample"


@dataclass
class LoadedProtocol:
    name: str
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_known_x: np.ndarray
    test_known_y: np.ndarray
    test_unknown_x: np.ndarray
    test_unknown_y: np.ndarray
    test_unknown_names: np.ndarray
    num_known_classes: int
    num_unknown_classes: int
    signal_length: int
    metadata: Dict[str, object]


class IQDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, normalize: str = "per_sample") -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.normalize = normalize

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, index: int):
        sample = self.x[index]
        if self.normalize == "per_sample":
            mean = sample.mean(axis=1, keepdims=True)
            std = sample.std(axis=1, keepdims=True)
            sample = (sample - mean) / (std + 1e-6)
        return torch.from_numpy(sample.copy()), torch.tensor(self.y[index], dtype=torch.long)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(protocol: DatasetProtocol) -> LoadedProtocol:
    root = protocol.processed_root
    required = {
        "train_known": root / "train_known.npz",
        "val_known": root / "val_known.npz",
        "test_known": root / "test_known.npz",
        "test_unknown": root / "test_unknown.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少对比方法所需 npz 数据: " + "; ".join(missing))

    train = _load_npz(required["train_known"])
    val = _load_npz(required["val_known"])
    test_known = _load_npz(required["test_known"])
    test_unknown = _load_npz(required["test_unknown"])

    unknown_y = test_unknown.get("y")
    if unknown_y is None:
        unknown_y = np.full(len(test_unknown["x"]), -1, dtype=np.int64)
    unknown_names = test_unknown.get("label_name")
    if unknown_names is None:
        unknown_names = unknown_y.astype(str)

    num_known = int(np.max(train["y"])) + 1
    unique_unknown = np.unique(unknown_names.astype(str))
    metadata = {
        "dataset_summary": _load_json_if_exists(root / "dataset_summary.json"),
        "split_manifest": _load_json_if_exists(root / "split_manifest.json"),
    }
    return LoadedProtocol(
        name=protocol.name,
        train_x=train["x"],
        train_y=train["y"].astype(np.int64),
        val_x=val["x"],
        val_y=val["y"].astype(np.int64),
        test_known_x=test_known["x"],
        test_known_y=test_known["y"].astype(np.int64),
        test_unknown_x=test_unknown["x"],
        test_unknown_y=unknown_y.astype(np.int64),
        test_unknown_names=unknown_names.astype(str),
        num_known_classes=num_known,
        num_unknown_classes=int(len(unique_unknown)),
        signal_length=int(train["x"].shape[-1]),
        metadata=metadata,
    )


def sample_balanced(
    x: np.ndarray,
    y: np.ndarray,
    max_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_per_class is None:
        return x, y
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for label in sorted(np.unique(y).tolist()):
        label_idx = np.where(y == label)[0]
        take = min(int(max_per_class), len(label_idx))
        indices.append(rng.choice(label_idx, size=take, replace=False))
    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return x[selected], y[selected]


def make_iq_dataset(
    x: np.ndarray,
    y: np.ndarray,
    normalize: str,
    max_per_class: int | None = None,
    seed: int = 0,
) -> IQDataset:
    sx, sy = sample_balanced(x, y, max_per_class=max_per_class, seed=seed)
    return IQDataset(sx, sy, normalize=normalize)


def known_unknown_eval_arrays(protocol: LoadedProtocol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unknown_label = protocol.num_known_classes
    x = np.concatenate([protocol.test_known_x, protocol.test_unknown_x], axis=0)
    y_for_rejection = np.concatenate(
        [
            protocol.test_known_y,
            np.full(len(protocol.test_unknown_y), unknown_label, dtype=np.int64),
        ],
        axis=0,
    )
    original_unknown_y = protocol.test_unknown_y
    return x, y_for_rejection, original_unknown_y


def protocol_summary(protocols: Iterable[LoadedProtocol]) -> list[dict]:
    rows = []
    for item in protocols:
        rows.append(
            {
                "dataset": item.name,
                "train_known": int(len(item.train_y)),
                "val_known": int(len(item.val_y)),
                "test_known": int(len(item.test_known_y)),
                "test_unknown": int(len(item.test_unknown_y)),
                "known_classes": int(item.num_known_classes),
                "unknown_classes": int(item.num_unknown_classes),
                "signal_length": int(item.signal_length),
            }
        )
    return rows
