from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, normalize: str = "none") -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        sample = self.x[index]
        if self.normalize == "per_sample":
            mean = sample.mean(axis=1, keepdims=True)
            std = sample.std(axis=1, keepdims=True)
            sample = (sample - mean) / (std + 1e-6)
        return torch.from_numpy(sample.copy()), torch.tensor(self.y[index], dtype=torch.long)


@dataclass
class NPZSplits:
    train_known: SignalDataset
    val_known: SignalDataset
    test_known: SignalDataset
    test_unknown: SignalDataset
    num_known_classes: int
    signal_length: int


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    payload = np.load(Path(path))
    return {k: payload[k] for k in payload.files}


def load_separate_npz_dataset(root: str | Path, normalize: str = "none") -> NPZSplits:
    root = Path(root)
    required = {
        "train_known": root / "train_known.npz",
        "val_known": root / "val_known.npz",
        "test_known": root / "test_known.npz",
        "test_unknown": root / "test_unknown.npz",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing NPZ files: {missing}. Expected under {root}")

    train = load_npz(required["train_known"])
    val = load_npz(required["val_known"])
    test_known = load_npz(required["test_known"])
    test_unknown = load_npz(required["test_unknown"])

    num_known_classes = int(np.max(train["y"])) + 1
    signal_length = int(train["x"].shape[-1])

    unknown_y = test_unknown.get("y")
    if unknown_y is None:
        unknown_y = np.full(len(test_unknown["x"]), -1, dtype=np.int64)

    return NPZSplits(
        train_known=SignalDataset(train["x"], train["y"], normalize=normalize),
        val_known=SignalDataset(val["x"], val["y"], normalize=normalize),
        test_known=SignalDataset(test_known["x"], test_known["y"], normalize=normalize),
        test_unknown=SignalDataset(test_unknown["x"], unknown_y, normalize=normalize),
        num_known_classes=num_known_classes,
        signal_length=signal_length,
    )



