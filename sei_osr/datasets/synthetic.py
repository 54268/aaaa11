from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticSignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.x[index]), torch.tensor(self.y[index], dtype=torch.long)


@dataclass
class SyntheticSplits:
    train_known: SyntheticSignalDataset
    val_known: SyntheticSignalDataset
    test_known: SyntheticSignalDataset
    test_unknown: SyntheticSignalDataset
    num_known_classes: int
    signal_length: int


def _make_samples(
    class_ids: list[int],
    samples_per_class: int,
    signal_length: int,
    noise_std: float,
    rng: np.random.Generator,
    unknown_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, signal_length, endpoint=False)
    all_x = []
    all_y = []
    for cls in class_ids:
        base_freq = 2.0 + 0.35 * cls + unknown_bias
        harmonic = 1.0 + 0.12 * cls
        amp = 1.0 + 0.04 * cls
        phase = 0.4 * cls + unknown_bias
        for _ in range(samples_per_class):
            phase_jitter = phase + rng.normal(0.0, 0.2)
            freq_jitter = base_freq + rng.normal(0.0, 0.08)
            i_sig = amp * np.cos(2 * np.pi * freq_jitter * t + phase_jitter)
            q_sig = amp * np.sin(2 * np.pi * freq_jitter * t + phase_jitter)
            i_sig += 0.25 * np.cos(2 * np.pi * harmonic * t + 0.5 * phase_jitter)
            q_sig += 0.25 * np.sin(2 * np.pi * harmonic * t + 0.5 * phase_jitter)
            signal = np.stack([i_sig, q_sig], axis=0)
            signal += rng.normal(0.0, noise_std, size=signal.shape)
            energy = np.sqrt(np.mean(np.sum(signal**2, axis=0), axis=0) + 1e-8)
            signal = signal / energy
            all_x.append(signal)
            all_y.append(cls)
    return np.asarray(all_x, dtype=np.float32), np.asarray(all_y, dtype=np.int64)


def build_synthetic_dataset(config: dict) -> SyntheticSplits:
    data_cfg = config["data"]
    rng = np.random.default_rng(int(config["train"]["seed"]))
    num_known = int(data_cfg["num_known_classes"])
    num_unknown = int(data_cfg["num_unknown_classes"])
    signal_length = int(data_cfg["signal_length"])
    noise_std = float(data_cfg.get("synthetic_noise_std", 0.08))

    known_ids = list(range(num_known))
    unknown_ids = list(range(num_known, num_known + num_unknown))

    train_x, train_y = _make_samples(
        known_ids,
        int(data_cfg["train_per_class"]),
        signal_length,
        noise_std,
        rng,
    )
    val_x, val_y = _make_samples(
        known_ids,
        int(data_cfg["val_per_class"]),
        signal_length,
        noise_std,
        rng,
    )
    test_known_x, test_known_y = _make_samples(
        known_ids,
        int(data_cfg["test_known_per_class"]),
        signal_length,
        noise_std,
        rng,
    )
    test_unknown_x, _ = _make_samples(
        unknown_ids,
        int(data_cfg["test_unknown_per_class"]),
        signal_length,
        noise_std * 1.15,
        rng,
        unknown_bias=0.6,
    )
    test_unknown_y = np.full(len(test_unknown_x), -1, dtype=np.int64)

    return SyntheticSplits(
        train_known=SyntheticSignalDataset(train_x, train_y),
        val_known=SyntheticSignalDataset(val_x, val_y),
        test_known=SyntheticSignalDataset(test_known_x, test_known_y),
        test_unknown=SyntheticSignalDataset(test_unknown_x, test_unknown_y),
        num_known_classes=num_known,
        signal_length=signal_length,
    )
