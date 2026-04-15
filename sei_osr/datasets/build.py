from __future__ import annotations

from torch.utils.data import DataLoader

from .base import DataBundle
from .npz_dataset import load_separate_npz_dataset
from .synthetic import build_synthetic_dataset


class OpenSetDataModule:
    def __init__(self, bundle: DataBundle, batch_size: int, num_workers: int = 0) -> None:
        self.bundle = bundle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.train_known,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_known_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.val_known,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_known_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.test_known,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_unknown_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bundle.test_unknown,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def build_data_module(config: dict) -> OpenSetDataModule:
    mode = config["data"]["mode"]
    if mode == "synthetic":
        splits = build_synthetic_dataset(config)
    elif mode == "separate_npz":
        splits = load_separate_npz_dataset(
            config["data"]["root"],
            normalize=str(config["data"].get("normalize", "none")),
        )
    else:
        raise ValueError(f"Unsupported data.mode: {mode}")

    bundle = DataBundle(
        train_known=splits.train_known,
        val_known=splits.val_known,
        test_known=splits.test_known,
        test_unknown=splits.test_unknown,
        num_known_classes=splits.num_known_classes,
        signal_length=splits.signal_length,
        class_names=[f"class_{idx}" for idx in range(splits.num_known_classes)],
    )
    return OpenSetDataModule(
        bundle=bundle,
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
    )
