from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def _register_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class TorchCvnnHybridBackbone(nn.Module):
    """
    Hybrid backbone for 1D I/Q.
    We reuse torchcvnn's complex normalization/activation blocks, while keeping
    1D convolutions in real-valued PyTorch because upstream torchcvnn does not
    currently expose a 1D complex convolution module.
    """

    def __init__(self, signal_length: int, embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        repo_src = Path(__file__).resolve().parents[3] / "third_party" / "torchcvnn" / "src"
        _register_path(repo_src)
        from torchcvnn.nn.modules import BatchNorm1d, CReLU  # type: ignore

        self.stem = nn.Sequential(
            nn.Conv1d(2, hidden_dim * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.complex_bn = BatchNorm1d(hidden_dim)
        self.complex_act = CReLU()
        self.readout = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )
        self.signal_length = signal_length
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        real, imag = torch.chunk(feats, 2, dim=1)
        z = torch.complex(real, imag)
        z = self.complex_bn(z)
        z = self.complex_act(z)
        mag = torch.abs(z)
        return self.readout(mag)
