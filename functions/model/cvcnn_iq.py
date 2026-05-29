from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.real_weight = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.imag_weight = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_channels))
            self.imag_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("real_bias", None)
            self.register_parameter("imag_bias", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.real_weight(z.real) - self.imag_weight(z.imag)
        imag = self.real_weight(z.imag) + self.imag_weight(z.real)
        if self.real_bias is not None and self.imag_bias is not None:
            real = real + self.real_bias.view(1, -1, 1)
            imag = imag + self.imag_bias.view(1, -1, 1)
        return torch.complex(real, imag)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real_bn = nn.BatchNorm1d(channels)
        self.imag_bn = nn.BatchNorm1d(channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_bn(z.real), self.imag_bn(z.imag))


class ComplexReLU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class ComplexConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ComplexConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = ComplexBatchNorm1d(out_channels)
        self.act = ComplexReLU()
        self.dropout = float(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv(z)
        z = self.norm(z)
        z = self.act(z)
        if self.dropout > 0.0:
            real = F.dropout(z.real, p=self.dropout, training=self.training)
            imag = F.dropout(z.imag, p=self.dropout, training=self.training)
            z = torch.complex(real, imag)
        return z


class CVCNNBackbone(nn.Module):
    """
    Complex-valued CNN for I/Q signals.
    Input shape: [B, 2, L], where channel 0 is I and channel 1 is Q.
    """

    def __init__(self, signal_length: int, embedding_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        widths = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.features = nn.ModuleList(
            [
                ComplexConvBlock(1, widths[0], kernel_size=7, padding=3, dropout=dropout * 0.25),
                ComplexConvBlock(widths[0], widths[1], kernel_size=5, padding=2, dropout=dropout * 0.25),
                ComplexConvBlock(widths[1], widths[2], kernel_size=3, padding=1, dropout=dropout * 0.25),
            ]
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.project = nn.Sequential(
            nn.Linear(widths[-1] * 2, widths[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(widths[-1] * 2, embedding_dim),
        )
        self.signal_length = signal_length
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, :1, :], x[:, 1:2, :])
        for block in self.features:
            z = block(z)
            z = torch.complex(self.pool(z.real), self.pool(z.imag))
        pooled_real = z.real.mean(dim=-1)
        pooled_imag = z.imag.mean(dim=-1)
        feat = torch.cat([pooled_real, pooled_imag], dim=1)
        emb = self.project(feat)
        return F.normalize(emb, dim=1)



