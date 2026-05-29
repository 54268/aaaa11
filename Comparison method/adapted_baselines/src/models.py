from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding_t = (kernel_size // 2) * dilation
        self.conv_t = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_t, dilation=dilation)
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_t(x) + self.conv_f(x)
        out = F.relu(self.bn(out))
        return out + residual


class HydraClassifier(nn.Module):
    """HyDRA TDSE-style CNN + Transformer classifier adapted to [B, 2, L] I/Q input."""

    def __init__(self, num_classes: int, signal_length: int, embedding_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ResConv1d(2, 32),
            ResConv1d(32, 32, dilation=3),
            ResConv1d(32, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, signal_length + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.project = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.normal_(self.cls_token, 0.0, 0.02)
        nn.init.normal_(self.pos_embed, 0.0, 0.02)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x).transpose(1, 2)
        cls = self.cls_token.expand(len(x), -1, -1)
        feat = torch.cat([cls, feat], dim=1)
        feat = feat + self.pos_embed[:, : feat.size(1), :]
        encoded = self.transformer(feat)[:, 0]
        return F.normalize(self.project(encoded), dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        del labels
        return self.classifier(self.embed(x))


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()
        self.real_weight = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.imag_weight = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.real_bias = nn.Parameter(torch.zeros(out_channels))
        self.imag_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.real_weight(z.real) - self.imag_weight(z.imag) + self.real_bias.view(1, -1, 1)
        imag = self.real_weight(z.imag) + self.imag_weight(z.real) + self.imag_bias.view(1, -1, 1)
        return torch.complex(real, imag)


class ComplexBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, padding)
        self.real_bn = nn.BatchNorm1d(out_channels)
        self.imag_bn = nn.BatchNorm1d(out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv(z)
        return torch.complex(F.relu(self.real_bn(z.real)), F.relu(self.imag_bn(z.imag)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 16.0, margin: float = 0.20) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        phi = cosine - self.margin
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


class HyperRSIClassifier(nn.Module):
    """Hypersphere embedding classifier adapted from HyperRSI complex CNN ideas."""

    def __init__(self, num_classes: int, embedding_dim: int = 128, hidden_dim: int = 48) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ComplexBlock(1, hidden_dim, 7, 3),
                ComplexBlock(hidden_dim, hidden_dim * 2, 5, 2),
                ComplexBlock(hidden_dim * 2, hidden_dim * 2, 3, 1),
            ]
        )
        self.pool = nn.AvgPool1d(2)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.head = ArcMarginProduct(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, :1, :], x[:, 1:2, :])
        for block in self.blocks:
            z = block(z)
            z = torch.complex(self.pool(z.real), self.pool(z.imag))
        pooled = torch.cat([z.real.mean(dim=-1), z.imag.mean(dim=-1)], dim=1)
        return F.normalize(self.project(pooled), dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x), labels)


class PatchEncoder(nn.Module):
    def __init__(self, signal_length: int, embedding_dim: int = 128, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = int(math.ceil(signal_length / patch_size))
        self.patch = nn.Conv1d(2, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.size(-1) % self.patch_size
        if remainder:
            x = F.pad(x, (0, self.patch_size - remainder))
        patches = self.patch(x).transpose(1, 2)
        encoded = self.transformer(patches)
        return F.normalize(encoded.mean(dim=1), dim=1)


class OpenRFIStyleClassifier(nn.Module):
    """Small RoInformer-like encoder used for OpenRFI-style prototype grouping."""

    def __init__(self, num_classes: int, signal_length: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = PatchEncoder(signal_length=signal_length, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        del labels
        return self.classifier(self.embed(x))
