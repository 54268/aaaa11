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


class SoftmaxCNNClassifier(nn.Module):
    """Mature softmax baseline with a compact I/Q CNN backbone."""

    def __init__(self, num_classes: int, embedding_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(self.encoder(x)), dim=1)

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
        return torch.complex(self.real_bn(F.relu(z.real)), self.imag_bn(F.relu(z.imag)))


class CompactComplexBlock(nn.Module):
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


class CosFaceProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 8.0, margin: float = 0.20) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))
        if labels is None:
            return cosine * self.scale
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = cosine - one_hot * self.margin
        return logits * self.scale


def openrfi_noise_jitter(
    x: torch.Tensor,
    *,
    noise_std: float = 0.05,
    amplitude_jitter: float = 0.05,
) -> torch.Tensor:
    if noise_std <= 0.0 and amplitude_jitter <= 0.0:
        return x.clone()
    jitter = 1.0
    if amplitude_jitter > 0.0:
        jitter = torch.empty(x.size(0), 1, 1, device=x.device, dtype=x.dtype).uniform_(
            1.0 - amplitude_jitter,
            1.0 + amplitude_jitter,
        )
    noise = torch.zeros_like(x)
    if noise_std > 0.0:
        noise = torch.randn_like(x) * float(noise_std)
    return x * jitter + noise


def openrfi_frame_rearrangement(
    x: torch.Tensor,
    *,
    segments: int = 4,
    perm: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("openrfi_frame_rearrangement expects a [B, C, L] tensor")
    segments = max(int(segments), 1)
    if segments == 1 or x.size(-1) == 0:
        return x.clone()

    batch, channels, length = x.shape
    padded_length = max(int(math.ceil(length / segments) * segments), segments)
    if padded_length != length:
        x = F.pad(x, (0, padded_length - length))
    segment_length = padded_length // segments
    frames = x.view(batch, channels, segments, segment_length)

    if perm is None:
        perm_tensor = torch.randperm(segments, device=x.device)
    else:
        perm_tensor = torch.as_tensor(perm, device=x.device, dtype=torch.long)
        if perm_tensor.numel() != segments:
            raise ValueError("perm must contain exactly `segments` indices")
    rearranged = frames.index_select(2, perm_tensor)
    return rearranged.reshape(batch, channels, padded_length)[..., :length].contiguous()


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError("NT-Xent requires paired embeddings with the same shape")
    if z1.size(0) < 2:
        return F.mse_loss(z1, z2)

    temperature = max(float(temperature), 1e-6)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(reps, reps.t()) / temperature
    eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    targets = torch.arange(z1.size(0), device=z1.device)
    targets = torch.cat([targets + z1.size(0), targets], dim=0)
    return F.cross_entropy(logits, targets)


class HyperRSIClassifier(nn.Module):
    """HyperRSI-style complex CNN with hypersphere embedding and CosFace head."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        complex_channels: int = 64,
        output_length: int = 9,
        num_blocks: int = 9,
    ) -> None:
        super().__init__()
        self.output_length = int(output_length)
        self.blocks = nn.ModuleList(
            [
                ComplexBlock(
                    1 if idx == 0 else complex_channels,
                    complex_channels,
                    kernel_size=3,
                    padding=1,
                )
                for idx in range(num_blocks)
            ]
        )
        self.pool = nn.MaxPool1d(2)
        self.final_pool = nn.AdaptiveAvgPool1d(self.output_length)
        self.project = nn.Linear(complex_channels * 2 * self.output_length, embedding_dim)
        self.head = CosFaceProduct(embedding_dim, num_classes, scale=8.0, margin=0.20)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, :1, :], x[:, 1:2, :])
        for block in self.blocks:
            z = block(z)
            if z.size(-1) // 2 >= self.output_length:
                z = torch.complex(self.pool(z.real), self.pool(z.imag))
        pooled = torch.cat([self.final_pool(z.real), self.final_pool(z.imag)], dim=1)
        return F.normalize(self.project(pooled.flatten(1)), dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x), labels)


class CompactHyperRSIClassifier(nn.Module):
    """Compact HyperRSI variant used for the current 256-sample I/Q protocol."""

    def __init__(self, num_classes: int, embedding_dim: int = 128, hidden_dim: int = 48) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CompactComplexBlock(1, hidden_dim, 7, 3),
                CompactComplexBlock(hidden_dim, hidden_dim * 2, 5, 2),
                CompactComplexBlock(hidden_dim * 2, hidden_dim * 2, 3, 1),
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


class ARPLClassifier(nn.Module):
    """ARPL-style reciprocal-point classifier for 1-D I/Q inputs.

    保留 reciprocal points 与 margin ranking 约束，不引入图像版 GAN confusing-sample 流程。
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        temp: float = 1.0,
        weight_pl: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.points = nn.Parameter(0.1 * torch.randn(num_classes, embedding_dim))
        self.radius = nn.Parameter(torch.zeros(1))
        self.temp = float(temp)
        self.weight_pl = float(weight_pl)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        feat_sq = torch.sum(features.pow(2), dim=1, keepdim=True)
        point_sq = torch.sum(self.points.pow(2), dim=1, keepdim=True).transpose(0, 1)
        dist_l2 = feat_sq - 2.0 * features @ self.points.t() + point_sq
        dist_l2 = dist_l2 / float(features.shape[1])
        dist_dot = features @ self.points.t()
        return dist_l2 - dist_dot

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        del labels
        return self.logits_from_features(self.embed(x))

    def training_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.embed(x)
        logits = self.logits_from_features(features)
        loss_cls = F.cross_entropy(logits / self.temp, labels)
        center_batch = self.points[labels]
        dist_known = (features - center_batch).pow(2).mean(dim=1)
        target = torch.ones_like(dist_known)
        loss_r = self.margin_loss(self.radius.expand_as(dist_known), dist_known, target)
        return loss_cls + self.weight_pl * loss_r

    def fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits_from_features(self.embed(x))
        prob = F.softmax(logits, dim=1)
        entropy = torch.sum(prob * torch.log(prob.clamp_min(1e-12)), dim=1)
        return torch.exp(entropy.mean())


class ARPLConfusingSampleGenerator1D(nn.Module):
    """Light 1-D confusing-sample generator inspired by ARPL+CS."""

    def __init__(self, noise_dim: int, signal_length: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if signal_length % 16 != 0:
            raise ValueError("signal_length must be divisible by 16 for the ARPL generator")
        self.signal_length = signal_length
        self.start_length = signal_length // 16
        self.fc = nn.Linear(noise_dim, hidden_dim * self.start_length)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden_dim // 8, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        feat = self.fc(noise).view(noise.size(0), -1, self.start_length)
        return self.net(feat)


class ARPLConfusingSampleDiscriminator1D(nn.Module):
    """Light 1-D discriminator for ARPL confusing samples."""

    def __init__(self, signal_length: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
    """OpenRFI-style RF encoder with paper-inspired augmentations and prototype regularizers."""

    def __init__(
        self,
        num_classes: int,
        signal_length: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_prototypes: int | None = None,
        arc_scale: float = 16.0,
        arc_margin: float = 0.20,
        prototype_temperature: float = 0.10,
        augmentation_segments: int = 4,
        noise_std: float = 0.05,
        amplitude_jitter: float = 0.05,
        instance_weight: float = 0.01,
        prototype_weight: float = 0.01,
        group_weight: float = 0.01,
        entropy_weight: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.signal_length = int(signal_length)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.prototype_temperature = float(prototype_temperature)
        self.augmentation_segments = max(int(augmentation_segments), 1)
        self.noise_std = float(noise_std)
        self.amplitude_jitter = float(amplitude_jitter)
        self.instance_weight = float(instance_weight)
        self.prototype_weight = float(prototype_weight)
        self.group_weight = float(group_weight)
        self.entropy_weight = float(entropy_weight)

        self.backbone = nn.Sequential(
            ResConv1d(2, 32),
            ResConv1d(32, 32, dilation=3),
            ResConv1d(32, self.hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.signal_length + 1, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.project = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.classifier = ArcMarginProduct(self.embedding_dim, self.num_classes, scale=arc_scale, margin=arc_margin)

        self.num_prototypes = max(int(num_prototypes or max(self.num_classes * 4, self.num_classes)), self.num_classes)
        proto = F.normalize(torch.randn(self.num_prototypes, self.embedding_dim), dim=1)
        self.prototypes = nn.Parameter(proto)
        group_mask = self._build_round_robin_group_mask(self.num_classes, self.num_prototypes)
        self.register_buffer("group_mask", group_mask)
        self.register_buffer("proto_graph", torch.eye(self.num_prototypes))
        self.register_buffer("proto_ind", torch.ones(self.num_prototypes, dtype=torch.bool))

        nn.init.normal_(self.cls_token, 0.0, 0.02)
        nn.init.normal_(self.pos_embed, 0.0, 0.02)

    @staticmethod
    def _build_round_robin_group_mask(num_groups: int, num_prototypes: int) -> torch.Tensor:
        group_mask = torch.zeros(num_groups, num_prototypes, dtype=torch.float32)
        for proto_idx in range(num_prototypes):
            group_mask[proto_idx % num_groups, proto_idx] = 1.0
        return group_mask

    def set_group_mask(self, group_mask: torch.Tensor) -> None:
        if group_mask.ndim != 2 or group_mask.shape[1] != self.num_prototypes:
            raise ValueError("group_mask must have shape [num_groups, num_prototypes]")
        if group_mask.shape[0] != self.num_classes:
            raise ValueError("group_mask must keep the known-class group count aligned with the classifier")
        self.group_mask.copy_(group_mask.to(device=self.group_mask.device, dtype=self.group_mask.dtype))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).transpose(1, 2)
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        encoded = self.transformer(tokens)[:, 0]
        return F.normalize(self.project(encoded), dim=1)

    def prototype_logits(self, features: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=1)
        return torch.matmul(features, prototypes.t()) / self.prototype_temperature

    def group_logits(self, features: torch.Tensor) -> torch.Tensor:
        proto_probs = F.softmax(self.prototype_logits(features), dim=1)
        return torch.matmul(proto_probs, self.group_mask.t())

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.embed(x), labels)

    def paper_loss_components(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.embed(x)
        logits = self.classifier(features, labels)
        cls_loss = F.cross_entropy(logits, labels)

        augmented = openrfi_frame_rearrangement(
            openrfi_noise_jitter(x, noise_std=self.noise_std, amplitude_jitter=self.amplitude_jitter),
            segments=self.augmentation_segments,
        )
        instance_loss = _nt_xent_loss(features, self.embed(augmented))

        proto_logits = self.prototype_logits(features)
        proto_target = F.normalize(self.group_mask[labels], p=1, dim=1)
        prototype_loss = F.kl_div(
            F.log_softmax(proto_logits, dim=1),
            proto_target,
            reduction="batchmean",
        )

        group_probs = self.group_logits(features)
        group_loss = F.nll_loss(torch.log(group_probs.clamp_min(1e-12)), labels)

        proto_prior = self.group_mask.sum(dim=0)
        proto_prior = proto_prior / proto_prior.sum().clamp_min(1.0)
        proto_mean = F.softmax(proto_logits, dim=1).mean(dim=0)
        entropy_loss = torch.sum(
            proto_mean * (torch.log(proto_mean.clamp_min(1e-12)) - torch.log(proto_prior.clamp_min(1e-12)))
        )

        return {
            "cls": cls_loss,
            "instance": instance_loss,
            "prototype": prototype_loss,
            "group": group_loss,
            "entropy": entropy_loss,
        }

    def training_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = self.paper_loss_components(x, labels)
        return (
            losses["cls"]
            + self.instance_weight * losses["instance"]
            + self.prototype_weight * losses["prototype"]
            + self.group_weight * losses["group"]
            + self.entropy_weight * losses["entropy"]
        )
