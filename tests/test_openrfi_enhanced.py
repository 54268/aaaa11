from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Comparison method" / "adapted_baselines" / "src"))

from models import OpenRFIStyleClassifier, openrfi_frame_rearrangement, openrfi_noise_jitter


def test_openrfi_views_preserve_shape_and_can_be_deterministic() -> None:
    x = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16)

    jittered = openrfi_noise_jitter(x, noise_std=0.0, amplitude_jitter=0.0)
    rearranged = openrfi_frame_rearrangement(x, segments=4, perm=[1, 0, 3, 2])

    assert jittered.shape == x.shape
    assert rearranged.shape == x.shape
    assert torch.equal(jittered, x)
    assert not torch.equal(rearranged, x)


def test_openrfi_model_exposes_training_loss_and_embeddings() -> None:
    model = OpenRFIStyleClassifier(num_classes=3, signal_length=16, embedding_dim=32)
    x = torch.randn(4, 2, 16)
    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    emb = model.embed(x)
    logits = model(x, y)
    loss = model.training_loss(x, y)

    assert emb.shape == (4, 32)
    assert logits.shape == (4, 3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
