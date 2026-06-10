from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import OpenRFIStyleClassifier, openrfi_frame_rearrangement, openrfi_noise_jitter


def test_openrfi_augmentations_preserve_iq_shape():
    x = torch.arange(4 * 2 * 16, dtype=torch.float32).view(4, 2, 16)

    jittered = openrfi_noise_jitter(x, noise_std=0.0, amplitude_jitter=0.0)
    assert jittered.shape == x.shape
    assert torch.equal(jittered, x)

    rearranged = openrfi_frame_rearrangement(x, segments=4, perm=torch.tensor([1, 0, 3, 2]))
    assert rearranged.shape == x.shape
    assert not torch.equal(rearranged, x)
    assert torch.equal(rearranged[..., :4], x[..., 4:8])


def test_openrfi_style_classifier_has_enhanced_training_path():
    model = OpenRFIStyleClassifier(num_classes=3, signal_length=16, embedding_dim=32, num_prototypes=12)
    x = torch.randn(4, 2, 16)
    labels = torch.tensor([0, 1, 2, 1])

    emb = model.embed(x)
    logits = model(x, labels)
    loss = model.training_loss(x, labels)

    assert emb.shape == (4, 32)
    assert logits.shape == (4, 3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
