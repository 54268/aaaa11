from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import ARPLClassifier, ARPLConfusingSampleDiscriminator1D, ARPLConfusingSampleGenerator1D


def test_arpl_confusing_sample_components_have_expected_shapes():
    batch_size = 4
    signal_length = 256
    noise_dim = 32
    model = ARPLClassifier(num_classes=3, embedding_dim=16)
    generator = ARPLConfusingSampleGenerator1D(noise_dim=noise_dim, signal_length=signal_length)
    discriminator = ARPLConfusingSampleDiscriminator1D(signal_length=signal_length)

    noise = torch.randn(batch_size, noise_dim)
    fake = generator(noise)
    assert fake.shape == (batch_size, 2, signal_length)

    disc_logits = discriminator(fake)
    assert disc_logits.shape == (batch_size,)

    loss = model.fake_loss(fake)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
