from types import SimpleNamespace

import torch

from functions.model.closed_set import ClosedSetTrainer


def test_prototype_loss_uses_euclidean_compactness() -> None:
    trainer = ClosedSetTrainer.__new__(ClosedSetTrainer)
    trainer.config = {
        "loss": {
            "lambda_basic": 1.0,
            "lambda_angle": 0.0,
            "lambda_prototype": 1.0,
            "angle_margin": 0.15,
        }
    }
    trainer.head = SimpleNamespace(
        num_classes=2,
        prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    embeddings = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1])

    losses = trainer._scheme_regularization_loss(embeddings, logits, labels)

    assert torch.isclose(losses["prototype"], torch.tensor(0.5))
