from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import HyperRSIClassifier
from trainers import _hyperrsi_gpd_open_set_prediction


def test_hyperrsi_has_paper_scale_embedding_and_cosface_head() -> None:
    model = HyperRSIClassifier(num_classes=3)
    x = torch.randn(2, 2, 256)
    labels = torch.tensor([0, 2])

    embedding = model.embed(x)
    logits = model(x, labels)

    assert len(model.blocks) == 9
    assert embedding.shape == (2, 512)
    assert logits.shape == (2, 3)
    assert model.head.scale == 8.0
    assert model.head.margin == 0.2


def test_hyperrsi_gpd_rejects_low_similarity_queries() -> None:
    train_features = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.02],
            [0.98, -0.02],
            [0.0, 1.0],
            [0.02, 0.99],
            [-0.02, 0.98],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    query_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.7, -0.7],
        ],
        dtype=np.float32,
    )

    pred, unknown_score, threshold = _hyperrsi_gpd_open_set_prediction(
        train_features,
        train_labels,
        query_features,
        unknown_label=2,
        tail_quantile=0.67,
        probability_threshold=0.5,
    )

    assert pred.tolist()[:2] == [0, 1]
    assert pred[2] == 2
    assert unknown_score[2] > max(unknown_score[0], unknown_score[1])
    assert threshold == 0.5
