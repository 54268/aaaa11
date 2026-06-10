from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainers import _openmax_probabilities


def test_openmax_probabilities_use_repo_backend_and_return_valid_probabilities() -> None:
    train_logits = np.asarray(
        [
            [5.0, 0.1],
            [4.8, 0.2],
            [5.2, 0.0],
            [0.1, 5.0],
            [0.2, 4.8],
            [0.0, 5.2],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    train_predictions = train_logits.argmax(axis=1)
    query_logits = np.asarray([[5.0, 0.0], [0.0, 5.0], [2.5, 2.5]], dtype=np.float32)

    known_probs, unknown_probs = _openmax_probabilities(
        train_logits,
        train_labels,
        train_predictions,
        query_logits,
        alpha_rank=2,
        tail_size=2,
        backend="repo_openmax",
        distance_type="eucos",
    )

    assert known_probs.shape == (3, 2)
    assert unknown_probs.shape == (3,)
    np.testing.assert_allclose(known_probs.sum(axis=1) + unknown_probs, 1.0, atol=1e-6)
    assert unknown_probs[2] > min(unknown_probs[0], unknown_probs[1])
