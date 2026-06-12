from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainers import (  # noqa: E402
    _arpl_open_set_prediction,
    _center_open_set_prediction,
    _openmax_open_set_prediction,
    _softmax_open_set_prediction,
)


def test_softmax_threshold_uses_only_known_validation_logits() -> None:
    val_logits = np.asarray([[5.0, 0.0], [1.4, 0.0], [0.0, 5.0]], dtype=np.float32)
    test_logits = np.asarray([[8.0, 0.0], [0.1, 0.0], [0.0, 8.0]], dtype=np.float32)

    pred, unknown_score, threshold = _softmax_open_set_prediction(
        val_logits,
        test_logits,
        known_quantile=0.5,
        unknown_label=2,
    )

    val_conf = np.exp(val_logits) / np.exp(val_logits).sum(axis=1, keepdims=True)
    expected_threshold = float(np.quantile(val_conf.max(axis=1), 0.5))
    assert threshold == expected_threshold
    assert pred.tolist() == [0, 2, 1]
    np.testing.assert_allclose(unknown_score, [1.0 - 0.99966465, 0.47502083, 1.0 - 0.99966465], atol=1e-6)


def test_center_threshold_uses_only_known_training_and_validation_features() -> None:
    train_features = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    train_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    val_features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    test_features = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    pred, unknown_score, threshold = _center_open_set_prediction(
        train_features,
        train_labels,
        val_features,
        test_features,
        known_quantile=0.5,
        unknown_label=2,
    )

    centers = np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    expected_threshold = float(np.quantile((val_features @ centers.T).max(axis=1), 0.5))
    assert threshold == expected_threshold
    assert pred.tolist() == [0, 1, 2]
    assert unknown_score.shape == (3,)


def test_arpl_threshold_uses_only_known_validation_logits() -> None:
    val_logits = np.asarray([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
    test_logits = np.asarray([[3.0, 0.0], [0.0, 2.0], [0.2, 0.0]], dtype=np.float32)

    pred, unknown_score, threshold = _arpl_open_set_prediction(
        val_logits,
        test_logits,
        known_quantile=0.5,
        unknown_label=2,
    )

    assert threshold == float(np.quantile(np.max(val_logits, axis=1), 0.5))
    assert pred.tolist() == [0, 1, 2]
    np.testing.assert_allclose(unknown_score, [-3.0, -2.0, -0.2], atol=1e-6)


def test_openmax_threshold_uses_known_training_and_validation_logits_only() -> None:
    train_logits = np.asarray(
        [
            [5.0, 0.0],
            [4.8, 0.1],
            [5.2, 0.0],
            [0.0, 5.0],
            [0.2, 4.8],
            [0.0, 5.2],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    val_logits = np.asarray([[5.0, 0.0], [0.0, 5.0], [4.9, 0.1]], dtype=np.float32)
    test_logits = np.asarray([[5.1, 0.0], [0.0, 5.1], [2.5, 2.5]], dtype=np.float32)

    pred, unknown_score, threshold = _openmax_open_set_prediction(
        train_logits,
        train_labels,
        val_logits,
        test_logits,
        known_quantile=0.5,
        unknown_label=2,
        backend="repo_openmax",
        distance_type="eucos",
    )

    assert 0.0 <= threshold <= 1.0
    assert pred.shape == (3,)
    assert unknown_score.shape == (3,)
