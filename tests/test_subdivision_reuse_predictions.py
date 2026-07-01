from __future__ import annotations

import numpy as np
import pytest

from functions.subdivision_pipeline import _load_open_set_predictions, _merge_labels_to_target_k


def test_load_open_set_predictions_reads_rejection_labels_and_scores(tmp_path) -> None:
    path = tmp_path / "open_set_predictions.csv"
    path.write_text(
        "y_true,y_pred,unknown_score,q_om,q_pd,d_min\n"
        "0,0,0.10,0.01,0.20,0.30\n"
        "2,3,0.90,0.80,0.70,0.60\n",
        encoding="utf-8",
    )

    loaded = _load_open_set_predictions(path, expected_size=2)

    np.testing.assert_array_equal(loaded["y_true"], [0, 2])
    np.testing.assert_array_equal(loaded["y_pred"], [0, 3])
    np.testing.assert_allclose(loaded["unknown_score"], [0.10, 0.90])
    np.testing.assert_allclose(loaded["q_om"], [0.01, 0.80])
    np.testing.assert_allclose(loaded["q_pd"], [0.20, 0.70])


def test_load_open_set_predictions_rejects_wrong_length(tmp_path) -> None:
    path = tmp_path / "open_set_predictions.csv"
    path.write_text(
        "y_true,y_pred,unknown_score,q_om,q_pd,d_min\n"
        "0,0,0.10,0.01,0.20,0.30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected 2 prediction rows"):
        _load_open_set_predictions(path, expected_size=2)


def test_merge_labels_to_target_k_folds_smallest_cluster_into_nearest_neighbor() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.2, 0.0],
            [9.0, 0.0],
            [9.1, 0.0],
            [5.05, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int64)

    merged_labels, centers, diagnostics = _merge_labels_to_target_k(features, labels, target_k=3)

    assert diagnostics["merged_cluster_count"] == 1
    assert len(np.unique(merged_labels[merged_labels != -1])) == 3
    np.testing.assert_array_equal(merged_labels, [0, 0, 1, 1, 1, 2, 2, 1])
    assert centers.shape == (3, 2)
