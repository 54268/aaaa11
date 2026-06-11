from __future__ import annotations

import numpy as np

from functions.methods.fusion import search_fusion_params


def test_search_fusion_params_applies_score_calibration() -> None:
    unknown_label = 2
    y_true = np.asarray([0, 0, 1, 1, unknown_label, unknown_label], dtype=np.int64)
    known_pred = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    raw_scores = np.asarray([0.10, 0.12, 0.80, 0.82, 0.50, 1.20], dtype=np.float64)

    result = search_fusion_params(
        y_true=y_true,
        known_pred=known_pred,
        q_om=raw_scores,
        q_pd=raw_scores,
        unknown_label=unknown_label,
        lambda_grid=[1.0],
        threshold_grid=[0.2, 0.6, 0.9],
        threshold_mode="global",
        score_calibration_mode="classwise_z",
    )

    assert result.score_calibration is not None
    assert result.score_calibration["mode"] == "classwise_z"
    assert result.metrics["overall_accuracy"] == 1.0
