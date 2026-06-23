from __future__ import annotations

import numpy as np

from run_oracle_supervised_calibrator import (
    rescale_pseudo_embeddings,
    select_cross_fold_candidate,
)


def test_rescale_pseudo_embeddings_extends_from_source() -> None:
    sources = np.array([[1.0, 2.0], [0.0, 0.0]])
    pseudo = np.array([[3.0, 4.0], [1.0, -1.0]])

    scaled = rescale_pseudo_embeddings(sources, pseudo, scale=0.5)

    assert np.allclose(scaled, [[2.0, 3.0], [0.5, -0.5]])


def test_select_cross_fold_candidate_obeys_mean_known_constraint() -> None:
    candidates = [
        {
            "key": "aggressive",
            "fold_metrics": [
                {"known_accuracy": 0.94, "selection_score": 0.90},
                {"known_accuracy": 0.95, "selection_score": 0.90},
            ],
            "seed": 42,
        },
        {
            "key": "feasible",
            "fold_metrics": [
                {"known_accuracy": 0.95, "selection_score": 0.80},
                {"known_accuracy": 0.96, "selection_score": 0.82},
            ],
            "seed": 43,
        },
    ]

    chosen = select_cross_fold_candidate(candidates, min_known_accuracy=0.95)

    assert chosen["key"] == "feasible"
    assert chosen["mean_known_accuracy"] == 0.955
