from __future__ import annotations

import numpy as np

from run_oracle_supervised_calibrator import (
    apply_formal_oracle_subdivision_config,
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


def test_apply_formal_oracle_subdivision_config_uses_true_auto_k(tmp_path) -> None:
    config = {
        "unknown_subdivision": {
            "enabled": False,
            "feature_mode": "embedding_distance",
            "pca_dim": 16,
            "target_num_clusters": 6,
        }
    }
    predictions_path = tmp_path / "open_set_predictions.csv"

    apply_formal_oracle_subdivision_config(config, predictions_path)

    subdivision = config["unknown_subdivision"]
    assert subdivision["enabled"] is True
    assert subdivision["reuse_open_set_predictions"] is True
    assert subdivision["open_set_predictions_path"] == str(predictions_path)
    assert subdivision["output_subdir"] == "unknown_subdivision_true_auto_k_sample_unified_k2_20_merge"
    assert subdivision["feature_mode"] == "embedding_stats"
    assert subdivision["pca_dim"] == 96
    assert subdivision["k_min"] == 2
    assert subdivision["k_max"] == 20
    assert subdivision["target_num_clusters"] is None
    assert subdivision["target_k_strength"] == 0.0
    assert subdivision["k_selection_mode"] == "sample_unified"
    assert subdivision["overcluster_extra_candidates"] == [0]
    assert subdivision["m_selection_mode"] == "unsupervised"
    assert subdivision["auto_merge_small_clusters"] is True
    assert subdivision["direct_confidence_quantile"] == 0.02
    assert subdivision["direct_min_cluster_size"] == 200
