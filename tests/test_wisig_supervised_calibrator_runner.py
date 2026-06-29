from __future__ import annotations

from run_wisig_supervised_calibrator import apply_formal_wisig_subdivision_config


def test_apply_formal_wisig_subdivision_config_uses_recovered_settings(tmp_path) -> None:
    config = {
        "unknown_subdivision": {
            "enabled": False,
            "feature_mode": "embedding_stats",
            "pca_dim": 96,
            "direct_confidence_quantile": 0.02,
            "direct_min_cluster_size": 80,
        }
    }
    predictions_path = tmp_path / "open_set_predictions.csv"

    apply_formal_wisig_subdivision_config(config, predictions_path)

    subdivision = config["unknown_subdivision"]
    assert subdivision["enabled"] is True
    assert subdivision["reuse_open_set_predictions"] is True
    assert subdivision["open_set_predictions_path"] == str(predictions_path)
    assert subdivision["feature_mode"] == "embedding_iq_stats"
    assert subdivision["pca_dim"] == 96
    assert subdivision["clustering_backend"] == "gmm_full_direct"
    assert subdivision["overcluster_extra_candidates"] == [0, 1, 2, 3]
    assert subdivision["m_selection_mode"] == "offline_min_gain"
    assert subdivision["merge_extra_clusters_to_target"] is True
    assert subdivision["direct_confidence_quantile"] == 0.0
    assert subdivision["direct_min_cluster_size"] == 0
