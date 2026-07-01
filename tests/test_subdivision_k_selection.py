import numpy as np

from functions.subdivision_pipeline import _merge_unbalanced_small_clusters
from functions.methods.unknown_subdivision import _select_k_by_unified_score, run_ofscil_subdivision


def test_unified_k_score_prefers_balanced_likelihood_over_under_split_cluster():
    history = [
        {"k": 3, "score": 1.18, "gmm_lower_bound": 64.27, "cluster_max_balance": 0.98},
        {"k": 6, "score": 0.55, "gmm_lower_bound": 77.99, "cluster_max_balance": 0.49},
        {"k": 7, "score": -3.77, "gmm_lower_bound": 78.88, "cluster_max_balance": 0.94},
        {"k": 8, "score": -8.89, "gmm_lower_bound": 79.58, "cluster_max_balance": 0.86},
        {"k": 12, "score": -9.36, "gmm_lower_bound": 82.03, "cluster_max_balance": 0.57},
    ]

    selected_k, enriched = _select_k_by_unified_score(history)

    assert selected_k == 7
    assert max(enriched, key=lambda row: row["auto_k_score"])["k"] == 7


def test_unified_k_score_can_select_higher_split_when_likelihood_gain_is_large():
    history = [
        {"k": 10, "score": 1.17, "gmm_lower_bound": 99.19, "cluster_max_balance": 0.60},
        {"k": 11, "score": 1.24, "gmm_lower_bound": 102.16, "cluster_max_balance": 0.55},
        {"k": 12, "score": 0.88, "gmm_lower_bound": 103.05, "cluster_max_balance": 0.50},
        {"k": 13, "score": 0.91, "gmm_lower_bound": 104.93, "cluster_max_balance": 0.91},
    ]

    selected_k, enriched = _select_k_by_unified_score(history)

    assert selected_k == 13
    assert max(enriched, key=lambda row: row["auto_k_score"])["k"] == 13


def test_fixed_candidate_without_target_records_fitted_k():
    rng = np.random.default_rng(123)
    features = np.vstack(
        [
            rng.normal(loc=-3.0, scale=0.2, size=(12, 2)),
            rng.normal(loc=0.0, scale=0.2, size=(12, 2)),
            rng.normal(loc=3.0, scale=0.2, size=(12, 2)),
        ]
    )

    result = run_ofscil_subdivision(
        features,
        known_anchor_features=None,
        k_min=3,
        k_max=3,
        seed=123,
        backend="gmm_full_direct",
        target_num_clusters=None,
        target_k_strength=0.0,
        n_init=5,
        direct_confidence_quantile=0.0,
        direct_min_cluster_size=0,
    )

    assert result.diagnostics["selected_num_clusters"] == 3
    assert result.resolved_k == 3


def test_balance_merge_combines_small_oversplit_cluster_without_target_k():
    features = np.vstack(
        [
            np.full((10, 2), [-2.0, 0.0]),
            np.full((10, 2), [2.0, 0.0]),
            np.full((4, 2), [2.2, 0.1]),
        ]
    )
    labels = np.asarray([0] * 10 + [1] * 10 + [2] * 4)

    merged, centers, info = _merge_unbalanced_small_clusters(
        features,
        labels,
        max_source_mean_ratio=0.60,
        min_balance_gain=0.20,
    )

    assert info["auto_merged_cluster_count"] == 1
    assert info["post_auto_merge_num_clusters"] == 2
    assert sorted(np.unique(merged).tolist()) == [0, 1]
    assert centers.shape == (2, 2)
