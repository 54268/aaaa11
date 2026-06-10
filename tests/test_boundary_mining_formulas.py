from __future__ import annotations

import numpy as np

from functions.methods.boundary_mining import mine_boundary_samples


def test_boundary_mining_uses_formula_terms_and_classwise_thresholds() -> None:
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.9, 0.0],
            [1.2, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [10.4, 0.0],
            [11.4, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    prototypes = np.asarray([[0.5, 0.0], [10.5, 0.0]], dtype=np.float32)

    result = mine_boundary_samples(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        k=1,
        beta=2.0,
        alpha=0.5,
        top_m=1,
        ordinary_edge_ratio=0.25,
    )

    expected_own_sq = np.sum((embeddings - prototypes[labels]) ** 2, axis=1)
    expected_local_sparsity = np.asarray(
        [0.04, 0.04, 0.09, 0.09, 0.01, 0.01, 0.09, 1.0],
        dtype=np.float32,
    )
    expected_marginality = expected_own_sq + 2.0 * expected_local_sparsity

    np.testing.assert_allclose(result["prototype_deviation"], expected_own_sq, atol=1e-5)
    np.testing.assert_allclose(result["local_sparsity"], expected_local_sparsity, atol=1e-5)
    np.testing.assert_allclose(result["local_marginality"], expected_marginality, rtol=1e-4, atol=1e-5)

    thresholds = [result["summary"][cls]["marginal_threshold"] for cls in (0, 1)]
    assert thresholds[0] != thresholds[1]
    assert np.all(result["critical_mask"] <= result["marginal_mask"])


def test_negative_prototype_competition_distance_is_excluded() -> None:
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [8.0, 0.0],
            [10.0, 0.0],
            [10.2, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    prototypes = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

    result = mine_boundary_samples(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        k=1,
        beta=1.0,
        alpha=0.5,
        top_m=1,
        ordinary_edge_ratio=0.34,
    )

    negative_indices = np.where(result["competition_distance"] < 0.0)[0]
    assert set(negative_indices.tolist()) == {2, 5}
    assert not np.any(result["critical_mask"][negative_indices])
    assert not np.any(result["ordinary_edge_mask"][negative_indices])
    assert np.all(result["noise_mask"][negative_indices])

    for cls in (0, 1):
        selected = np.where(result["critical_mask"] & (labels == cls))[0]
        threshold = result["summary"][cls]["competition_threshold"]
        assert np.all(result["competition_distance"][selected] >= 0.0)
        assert np.all(result["competition_distance"][selected] <= threshold + 1e-7)
