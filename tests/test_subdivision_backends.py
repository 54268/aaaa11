import numpy as np

from functions.methods.unknown_subdivision import prototype_guided_clustering


def test_hdbscan_backend_finds_density_clusters_without_target_k() -> None:
    rng = np.random.default_rng(7)
    features = np.vstack(
        [
            rng.normal(loc=(-5.0, 0.0), scale=0.18, size=(40, 2)),
            rng.normal(loc=(0.0, 5.0), scale=0.18, size=(40, 2)),
            rng.normal(loc=(5.0, 0.0), scale=0.18, size=(40, 2)),
        ]
    ).astype(np.float32)

    result = prototype_guided_clustering(
        features,
        known_anchor_features=None,
        num_clusters=1,
        backend="hdbscan",
        density_min_cluster_size=12,
        density_min_samples=5,
    )

    assert result.resolved_k == 3
    assert set(np.unique(result.labels)).issuperset({0, 1, 2})
    assert result.diagnostics["density_backend"] == "hdbscan"
