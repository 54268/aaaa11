from __future__ import annotations

import numpy as np

from functions.methods.openmax_backends import RepoOpenMaxAdapter


def test_repo_openmax_backend_runs_from_tracked_source() -> None:
    activations = np.asarray(
        [
            [4.0, 0.2],
            [3.8, 0.3],
            [4.2, 0.1],
            [0.2, 4.0],
            [0.3, 3.8],
            [0.1, 4.2],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    adapter = RepoOpenMaxAdapter(alpha_rank=2, tail_size=2, distance_type="eucl")

    adapter.fit(activations, labels, labels)
    result = adapter.predict(np.asarray([[4.1, 0.2], [0.2, 4.1]], dtype=np.float32))

    assert result["known_probs"].shape == (2, 2)
    assert result["unknown_prob"].shape == (2,)
    assert np.isfinite(result["known_probs"]).all()
    assert np.isfinite(result["unknown_prob"]).all()
    np.testing.assert_allclose(
        result["known_probs"].sum(axis=1) + result["unknown_prob"],
        np.ones(2),
        atol=1e-6,
    )
