import os
from pathlib import Path
import sys

import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openrfi_grouping import openrfi_world_prototype_grouping, openrfi_world_prototype_grouping_scores


def test_openrfi_world_grouping_uses_total_cluster_budget():
    rng = np.random.default_rng(0)
    clusters = []
    for center in ([0.0, 0.0], [5.0, 5.0], [10.0, 0.0]):
        clusters.append(rng.normal(loc=center, scale=0.05, size=(12, 2)))
    features = np.vstack(clusters).astype(np.float32)

    labels = openrfi_world_prototype_grouping(
        features,
        total_num_clusters=3,
        n_prototypes=6,
        seed=42,
        n_neighbors=1,
        graph_lambda=1.0,
    )

    assert labels.shape == (36,)
    assert len(np.unique(labels)) == 3


def test_openrfi_world_grouping_scores_align_with_labels():
    rng = np.random.default_rng(1)
    clusters = []
    for center in ([0.0, 0.0], [4.0, 4.0], [8.0, 0.0]):
        clusters.append(rng.normal(loc=center, scale=0.1, size=(10, 2)))
    features = np.vstack(clusters).astype(np.float32)

    labels, confidence = openrfi_world_prototype_grouping_scores(
        features,
        total_num_clusters=3,
        n_prototypes=6,
        seed=7,
        n_neighbors=1,
        graph_lambda=1.0,
    )

    plain_labels = openrfi_world_prototype_grouping(
        features,
        total_num_clusters=3,
        n_prototypes=6,
        seed=7,
        n_neighbors=1,
        graph_lambda=1.0,
    )

    assert np.array_equal(labels, plain_labels)
    assert confidence.shape == (30,)
    assert np.all(np.isfinite(confidence))
    assert confidence.max() > confidence.min()
