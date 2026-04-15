from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.modules.boundary_mining import mine_boundary_samples
from sei_osr.modules.pseudo_unknown import generate_hybrid_pseudo_unknown


def build_toy_data():
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [1.0, 1.0],
            [1.2, 1.1],
            [1.4, 1.3],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    prototypes = np.asarray(
        [
            embeddings[labels == 0].mean(axis=0),
            embeddings[labels == 1].mean(axis=0),
        ],
        dtype=np.float32,
    )
    return embeddings, labels, prototypes


def test_boundary_and_pseudo_pipeline():
    embeddings, labels, prototypes = build_toy_data()
    boundary = mine_boundary_samples(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        k=2,
        alpha=0.5,
        top_m=1,
        ordinary_edge_ratio=0.34,
    )
    assert boundary["critical_mask"].sum() == 2
    assert boundary["ordinary_edge_mask"].sum() >= 1

    pseudo = generate_hybrid_pseudo_unknown(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        boundary_result=boundary,
        ordinary_eta=1.0,
        critical_eta=1.0,
        critical_beta=0.7,
        ordinary_variations=1,
        critical_variations=2,
        jitter=0.0,
        seed=0,
    )
    assert pseudo["summary"]["num_total_pseudo"] == len(pseudo["pseudo_embeddings"])
    assert pseudo["summary"]["num_critical_pseudo"] == 4
    assert pseudo["pseudo_embeddings"].shape[1] == 2
