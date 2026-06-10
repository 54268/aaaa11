from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + eps)


def _openrfi_world_prototype_grouping_core(
    features: np.ndarray,
    *,
    total_num_clusters: int,
    n_prototypes: int,
    seed: int,
    n_neighbors: int = 3,
    graph_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style OpenRFI prototype-graph grouping over a full unlabeled pool."""

    total_num_clusters = int(total_num_clusters)
    if total_num_clusters <= 0:
        raise ValueError("total_num_clusters must be positive")

    feat = _normalize_rows(features)
    proto_count = min(max(total_num_clusters, int(n_prototypes)), max(total_num_clusters, len(feat) // 2))
    proto_kmeans = KMeans(n_clusters=proto_count, n_init=20, random_state=seed).fit(feat)
    proto_centers = _normalize_rows(proto_kmeans.cluster_centers_)
    proto_sim = feat @ proto_centers.T

    n_top = min(int(n_neighbors) + 1, proto_count)
    top_indices = np.argpartition(proto_sim, kth=-n_top, axis=1)[:, -n_top:]
    anchor_index = np.argmax(np.take_along_axis(proto_sim, top_indices, axis=1), axis=1)
    anchor_proto = top_indices[np.arange(len(top_indices)), anchor_index]

    edge_graph = np.zeros((proto_count, proto_count), dtype=np.float64)
    for offset in range(n_top):
        np.add.at(edge_graph, (anchor_proto, top_indices[:, offset]), 1.0)

    diag = np.diag(edge_graph).copy()
    edge_graph = edge_graph / np.maximum(diag[:, None], 1.0)
    edge_graph = (edge_graph + edge_graph.T) / 2.0
    np.fill_diagonal(edge_graph, 0.0)

    attr_graph = (proto_centers @ proto_centers.T + 1.0) / 2.0
    graph = float(graph_lambda) * edge_graph + (1.0 - float(graph_lambda)) * attr_graph
    np.fill_diagonal(graph, np.maximum(np.diag(graph), 1e-6))

    proto_labels = SpectralClustering(
        n_clusters=total_num_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=20,
    ).fit_predict(graph)

    nearest_proto = proto_sim.argmax(axis=1)
    labels = proto_labels[nearest_proto].astype(np.int64)
    confidence = proto_sim.max(axis=1).astype(np.float64)
    return labels, confidence


def openrfi_world_prototype_grouping(
    features: np.ndarray,
    *,
    total_num_clusters: int,
    n_prototypes: int,
    seed: int,
    n_neighbors: int = 3,
    graph_lambda: float = 1.0,
) -> np.ndarray:
    labels, _ = _openrfi_world_prototype_grouping_core(
        features,
        total_num_clusters=total_num_clusters,
        n_prototypes=n_prototypes,
        seed=seed,
        n_neighbors=n_neighbors,
        graph_lambda=graph_lambda,
    )
    return labels


def openrfi_world_prototype_grouping_scores(
    features: np.ndarray,
    *,
    total_num_clusters: int,
    n_prototypes: int,
    seed: int,
    n_neighbors: int = 3,
    graph_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    return _openrfi_world_prototype_grouping_core(
        features,
        total_num_clusters=total_num_clusters,
        n_prototypes=n_prototypes,
        seed=seed,
        n_neighbors=n_neighbors,
        graph_lambda=graph_lambda,
    )
