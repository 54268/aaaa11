from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .features import fft_magnitude_features, raw_iq_features, scale_and_pca
from .metrics import clustering_metrics


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    feature_type: str
    clustering_method: str


BASELINES = [
    BaselineSpec("Raw IQ + PCA + K-Means", "raw_iq", "kmeans"),
    BaselineSpec("Raw IQ + PCA + GMM", "raw_iq", "gmm"),
    BaselineSpec("FFT Magnitude + PCA + K-Means", "fft_magnitude", "kmeans"),
]


def run_baseline(x: np.ndarray, y_true: np.ndarray, n_clusters: int, spec: BaselineSpec, seed: int) -> tuple[dict, np.ndarray, int]:
    random.seed(seed)
    np.random.seed(seed)
    if spec.feature_type == "raw_iq":
        features = raw_iq_features(x)
    elif spec.feature_type == "fft_magnitude":
        features = fft_magnitude_features(x)
    else:
        raise ValueError(f"Unsupported feature_type: {spec.feature_type}")

    reduced, pca_dim = scale_and_pca(features, max_components=32, seed=seed)
    if spec.clustering_method == "kmeans":
        pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(reduced)
    elif spec.clustering_method == "gmm":
        pred = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            n_init=5,
            random_state=seed,
        ).fit_predict(reduced)
    else:
        raise ValueError(f"Unsupported clustering_method: {spec.clustering_method}")
    return clustering_metrics(y_true, pred), pred.astype(np.int64), pca_dim

