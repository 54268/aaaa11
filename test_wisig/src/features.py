from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def per_sample_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    return (x - mean) / (std + 1e-6)


def raw_iq_features(x: np.ndarray) -> np.ndarray:
    x = per_sample_normalize(x)
    return x.reshape(len(x), -1).astype(np.float32)


def fft_magnitude_features(x: np.ndarray) -> np.ndarray:
    x = per_sample_normalize(x)
    z = x[:, 0, :] + 1j * x[:, 1, :]
    fft = np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)
    return np.log1p(np.abs(fft)).astype(np.float32)


def scale_and_pca(features: np.ndarray, max_components: int = 32, seed: int = 0) -> tuple[np.ndarray, int]:
    features = np.asarray(features, dtype=np.float32)
    scaled = StandardScaler().fit_transform(features)
    n_components = int(min(max_components, scaled.shape[0] - 1, scaled.shape[1]))
    if n_components < 1:
        return scaled, int(scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(scaled)
    return reduced.astype(np.float32), n_components

