"""Group prompts by fingerprint similarity into K clusters (the PGO-style "buckets"
each activation profile is keyed on).
"""

from __future__ import annotations

import numpy as np


def fit_clusters(
    fingerprints: np.ndarray, k: int, seed: int = 42, minibatch: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans (or MiniBatchKMeans for larger N) on the fingerprint matrix.

    Returns (centroids [k, dim] float32, labels [N] int32).
    """
    n = fingerprints.shape[0]
    k = min(k, n)  # KMeans can't make more clusters than points
    if minibatch:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=min(256, n))
    else:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=seed, n_init=10)

    labels = model.fit_predict(fingerprints)
    return model.cluster_centers_.astype(np.float32), labels.astype(np.int32)


def nearest_centroid(centroids: np.ndarray, fingerprint: np.ndarray) -> tuple[int, float]:
    """Nearest centroid to a single fingerprint vector by Euclidean distance."""
    dists = np.linalg.norm(centroids - fingerprint[None, :], axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])
