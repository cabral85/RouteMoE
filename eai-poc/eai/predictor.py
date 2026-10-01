"""Inference-time lookup: prompt fingerprint -> nearest centroid -> cluster's
activation profile -> predicted top-K experts per layer.

This module never touches the model or the router. It only reads the .eai index.
Nothing here blocks, reorders, or otherwise changes real routing - see the
project README's "does not touch the router" note.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .clustering import nearest_centroid
from .storage import EaiIndex


@dataclass
class PredictionResult:
    cluster_id: int
    centroid_distance: float
    top_experts_per_layer: np.ndarray  # (num_layers, stored_top_k) int32
    probabilities_per_layer: np.ndarray  # (num_layers, stored_top_k) float32
    lookup_seconds: float


class Predictor:
    def __init__(self, index: EaiIndex):
        self.index = index

    def predict(self, fingerprint: np.ndarray) -> PredictionResult:
        """Single-prompt prediction, timed - this is the number "index lookup time" reports."""
        t0 = time.perf_counter()
        cluster_id, dist = nearest_centroid(self.index.centroids, fingerprint)
        top_experts = self.index.top_experts[cluster_id]
        probs = self.index.probabilities[cluster_id]
        elapsed = time.perf_counter() - t0
        return PredictionResult(cluster_id, dist, top_experts, probs, elapsed)
