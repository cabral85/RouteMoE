"""Build per-(cluster, layer) expert activation profiles from training traces.

For every cluster/layer cell we keep: the top-K experts ranked by how often they
fired for tokens in that cluster at that layer, their activation frequency
("probabilities"), how many token observations back that estimate, a confidence
score, and a small pairwise co-activation matrix among the stored experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracing import TraceSet


@dataclass
class ActivationProfiles:
    num_clusters: int
    num_layers: int
    num_experts: int
    stored_top_k: int
    top_experts: np.ndarray  # (K, L, stored_top_k) int32, expert id ranked by frequency desc
    probabilities: np.ndarray  # (K, L, stored_top_k) float32, activation frequency in [0, 1]
    observation_counts: np.ndarray  # (K, L) int32, token observations backing the estimate
    confidence: np.ndarray  # (K, L) float32, mean probability over the stored top-K
    coactivation: np.ndarray  # (K, L, stored_top_k, stored_top_k) float32, joint frequency
    global_expert_freq: np.ndarray  # (L, num_experts) float32, all-clusters frequency (for baselines)


def build_profiles(
    traces: TraceSet,
    labels: np.ndarray,
    num_clusters: int,
    num_experts: int,
    stored_top_k: int,
    granularity: str = "prompt",
) -> ActivationProfiles:
    """`labels` is per-prompt (granularity="prompt", broadcast to that prompt's
    tokens) or already per-token (granularity="token", used directly)."""
    num_layers = traces.num_layers
    total_tokens = traces.selected_experts.shape[1]
    if granularity == "prompt":
        token_cluster = labels[traces.token_prompt_idx]  # (total_tokens,) cluster id per token row
    elif granularity == "token":
        token_cluster = labels  # already one label per token row
    else:
        raise ValueError(f"unknown granularity: {granularity}")
    cluster_masks = [token_cluster == c for c in range(num_clusters)]

    top_experts = np.full((num_clusters, num_layers, stored_top_k), -1, dtype=np.int32)
    probabilities = np.zeros((num_clusters, num_layers, stored_top_k), dtype=np.float32)
    observation_counts = np.zeros((num_clusters, num_layers), dtype=np.int32)
    confidence = np.zeros((num_clusters, num_layers), dtype=np.float32)
    coactivation = np.zeros((num_clusters, num_layers, stored_top_k, stored_top_k), dtype=np.float32)
    global_expert_freq = np.zeros((num_layers, num_experts), dtype=np.float32)

    for l in range(num_layers):
        sel = traces.selected_experts[l]  # (total_tokens, model_top_k) expert ids
        presence = np.zeros((total_tokens, num_experts), dtype=bool)
        presence[np.arange(total_tokens)[:, None], sel] = True
        global_expert_freq[l] = presence.mean(axis=0)

        for c in range(num_clusters):
            mask = cluster_masks[c]
            n_obs = int(mask.sum())
            observation_counts[c, l] = n_obs
            if n_obs == 0:
                continue
            freq = presence[mask].mean(axis=0)  # (num_experts,)
            order = np.argsort(-freq, kind="stable")[:stored_top_k]
            top_experts[c, l] = order
            probabilities[c, l] = freq[order]
            confidence[c, l] = float(probabilities[c, l].mean())

            sub = presence[mask][:, order].astype(np.float32)  # (n_obs, stored_top_k)
            coactivation[c, l] = (sub.T @ sub) / n_obs

    return ActivationProfiles(
        num_clusters=num_clusters,
        num_layers=num_layers,
        num_experts=num_experts,
        stored_top_k=stored_top_k,
        top_experts=top_experts,
        probabilities=probabilities,
        observation_counts=observation_counts,
        confidence=confidence,
        coactivation=coactivation,
        global_expert_freq=global_expert_freq,
    )
