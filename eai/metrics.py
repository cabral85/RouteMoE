"""Metrics comparing predicted top-K experts against what the router actually
selected, plus the two frequency baselines every result should be read against.

Ground truth granularity: evaluated per TOKEN, against the model's own
selected_experts for that token (always exactly `top_k` ids). A prompt's single
prediction (the fingerprint/cluster pipeline predicts once per prompt per
layer, not once per token) is broadcast to every token in that prompt before
comparing.

Earlier version of this module defined "actual" as the *union* of experts used
anywhere in a prompt. That denominator saturates fast: with 64 experts and
~8 selected per token, a ~20-token prompt already touches most of the expert
vocabulary at least once, making recall@K tiny and nearly identical for every
predictor regardless of how good it is. Per-token ground truth keeps the
denominator fixed at the model's real top_k and directly answers the question
that matters for prefetch/caching: "if we warmed the predicted experts before
running this prompt, what fraction of the router's real per-token decisions
would have been covered?"
"""

from __future__ import annotations

import numpy as np

from .tracing import TraceSet


def evaluate_predictions(
    predicted_top_experts: np.ndarray,  # (num_prompts, L, K) or (total_tokens, L, K), ranked desc, -1 = no candidate
    traces: TraceSet,
    num_experts: int,
    ks: tuple[int, ...] = (1, 4, 8),
    granularity: str = "prompt",
) -> dict:
    """Per-layer and aggregate top1/recall@k/precision@k for one predictor's output.

    granularity="prompt": `predicted_top_experts` has one row per prompt and is
    broadcast to that prompt's tokens via `traces.token_prompt_idx`.
    granularity="token": `predicted_top_experts` already has one row per token
    (same order as `traces.token_prompt_idx`/`traces.selected_experts`), used as-is.
    """
    num_layers = traces.num_layers
    total_tokens = traces.selected_experts.shape[1]
    row_idx = np.arange(total_tokens)

    top1_hits = np.zeros((total_tokens, num_layers), dtype=np.float64)
    recall = {k: np.zeros((total_tokens, num_layers), dtype=np.float64) for k in ks}
    precision = {k: np.zeros((total_tokens, num_layers), dtype=np.float64) for k in ks}

    for l in range(num_layers):
        actual_ids = traces.selected_experts[l]  # (total_tokens, model_top_k)
        actual_presence = np.zeros((total_tokens, num_experts), dtype=bool)
        actual_presence[row_idx[:, None], actual_ids] = True
        actual_count = actual_ids.shape[1]  # model's own top_k - constant, always > 0

        if granularity == "prompt":
            pred_for_tokens = predicted_top_experts[traces.token_prompt_idx, l]  # (total_tokens, stored_top_k)
        elif granularity == "token":
            pred_for_tokens = predicted_top_experts[:, l]  # (total_tokens, stored_top_k)
        else:
            raise ValueError(f"unknown granularity: {granularity}")

        top1 = pred_for_tokens[:, 0]
        top1_valid = top1 >= 0
        safe_top1 = np.clip(top1, 0, num_experts - 1)
        top1_hits[:, l] = (actual_presence[row_idx, safe_top1] & top1_valid).astype(np.float64)

        for k in ks:
            pred_k = pred_for_tokens[:, :k]
            valid = pred_k >= 0
            safe_ids = np.clip(pred_k, 0, num_experts - 1)
            hit_bool = actual_presence[row_idx[:, None], safe_ids] & valid
            hits = hit_bool.sum(axis=1)
            recall[k][:, l] = hits / actual_count
            precision[k][:, l] = hits / np.clip(valid.sum(axis=1), 1, None)

    per_layer = {
        "top1_accuracy": top1_hits.mean(axis=0),
        "recall_at_k": {k: recall[k].mean(axis=0) for k in ks},
        "precision_at_k": {k: precision[k].mean(axis=0) for k in ks},
    }
    aggregate = {
        "top1_accuracy": float(top1_hits.mean()),
        "recall_at_k": {k: float(recall[k].mean()) for k in ks},
        "precision_at_k": {k: float(precision[k].mean()) for k in ks},
    }
    return {"per_layer": per_layer, "aggregate": aggregate}


def coverage_at(cluster_ids: np.ndarray, observation_counts: np.ndarray, min_observations: int = 3) -> np.ndarray:
    """(N, L) bool: whether the cluster/layer cell backing each prediction had enough
    training observations to be trusted, rather than being a cold/sparse cell."""
    return observation_counts[cluster_ids] >= min_observations


def baseline_global_topk(global_expert_freq: np.ndarray, k_store: int) -> np.ndarray:
    """Baseline 1: one ranked list (mean frequency across all layers), reused for every layer."""
    overall = global_expert_freq.mean(axis=0)
    return np.argsort(-overall)[:k_store].astype(np.int32)


def baseline_per_layer_topk(global_expert_freq: np.ndarray, k_store: int) -> np.ndarray:
    """Baseline 2: (num_layers, k_store) most frequent experts, independently per layer."""
    num_layers = global_expert_freq.shape[0]
    return np.stack(
        [np.argsort(-global_expert_freq[l])[:k_store] for l in range(num_layers)]
    ).astype(np.int32)


def broadcast_baseline(per_layer_order: np.ndarray, num_prompts: int) -> np.ndarray:
    """Tile a fixed (num_layers, k_store) prediction (shared by every prompt) to (N, L, k_store)."""
    return np.tile(per_layer_order[None, :, :], (num_prompts, 1, 1))


def latency_stats(seconds: np.ndarray) -> dict:
    ms = seconds * 1000.0
    return {
        "mean_ms": float(np.mean(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "max_ms": float(np.max(ms)),
    }
