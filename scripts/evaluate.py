#!/usr/bin/env python
"""Phase 5+6: load the .eai index, predict expert activation for the test set,
and compare against what the router actually did - plus two frequency baselines.

    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai import metrics as M
from eai.fingerprint import compute_fingerprints
from eai.predictor import Predictor
from eai.storage import load_index
from eai.tracing import TraceSet, load_prompts_jsonl


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="artifacts/model.eai")
    parser.add_argument("--traces", default="artifacts/traces_test.npz")
    parser.add_argument("--prompts", default="data/prompts_test.jsonl")
    parser.add_argument("--ks", default="1,4,8", help="comma-separated K values for recall/precision@K")
    parser.add_argument("--min-observations", type=int, default=3, help="coverage threshold")
    args = parser.parse_args()
    ks = tuple(int(x) for x in args.ks.split(","))

    index = load_index(args.index)
    traces = TraceSet.load(args.traces)
    md = index.metadata
    total_tokens = traces.selected_experts.shape[1]

    prompts_by_id = None
    if md.fingerprint_method == "hashing":
        prompts_by_id = {p["id"]: p["text"] for p in load_prompts_jsonl(args.prompts)}
    fingerprints = compute_fingerprints(
        md.fingerprint_method, traces, prompts_by_id, dim=md.fingerprint_dimension, granularity=md.granularity
    )
    num_rows = fingerprints.shape[0]  # num_prompts (granularity="prompt") or total_tokens (granularity="token")

    predictor = Predictor(index)
    stored_k = md.stored_top_k
    our_top_experts = np.full((num_rows, md.num_layers, stored_k), -1, dtype=np.int32)
    our_confidence = np.zeros((num_rows, md.num_layers), dtype=np.float32)
    cluster_ids = np.zeros(num_rows, dtype=np.int32)
    lookup_seconds = np.zeros(num_rows, dtype=np.float64)
    for i in range(num_rows):
        result = predictor.predict(fingerprints[i])
        our_top_experts[i] = result.top_experts_per_layer
        our_confidence[i] = index.confidence[result.cluster_id]
        cluster_ids[i] = result.cluster_id
        lookup_seconds[i] = result.lookup_seconds

    our_eval = M.evaluate_predictions(our_top_experts, traces, md.num_experts, ks=ks, granularity=md.granularity)
    coverage = M.coverage_at(cluster_ids, index.observation_counts, args.min_observations)

    # baselines don't depend on cluster/prompt at all (same list everywhere), so
    # they're always evaluated directly at token granularity regardless of md.granularity.
    b1_layer = np.tile(
        M.baseline_global_topk(index.global_expert_freq, stored_k)[None, :], (md.num_layers, 1)
    )
    b1_pred = M.broadcast_baseline(b1_layer, total_tokens)
    b1_eval = M.evaluate_predictions(b1_pred, traces, md.num_experts, ks=ks, granularity="token")

    b2_layer = M.baseline_per_layer_topk(index.global_expert_freq, stored_k)
    b2_pred = M.broadcast_baseline(b2_layer, total_tokens)
    b2_eval = M.evaluate_predictions(b2_pred, traces, md.num_experts, ks=ks, granularity="token")

    lat = M.latency_stats(lookup_seconds)
    fp_bytes = Path(args.index).stat().st_size
    avg_fwd_ms = float(traces.forward_pass_seconds.mean()) * 1000

    print(f"Model: {md.model_id}")
    print(f"Layers: {md.num_layers}")
    print(f"Experts/layer: {md.num_experts}  (router top_k={md.top_k})")
    if md.quantization != "none":
        print(f"** QUANTIZED ({md.quantization}) - router ground truth may differ from full precision **")
    print()
    print("Index:")
    print(f"  clusters: {md.number_of_clusters}")
    print(f"  fingerprint: {md.fingerprint_method} ({md.fingerprint_dimension}d)")
    print(f"  stored candidates/cell: {stored_k}")
    print(f"  size: {fp_bytes / (1024 * 1024):.2f} MB")
    print()
    print(f"Test set: {traces.num_prompts} prompts, {total_tokens} tokens")
    print()
    print(f"Prediction - cluster-based (ours, granularity={md.granularity}):")
    print(f"  Top-1 accuracy: {pct(our_eval['aggregate']['top1_accuracy'])}")
    for k in ks:
        print(
            f"  Recall@{k}: {pct(our_eval['aggregate']['recall_at_k'][k]):<8} "
            f"Precision@{k}: {pct(our_eval['aggregate']['precision_at_k'][k])}"
        )
    print(f"  Coverage (>= {args.min_observations} train obs): {pct(float(coverage.mean()))}")
    print(f"  Mean prediction confidence: {float(our_confidence.mean()):.3f}")
    print()
    print("Baseline 1 - globally most-frequent experts (same list, every layer):")
    print(f"  Top-1 accuracy: {pct(b1_eval['aggregate']['top1_accuracy'])}")
    for k in ks:
        print(
            f"  Recall@{k}: {pct(b1_eval['aggregate']['recall_at_k'][k]):<8} "
            f"Precision@{k}: {pct(b1_eval['aggregate']['precision_at_k'][k])}"
        )
    print()
    print("Baseline 2 - most-frequent experts per layer:")
    print(f"  Top-1 accuracy: {pct(b2_eval['aggregate']['top1_accuracy'])}")
    for k in ks:
        print(
            f"  Recall@{k}: {pct(b2_eval['aggregate']['recall_at_k'][k]):<8} "
            f"Precision@{k}: {pct(b2_eval['aggregate']['precision_at_k'][k])}"
        )
    print()
    top_k_report = ks[-1]
    print(f"Per-layer Recall@{top_k_report} (ours vs baseline-2):")
    ours_pl = our_eval["per_layer"]["recall_at_k"][top_k_report]
    b2_pl = b2_eval["per_layer"]["recall_at_k"][top_k_report]
    for l in range(md.num_layers):
        print(f"  layer {l:2d}: ours={pct(ours_pl[l]):>7}   baseline2={pct(b2_pl[l]):>7}")
    print()
    print("Index lookup (nearest-centroid search only):")
    print(f"  mean: {lat['mean_ms']:.3f} ms   p95: {lat['p95_ms']:.3f} ms   max: {lat['max_ms']:.3f} ms")
    print()
    print(f"Reference: mean full-model forward pass = {avg_fwd_ms:.1f} ms/prompt")
    print(f"Predictor overhead vs full forward pass: {100 * lat['mean_ms'] / avg_fwd_ms:.4f}%")


if __name__ == "__main__":
    main()
