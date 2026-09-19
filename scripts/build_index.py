#!/usr/bin/env python
"""Phase 3+4: cluster training prompts by fingerprint, build per-cluster activation
profiles, and persist everything into artifacts/model.eai.

    python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.clustering import fit_clusters
from eai.fingerprint import FINGERPRINT_METHODS, GRANULARITIES, compute_fingerprints
from eai.profile import build_profiles
from eai.storage import EAI_VERSION, EaiMetadata, save_index
from eai.tracing import TraceSet, load_prompts_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="artifacts/traces_train.npz")
    parser.add_argument("--prompts", default="data/prompts_train.jsonl")
    parser.add_argument("--out", default="artifacts/model.eai")
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--fingerprint-method", default="hidden_state", choices=FINGERPRINT_METHODS)
    parser.add_argument("--granularity", default="prompt", choices=GRANULARITIES,
                         help="cluster/predict per whole prompt (default) or per individual token")
    parser.add_argument("--hashing-dim", type=int, default=256)
    parser.add_argument("--top-k-store", type=int, default=16)
    parser.add_argument("--algorithm", default="kmeans", choices=["kmeans", "minibatch"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading traces from {args.traces} ...")
    traces = TraceSet.load(args.traces)
    total_tokens = traces.selected_experts.shape[1]
    print(f"  {traces.num_prompts} prompts, {traces.num_layers} layers, {total_tokens} total tokens")

    prompts_by_id = None
    if args.fingerprint_method == "hashing":
        prompts_by_id = {p["id"]: p["text"] for p in load_prompts_jsonl(args.prompts)}

    fingerprints = compute_fingerprints(
        args.fingerprint_method, traces, prompts_by_id, dim=args.hashing_dim, granularity=args.granularity
    )
    fp_dim = fingerprints.shape[1]
    unit = "prompts" if args.granularity == "prompt" else "tokens"
    print(f"Fingerprint method={args.fingerprint_method} granularity={args.granularity} dim={fp_dim} ({fingerprints.shape[0]} {unit})")

    print(f"Clustering into K={args.clusters} ({args.algorithm}) ...")
    centroids, labels = fit_clusters(
        fingerprints, args.clusters, seed=args.seed, minibatch=(args.algorithm == "minibatch")
    )
    actual_k = centroids.shape[0]
    if actual_k != args.clusters:
        print(f"  (reduced to K={actual_k}: fewer training {unit} than requested clusters)")
    counts = np.bincount(labels, minlength=actual_k)
    print(f"  cluster sizes: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")

    print("Building activation profiles ...")
    profiles = build_profiles(
        traces, labels, actual_k, traces.num_experts, args.top_k_store, granularity=args.granularity
    )

    metadata = EaiMetadata(
        version=EAI_VERSION,
        model_id=traces.model_id,
        architecture=traces.architecture,
        num_layers=traces.num_layers,
        num_experts=traces.num_experts,
        top_k=traces.top_k,
        quantization=traces.quantization,
        stored_top_k=args.top_k_store,
        fingerprint_method=args.fingerprint_method,
        fingerprint_dimension=fp_dim,
        granularity=args.granularity,
        number_of_clusters=actual_k,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_index(str(out_path), metadata, profiles, centroids)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print()
    print(f"Saved {out_path} ({size_mb:.2f} MB)")
    print(f"Model: {metadata.model_id}")
    print(f"Layers: {metadata.num_layers}  Experts/layer: {metadata.num_experts}  top_k: {metadata.top_k}")
    print(
        f"Index: clusters={metadata.number_of_clusters}  granularity={metadata.granularity}  "
        f"fingerprint={metadata.fingerprint_method}({fp_dim}d)  size={size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
