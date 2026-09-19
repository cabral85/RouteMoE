#!/usr/bin/env python
"""Round 5: token-by-token streaming with real eviction - the piece Round 4
identified as missing. Round 4 prefetched a whole prompt's union of predicted
experts before running anything; this instead processes one token at a time
(with a KV cache, like real generation) and evicts down to (ideally) just
what the next token needs between steps - literal load -> process -> unload,
matching what github.com/josesilva05/kimi-k3-in-c calls "expert streaming".

Three policies compared, all against the same prompts:
  1. no_eviction  - Round 4's approach: accumulate everything, never evict
                    (the ~88-99%-of-full-model ceiling already measured).
  2. lru          - plain least-recently-used, capped at a fixed budget,
                    zero prediction at all - the simplest possible baseline,
                    and what a system with no EAI index would fall back to.
  3. eai_predict  - evict to exactly what the EAI index predicts for the
                    upcoming token(s) before each step - prediction actually
                    driving memory, not just scored for accuracy.

Reports, per policy: peak resident memory across the whole generation (not
cumulative-by-end-of-run, which Round 4 showed is the wrong number), hit
rate, and correctness against a fresh same-process reference (same rigorous
methodology as Round 4 - see README).

    python scripts/streaming_eviction_experiment.py
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.chunked_expert_loader import load_chunked_model
from eai.fingerprint import compute_fingerprints
from eai.predictor import Predictor
from eai.storage import load_index
from eai.tracing import TraceSet, load_prompts_jsonl


def find_snapshot_dir(model_id: str) -> str:
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = model_id.split("/")
    pattern = os.path.join(cache_home, "hub", f"models--{org}--{name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no cached snapshot found for {model_id} under {pattern}")
    return matches[0]


def run_streaming(model, blocks, input_ids: torch.Tensor, policy: str,
                   per_token_predictions: np.ndarray | None, lru_budget: int,
                   lookahead: int = 0) -> dict:
    """Replay `input_ids` one token at a time through `model`, using a KV
    cache (like real generation), applying `policy`'s eviction rule between
    steps. Returns peak/mean resident bytes and per-step selected experts
    (for correctness checking against a reference)."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    num_tokens = input_ids.shape[1]
    peak_resident_bytes = 0
    resident_trace = []
    selected_per_step = []  # list of (num_layers,) arrays of the top-k experts, one row per layer

    for t in range(num_tokens):
        if policy == "eai_predict" and per_token_predictions is not None:
            # Evict down to what THIS token (plus a small lookahead window,
            # if any) needs, then prefetch that exact set - load, process,
            # unload, matching the "frenetic I/O" pattern this round tests.
            window = per_token_predictions[t : t + 1 + lookahead]  # (window, num_layers, stored_top_k)
            for layer_idx, block in enumerate(blocks):
                keep = set(int(e) for e in window[:, layer_idx, :].flatten() if e >= 0)
                block.evict_except(keep)
                block.prefetch(sorted(keep))
        elif policy == "lru":
            for block in blocks:
                block.evict_lru(lru_budget)
        # policy == "no_eviction": never evict, matches Round 4's behavior

        next_id = input_ids[:, t : t + 1]
        with torch.no_grad():
            model(input_ids=next_id, past_key_values=cache, use_cache=True)

        step_resident = sum(b.resident_bytes for b in blocks)
        peak_resident_bytes = max(peak_resident_bytes, step_resident)
        resident_trace.append(step_resident)
        selected_per_step.append(np.stack([b.last_selected[0].numpy() for b in blocks]))  # (num_layers, top_k)

    return {
        "peak_resident_bytes": peak_resident_bytes,
        "mean_resident_bytes": float(np.mean(resident_trace)),
        "resident_trace": resident_trace,
        "selected_per_step": np.stack(selected_per_step),  # (num_tokens, num_layers, top_k)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--index", default="artifacts/model_token128.eai")
    parser.add_argument("--traces", default="artifacts/traces_test.npz")
    parser.add_argument("--prompts", default="data/prompts_test.jsonl")
    parser.add_argument("--num-prompts", type=int, default=15)
    parser.add_argument("--lru-budget", type=int, default=16, help="experts/layer kept resident under the LRU policy")
    parser.add_argument("--lookahead", type=int, default=0, help="extra upcoming tokens' predictions to also keep resident")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--num-correctness-prompts", type=int, default=3)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    model_dir = find_snapshot_dir(args.model)
    index = load_index(args.index)
    md = index.metadata
    traces = TraceSet.load(args.traces)
    prompts = {p["id"]: p["text"] for p in load_prompts_jsonl(args.prompts)}
    all_fingerprints = compute_fingerprints(md.fingerprint_method, traces, granularity="token")
    predictor = Predictor(index)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Model: {args.model}  dtype={args.dtype}  lookahead={args.lookahead} tokens")
    print(f"Index: granularity={md.granularity} clusters={md.number_of_clusters} stored_top_k={md.stored_top_k}")

    print("\nLoading chunked model ...")
    model, blocks, shard_index, stats = load_chunked_model(args.model, model_dir, dtype=dtype)

    num_experts_per_layer = blocks[0].num_experts
    top_k = blocks[0].top_k
    bytes_per_expert = None  # filled in after first materialize

    # ---- Correctness: same rigorous same-process check as Round 4, but now
    # through the streaming (KV-cached, one-token-at-a-time) code path.
    # Only loads a second, full reference model if actually asked to check
    # something - for models too large to load that way at all (30B-class,
    # see README), pass --num-correctness-prompts 0 and rely on the
    # architecture-level correctness proof already established on smaller
    # models instead (see scripts/chunked_inference_experiment.py). ----
    n_correct = min(args.num_correctness_prompts, traces.num_prompts)
    if n_correct > 0:
        print(f"\n=== Correctness ({n_correct} prompts, streaming replay vs. a fresh reference) ===")
        ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cpu")
        ref_model.eval()

        total_tokens_checked = exact_matches = near_tie_matches = real_mismatches = 0
        for i in range(n_correct):
            prompt_id = str(traces.prompt_ids[i])
            text = prompts[prompt_id]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            mask = traces.mask_for_prompt(i)
            fp = all_fingerprints[mask]
            predicted = np.stack([predictor.predict(fp[t]).top_experts_per_layer for t in range(fp.shape[0])])

            for b in blocks:
                b.evict_all()
            result = run_streaming(model, blocks, inputs["input_ids"], "eai_predict", predicted, args.lru_budget, args.lookahead)

            with torch.no_grad():
                ref_out = ref_model(**inputs, output_router_logits=True)

            num_tok = inputs["input_ids"].shape[1]
            for layer_idx in range(len(blocks)):
                ref_probs = torch.softmax(ref_out.router_logits[layer_idx].float(), dim=-1)
                _, ref_idx = torch.topk(ref_probs, top_k, dim=-1)
                for t in range(num_tok):
                    ref_set = set(ref_idx[t].tolist())
                    our_set = set(result["selected_per_step"][t, layer_idx].tolist())
                    diff = len(ref_set.symmetric_difference(our_set))
                    total_tokens_checked += 1
                    if diff == 0:
                        exact_matches += 1
                    elif diff == 2:
                        near_tie_matches += 1
                    else:
                        real_mismatches += 1
            print(f"  [{i+1}/{n_correct}] {prompt_id:<16} tokens={num_tok}")

        print(
            f"\nCorrectness over {total_tokens_checked} (token, layer) pairs:\n"
            f"  exact match: {exact_matches} ({100*exact_matches/total_tokens_checked:.1f}%)\n"
            f"  near-tie (bf16 boundary rounding, see README): {near_tie_matches} ({100*near_tie_matches/total_tokens_checked:.1f}%)\n"
            f"  real mismatch: {real_mismatches} ({100*real_mismatches/total_tokens_checked:.1f}%)"
        )
        if real_mismatches > 0:
            print("  ** REAL MISMATCHES - streaming replay diverges from a single batched forward pass, investigate **")

        del ref_model
        gc.collect()
    else:
        print(
            "\n=== Correctness: skipped (--num-correctness-prompts 0) - "
            "relying on the architecture-level proof already established on smaller models ==="
        )

    # ---- Memory comparison: three policies, same prompts ----
    n_mem = min(args.num_prompts, traces.num_prompts)
    print(f"\n=== Memory comparison ({n_mem} prompts, three eviction policies) ===")

    policy_results = {}
    for policy in ["no_eviction", "lru", "eai_predict"]:
        peaks_mb = []
        means_mb = []
        for b in blocks:
            b.evict_all()
        stats.hits = stats.misses = stats.evictions = stats.prefetched = 0
        stats.bytes_loaded = 0
        stats.fetch_seconds = 0.0

        for i in range(n_mem):
            prompt_id = str(traces.prompt_ids[i])
            text = prompts[prompt_id]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            mask = traces.mask_for_prompt(i)
            fp = all_fingerprints[mask]
            predicted = np.stack([predictor.predict(fp[t]).top_experts_per_layer for t in range(fp.shape[0])])

            for b in blocks:
                b.evict_all()
            shard_index.recycle()  # release OS-level mmap pages from evicted tensors - see ExpertShardIndex.recycle; no-op-ish for no_eviction (nothing's evicted to release), needed for lru/eai_predict on large checkpoints
            result = run_streaming(model, blocks, inputs["input_ids"], policy, predicted, args.lru_budget, args.lookahead)
            peaks_mb.append(result["peak_resident_bytes"] / 1e6)
            means_mb.append(result["mean_resident_bytes"] / 1e6)

        s = stats.as_dict()
        policy_results[policy] = {
            "peak_mb_mean": float(np.mean(peaks_mb)),
            "peak_mb_max": float(np.max(peaks_mb)),
            "mean_mb_mean": float(np.mean(means_mb)),
            "hit_rate": s["hit_rate"],
            "evictions": s["evictions"],
            "bytes_loaded_mb": s["bytes_loaded"] / 1e6,
            "fetch_ms_per_prompt": 1000 * s["fetch_seconds"] / n_mem,
        }
        print(
            f"  {policy:<12} peak resident: mean={np.mean(peaks_mb):8.1f} MB  max={np.max(peaks_mb):8.1f} MB   "
            f"hit_rate={100*s['hit_rate']:5.1f}%   evictions={s['evictions']:5d}"
        )
        print(
            f"               bytes_loaded_from_disk={s['bytes_loaded']/1e6:8.1f} MB total   "
            f"fetch_time={s['fetch_seconds']*1000:8.1f} ms total "
            f"({1000*s['fetch_seconds']/n_mem:.1f} ms/prompt)"
        )

    # Bytes/expert computed from an actually-resident expert, not a hardcoded
    # constant - a fixed MB/expert figure from one model (e.g. OLMoE's
    # ~12.58MB) is architecture-specific (depends on hidden_size and
    # moe_intermediate_size) and silently wrong for any other model.
    some_block_with_residents = next((b for b in blocks if b.resident_experts), None)
    if some_block_with_residents is None:
        raise RuntimeError("no block has any resident experts after the memory comparison loop - can't measure bytes/expert")
    bytes_per_expert = some_block_with_residents.resident_bytes / len(some_block_with_residents.resident_experts)
    full_model_expert_mb = num_experts_per_layer * len(blocks) * bytes_per_expert / 1e6
    print(f"\nFull model's expert weights (reference, always-resident baseline): {full_model_expert_mb:.0f} MB "
          f"({bytes_per_expert/1e6:.2f} MB/expert x {num_experts_per_layer} experts x {len(blocks)} layers)")
    for policy, r in policy_results.items():
        print(f"  {policy:<12}: {100*r['peak_mb_mean']/full_model_expert_mb:.1f}% of full model, peak resident, per prompt")


if __name__ == "__main__":
    main()
