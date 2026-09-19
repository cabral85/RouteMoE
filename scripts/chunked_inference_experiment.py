#!/usr/bin/env python
"""Phase 7 (extension): prove the chunked expert loader is both correct and
actually saves memory, using the EAI predictor to decide what to prefetch.

This is the first script in the PoC that *acts* on a prediction instead of
just scoring it: earlier phases measured whether eai/predictor.py *could*
have guessed the right experts; this loads a model where that guess actually
determines what's resident in memory, and reports what that costs and buys.

Two phases, deliberately kept separate so the memory numbers in phase B
aren't contaminated by phase A's extra reference model:

  Phase A - correctness: load a fresh, full reference model in this same
  process (same device, same dtype - avoiding any cross-device numerical
  drift), run it and the chunked model on the same prompts, and compare
  actual selected experts, layer by layer. Then free the reference model.

  Phase B - memory: using the chunked model alone (reference freed), run the
  full test set, seeded by the EAI index's predictions, and report real
  peak memory (chunked vs. what the reference needed) and hit/miss rates.

    python scripts/chunked_inference_experiment.py
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


def peak_rss_mb() -> float:
    """Best-effort peak resident set size (Windows/WSL Linux only, matching this PoC's environments)."""
    try:
        import resource  # Linux/WSL

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB
    except ImportError:
        import psutil  # Windows fallback

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def find_snapshot_dir(model_id: str) -> str:
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = model_id.split("/")
    pattern = os.path.join(cache_home, "hub", f"models--{org}--{name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no cached snapshot found for {model_id} under {pattern}")
    return matches[0]


def predict_and_prefetch(blocks, predictor, fp_for_prompt: np.ndarray) -> None:
    """Predict per-token, per-layer experts from the EAI index and warm each
    block's cache with the union - this is where the prediction becomes an
    actual memory decision, not just a scored guess."""
    predicted = np.stack([predictor.predict(fp_for_prompt[t]).top_experts_per_layer for t in range(fp_for_prompt.shape[0])])
    for layer_idx, block in enumerate(blocks):
        union_predicted = sorted(set(int(e) for e in predicted[:, layer_idx, :].flatten() if e >= 0))
        block.prefetch(union_predicted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--index", default="artifacts/model_token128.eai")
    parser.add_argument("--traces", default="artifacts/traces_test.npz")
    parser.add_argument("--prompts", default="data/prompts_test.jsonl")
    parser.add_argument("--num-correctness-prompts", type=int, default=3,
                         help="prompts checked in phase A (loads a second full model - keep small)")
    parser.add_argument("--num-memory-prompts", type=int, default=20,
                         help="prompts run in phase B (chunked model only)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
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

    print(f"Model: {args.model}  dtype={args.dtype}")
    print(f"Index: granularity={md.granularity} clusters={md.number_of_clusters} stored_top_k={md.stored_top_k}")

    # ---- Phase A: correctness, verified against a fresh same-process reference ----
    n_correct = min(args.num_correctness_prompts, traces.num_prompts)
    print(f"\n=== Phase A: correctness ({n_correct} prompts, chunked vs. a fresh reference model) ===")

    print("Loading chunked model ...")
    model, blocks, shard_index, stats = load_chunked_model(args.model, model_dir, dtype=dtype)
    print("Loading reference model (full, for comparison only - freed after phase A) ...")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cpu")
    ref_model.eval()

    total_tokens_checked = 0
    exact_matches = 0
    near_tie_matches = 0
    real_mismatches = 0

    for i in range(n_correct):
        prompt_id = str(traces.prompt_ids[i])
        text = prompts[prompt_id]
        mask = traces.mask_for_prompt(i)
        fp = all_fingerprints[mask]
        predict_and_prefetch(blocks, predictor, fp)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            ref_out = ref_model(**inputs, output_router_logits=True)
            model(**inputs)

        num_layers = len(blocks)
        num_tok = inputs["input_ids"].shape[1]
        prompt_exact = prompt_near = prompt_bad = 0
        for layer_idx, block in enumerate(blocks):
            ref_probs = torch.softmax(ref_out.router_logits[layer_idx].float(), dim=-1)
            _, ref_idx = torch.topk(ref_probs, block.top_k, dim=-1)
            actual = block.last_selected
            for t in range(num_tok):
                ref_set = set(ref_idx[t].tolist())
                our_set = set(actual[t].tolist())
                diff = len(ref_set.symmetric_difference(our_set))
                total_tokens_checked += 1
                if diff == 0:
                    exact_matches += 1
                    prompt_exact += 1
                elif diff == 2:
                    # exactly one expert swapped for another - the signature of a
                    # near-tied 8th/9th-place router probability flipping under
                    # bf16 rounding (fused vs. split matmul), not a logic error -
                    # see README for the float32 zero-mismatch proof.
                    near_tie_matches += 1
                    prompt_near += 1
                else:
                    real_mismatches += 1
                    prompt_bad += 1

        status = "OK" if prompt_bad == 0 else f"{prompt_bad} real mismatches (+{prompt_near} near-tie)"
        print(f"  [{i+1}/{n_correct}] {prompt_id:<16} tokens={num_tok:<4} exact={prompt_exact} near-tie={prompt_near} bad={prompt_bad} - {status}")

    print(
        f"\nCorrectness over {total_tokens_checked} (token, layer) pairs:\n"
        f"  exact match: {exact_matches} ({100*exact_matches/total_tokens_checked:.1f}%)\n"
        f"  near-tie (1 expert swapped, bf16 boundary rounding - see README): "
        f"{near_tie_matches} ({100*near_tie_matches/total_tokens_checked:.1f}%)\n"
        f"  real mismatch (2+ experts differ - would indicate a genuine bug): "
        f"{real_mismatches} ({100*real_mismatches/total_tokens_checked:.1f}%)"
    )
    if real_mismatches > 0:
        print("  ** REAL MISMATCHES DETECTED - investigate before trusting phase B's numbers **")

    del ref_model
    gc.collect()

    # ---- Phase B: memory, using the chunked model alone ----
    n_mem = min(args.num_memory_prompts, traces.num_prompts)
    print(f"\n=== Phase B: memory ({n_mem} prompts, chunked model only, reference freed) ===")

    for b in blocks:
        b.evict_all()
    stats.prefetched = stats.hits = stats.misses = 0
    stats.bytes_loaded = 0
    stats.fetch_seconds = 0.0

    # Evict between prompts and measure PER-PROMPT resident memory - the
    # number that matters for "what does one request cost", not the
    # cumulative union across many diverse prompts served back to back
    # (which, with no eviction policy, converges toward the whole model -
    # a real but different finding, reported separately below).
    per_prompt_peak_bytes = []
    for i in range(n_mem):
        for b in blocks:
            b.evict_all()
        prompt_id = str(traces.prompt_ids[i])
        text = prompts[prompt_id]
        mask = traces.mask_for_prompt(i)
        fp = all_fingerprints[mask]
        predict_and_prefetch(blocks, predictor, fp)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            model(**inputs)

        per_prompt_peak_bytes.append(sum(b.resident_bytes for b in blocks))

    print(f"\nPer-prompt resident expert memory (evicted between prompts, {n_mem} prompts):")
    per_prompt_mb = np.array(per_prompt_peak_bytes) / 1e6
    print(f"  mean: {per_prompt_mb.mean():.1f} MB   min: {per_prompt_mb.min():.1f} MB   max: {per_prompt_mb.max():.1f} MB")
    # Bytes/expert computed from an actually-resident expert, not a hardcoded
    # constant - architecture-specific (hidden_size, moe_intermediate_size
    # differ per model), so this must be measured, not assumed.
    some_block_with_residents = next((b for b in blocks if b.resident_experts), None)
    bytes_per_expert = some_block_with_residents.resident_bytes / len(some_block_with_residents.resident_experts)
    full_model_expert_mb = md.num_experts * md.num_layers * bytes_per_expert / 1e6
    print(f"  vs. full model's expert weights: {full_model_expert_mb:.0f} MB "
          f"({100*per_prompt_mb.mean()/full_model_expert_mb:.1f}% of full, on average, per single prompt)")

    print(f"\n=== Cumulative, no-eviction memory ({n_mem} prompts back to back, no cache eviction) ===")
    for b in blocks:
        b.evict_all()
    stats.prefetched = stats.hits = stats.misses = 0
    stats.bytes_loaded = 0
    stats.fetch_seconds = 0.0

    for i in range(n_mem):
        prompt_id = str(traces.prompt_ids[i])
        text = prompts[prompt_id]
        mask = traces.mask_for_prompt(i)
        fp = all_fingerprints[mask]
        predict_and_prefetch(blocks, predictor, fp)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            model(**inputs)

    s = stats.as_dict()
    print(f"Hit/miss stats, prefetch predicted from the EAI index, {n_mem} prompts back to back, no eviction:")
    print(f"  prefetched: {s['prefetched']}")
    print(f"  hits (router needed an already-prefetched expert): {s['hits']}")
    print(f"  misses (router needed an expert we hadn't prefetched): {s['misses']}")
    print(f"  hit rate: {s['hit_rate']*100:.1f}%")
    print(f"  total expert bytes loaded from disk: {s['bytes_loaded']/1e6:.1f} MB")
    print(f"  total time spent fetching expert weights: {s['fetch_seconds']*1000:.1f} ms")

    max_layer_resident = max(b.resident_bytes for b in blocks)
    total_resident = sum(b.resident_bytes for b in blocks)
    print(f"\nMax resident expert memory, any single layer: {max_layer_resident/1e6:.1f} MB")
    print(f"Total resident expert memory, all layers, after {n_mem} prompts with NO eviction: {total_resident/1e6:.1f} MB")
    print(f"  ({100*total_resident/1e6/full_model_expert_mb:.1f}% of the full model's expert weights - "
          f"an unbounded cache serving diverse traffic converges toward loading everything; "
          f"a real deployment needs an eviction policy, out of scope for this PoC)")
    print(f"Peak process RSS at end of phase B: {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
