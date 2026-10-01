#!/usr/bin/env python
"""Round 7: rigorous throughput/I-O/stall benchmark comparing expert-cache
policies under a FIXED memory budget - reactive, lru, lfu, eai (+lookahead),
oracle. Does not touch the router: every policy sees the exact same
deterministic generation (same tokens, same real router decisions per
token/layer); only *when* expert weights are resident in memory differs.

Two-stage design, because a fair Oracle requires it:

  Stage 1 - trace: generate greedily (temperature=0, fixed seed) ONCE per
  prompt, with a generous (never-evicting) cache, recording the exact
  token sequence produced and the REAL expert selected by the REAL router
  at every (generated token, layer). This is the ground truth every policy
  below is compared against - and what makes Oracle honest: it "knows the
  future" only because that future was already fixed by an earlier,
  untouched run, not because it can see forward during its own pass.

  Stage 2 - replay: for each (policy, cache_budget) combination, rebuild a
  fresh chunked model, attach a `GlobalExpertCache` under that policy and
  budget, and feed the EXACT SAME token sequence from stage 1 through it
  one token at a time (real KV cache, real forward passes - not a
  simulation). Every policy does identical compute on identical inputs;
  only cache/prefetch decisions - and therefore timing, I/O, and memory -
  differ. That is what makes the comparison fair.

    python scripts/benchmark_streaming.py \\
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --index artifacts/qwen3_30b/model_token128.eai \\
        --cache-gb 8 \\
        --policies reactive,lru,lfu,eai,eai_lookahead_1,eai_lookahead_2,eai_lookahead_4,oracle
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.chunked_expert_loader import load_chunked_model
from eai.expert_cache import BenchmarkStats, GlobalExpertCache
from eai.fingerprint import compute_fingerprints
from eai.predictor import Predictor
from eai.storage import load_index
from eai.tracing import load_prompts_jsonl
from eai.workloads import WORKLOAD_NAMES, build_workload

POLICY_CHOICES = [
    "reactive", "lru", "lfu",
    "eai", "eai_lookahead_1", "eai_lookahead_2", "eai_lookahead_4", "eai_lookahead_8",
    "eai_coactivation", "eai_coactivation_lfu",
    "hybrid", "hybrid_adaptive",
    "oracle", "oracle_1", "oracle_2", "oracle_4", "oracle_8",
]
# oracle_N: N = how many future steps' real (ground-truth) expert selections
# the policy is allowed to see and prefetch, answering "LFU beat Oracle
# because prediction is useless, or because Oracle-1's horizon is too
# short?" (docs/benchmark_findings_current.md). Plain "oracle" is kept as an
# alias for oracle_1 - preserves every prior sweep's policy name/meaning
# unchanged (see docs/benchmark_findings_2026-09-19.md,
# docs/benchmark_findings_current.md's existing "oracle" rows). Oracle
# remains benchmark-only regardless of N: it only ever changes cache/
# prefetch decisions, never what the router computes or what tokens get
# generated - same guarantee as every other policy here.
# eai_coactivation_lfu: same coactivation-based prefetch as eai_coactivation,
# but LFU eviction instead of LRU underneath - motivated by a real finding
# (docs/benchmark_findings_current.md §2): in a warm, multi-prompt cache,
# plain lfu beat a perfect Oracle because Oracle's own prefetching evicts
# things a frequency count would have protected across prompts. This tests
# whether giving a PREDICTOR's prefetch that same cross-prompt memory (via
# its eviction rule, not the prediction itself) recovers that advantage.

# Single source of truth for both stage 1 (generate_and_trace) and stage 2
# (replay_policy)'s prefill chunking - they MUST match exactly, since bf16
# matmul isn't associative and a mismatched chunking between the two stages
# can flip a close top-k router pick (confirmed: caused real
# router-correctness-gate failures on Qwen3-30B before this was unified).
PREFILL_CHUNK_SIZE = 8


def find_snapshot_dir(model_id: str) -> str:
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = model_id.split("/")
    pattern = os.path.join(cache_home, "hub", f"models--{org}--{name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no cached snapshot found for {model_id} under {pattern}")
    return matches[0]


def tensor_name_fn(layer_idx: int, expert_idx: int, proj: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}_proj.weight"


def require_memory_headroom(min_free_gb: float, context: str) -> None:
    """Refuse to proceed if system RAM is already too tight - checked before
    the model load AND before every prompt/policy iteration, since a long
    sweep can run for a while and other processes on the machine (a
    concurrent Docker workload, say) can eat into what was free when the run
    started. Fails loud and immediately rather than letting RSS climb
    uncontrolled into a full-system OOM - the exact incident this guard
    exists to prevent."""
    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < min_free_gb:
        raise SystemExit(
            f"\nABORTING before {context}: only {available_gb:.1f}GB RAM free, below the "
            f"--min-free-ram-gb={min_free_gb} safety floor. This machine has other active "
            f"workloads (or a previous run left something resident) - free up memory or "
            f"lower --min-free-ram-gb only if you're sure it's safe, then retry."
        )


# ---------------------------------------------------------------------------
# Stage 1: deterministic generation + ground-truth trace
# ---------------------------------------------------------------------------

@dataclass
class GenerationTrace:
    prompt_id: str
    prompt_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]  # the new tokens actually generated, greedy
    selected_experts: np.ndarray  # (num_steps, num_layers, top_k) int32 - real router choice per generated token
    fingerprints: np.ndarray  # (num_steps, hidden_dim) float32 - fingerprint_layer hidden state per generated token, causal (each token's own state, not looking further ahead than itself)
    prefill_seconds: float  # time for the prompt forward pass alone (informs TTFT)


def generate_and_trace(
    model, blocks, tokenizer, prompt_id: str, category: str, text: str,
    max_new_tokens: int, fingerprint_layer: int, top_k: int,
    max_resident_per_layer: int = 24, prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
) -> GenerationTrace:
    """One deterministic greedy generation (temperature=0 == argmax, no
    sampling randomness - reproducible given fixed weights and inputs).

    Cache eviction here is IRRELEVANT to correctness: `selected_experts`/
    `fingerprints` come straight from the real router's own computation each
    step, which never depends on what happens to be cached (validated
    extensively - see docs/benchmark_findings_2026-09-19.md). The
    per-block dict cache was originally left fully unbounded ("generous,
    never-evicting") purely as an optimization to avoid redundant re-reads
    within one prompt - but that assumption breaks for a large model: a
    single Qwen3-30B prefill (before generating a single token) was measured
    touching 2013 distinct experts and pulling 19GB resident, growing to
    31GB by the 6th generated token - real, reproducible, twice, not a
    timing fluke.

    `evict_lru(max_resident_per_layer)` after every forward pass bounds the
    GENERATION loop (confirmed: resident_bytes flat at 10.87GB across 6
    steps, vs. unboundedly growing before). But a single-shot prefill is ONE
    forward call covering the WHOLE prompt at once - eviction can only run
    AFTER it returns, so a long enough prompt still spikes proportionally to
    its own length before eviction ever gets a chance to act (confirmed: a
    27-word prompt alone dropped free RAM from ~38GB to ~7.6GB - worse than
    a 10-word prompt's ~14GB). Fix: chunk the prefill into
    `prefill_chunk_size`-token pieces through the same growing KV cache,
    evicting between chunks - the same technique real inference engines call
    "chunked prefill". Only the LAST chunk's output is used (matches the
    original single-shot behavior, which only ever read the final position).
    """
    from transformers import DynamicCache

    for b in blocks:
        b.evict_all()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    prompt_ids = inputs["input_ids"][0].tolist()
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    cache = DynamicCache(config=model.config)
    t0 = time.perf_counter()
    out = None
    for chunk_start in range(0, input_ids.shape[1], prefill_chunk_size):
        chunk = input_ids[:, chunk_start : chunk_start + prefill_chunk_size]
        with torch.no_grad():
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        for b in blocks:
            b.evict_lru(max_resident_per_layer)
    prefill_seconds = time.perf_counter() - t0

    generated_ids: list[int] = []
    selected_per_step: list[np.ndarray] = []
    fingerprints: list[np.ndarray] = []

    next_id = out.logits[0, -1].argmax().item()
    fp = out.hidden_states[fingerprint_layer][0, -1].float().cpu().numpy()

    for _ in range(max_new_tokens):
        generated_ids.append(next_id)
        selected_per_step.append(np.stack([b.last_selected[-1].cpu().numpy() for b in blocks]))
        fingerprints.append(fp)

        if next_id == tokenizer.eos_token_id:
            break

        next_input = torch.tensor([[next_id]], device=device)
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=cache, use_cache=True, output_hidden_states=True)
        for b in blocks:
            b.evict_lru(max_resident_per_layer)
        next_id = out.logits[0, -1].argmax().item()
        fp = out.hidden_states[fingerprint_layer][0, -1].float().cpu().numpy()

    return GenerationTrace(
        prompt_id=prompt_id,
        prompt_text=text,
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        selected_experts=np.stack(selected_per_step) if selected_per_step else np.empty((0, len(blocks), top_k), dtype=np.int32),
        fingerprints=np.stack(fingerprints) if fingerprints else np.empty((0, model.config.hidden_size), dtype=np.float32),
        prefill_seconds=prefill_seconds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True, help="path to a .eai index (token granularity recommended)")
    parser.add_argument("--prompts", default="data/prompts_test.jsonl")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--workload", default="file_order", choices=["file_order", *WORKLOAD_NAMES], help="ordering applied to the selected prompts before tracing - file_order keeps the file's own order (previous default behavior)")
    parser.add_argument("--workload-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fingerprint-layer", type=int, default=2)
    parser.add_argument("--cache-gb", type=float, default=4.0, help="single budget to run (use a shell loop for a sweep)")
    parser.add_argument("--policies", default="reactive,lru,eai")
    parser.add_argument("--lru-budget-experts", type=int, default=None, help="unused - LRU now sized by --cache-gb like every other policy")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out", default="artifacts/benchmark/events.jsonl")
    parser.add_argument("--min-free-ram-gb", type=float, default=4.0, help="abort rather than proceed if free system RAM drops below this, checked before model load and before every prompt/policy iteration")
    parser.add_argument("--cache-scenario", default="cold", choices=["cold", "warm"], help="cold: cache evicted before every prompt (default). warm: one cache per policy, residency carries over across prompts in the run (simulates an already-serving cache) - only finalized (pending prefetches resolved) after the LAST prompt.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="where the backbone and streamed experts live - cuda makes --cache-gb a VRAM budget instead of a RAM one (see eai/expert_cache.py's device parameter). Backbone load and stage-1 trace generation still use host RAM as a staging area regardless (disk -> host -> device), so --min-free-ram-gb still applies.")
    parser.add_argument("--alpha", type=float, default=1.0, help="hybrid policy: weight on predicted_probability")
    parser.add_argument("--beta", type=float, default=1.0, help="hybrid policy: weight on coactivation_score")
    parser.add_argument("--gamma", type=float, default=1.0, help="hybrid policy: weight on normalized access frequency")
    parser.add_argument("--delta", type=float, default=1.0, help="hybrid policy: weight on normalized recency")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0, help="hybrid policy: weight SUBTRACTED for normalized load cost (bigger experts score lower, all else equal)")
    parser.add_argument("--admission-control", action="store_true", help="gate every prefetch through GlobalExpertCache.should_prefetch() (benefit vs. cost) instead of admitting unconditionally - see eai/expert_cache.py")
    parser.add_argument("--admission-margin", type=float, default=1.0, help="multiplies admission control's cost side - >1.0 stricter (fewer prefetches), <1.0 looser")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is available in this environment")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    budget_bytes = int(args.cache_gb * 1e9)
    policies = args.policies.split(",")
    for p in policies:
        if p not in POLICY_CHOICES:
            raise ValueError(f"unknown policy {p!r} (choices: {POLICY_CHOICES})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    require_memory_headroom(args.min_free_ram_gb, "model load")
    model_dir = find_snapshot_dir(args.model)
    index = load_index(args.index)
    predictor = Predictor(index)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_prompts = load_prompts_jsonl(args.prompts)
    if args.workload == "file_order":
        prompts = all_prompts[: args.num_prompts]
    else:
        # order the FULL prompt set first, then take the first num_prompts of
        # THAT ordering - slicing before ordering would silently defeat most
        # workloads (data/prompts_test.jsonl is grouped by category, so a
        # small file-order slice may contain only one category to begin with)
        prompts = build_workload(args.workload, all_prompts, seed=args.workload_seed)[: args.num_prompts]

    print(f"Model: {args.model}  dtype={args.dtype}  cache_budget={args.cache_gb}GB ({budget_bytes} bytes)")
    print(f"Policies: {policies}")
    print(f"Prompts: {len(prompts)}  max_new_tokens={args.max_new_tokens}")

    print(f"\nLoading chunked model on device={args.device} (for stage 1: deterministic trace generation)...")
    model, blocks, shard_index, load_stats = load_chunked_model(args.model, model_dir, dtype=dtype, device=args.device)
    top_k = blocks[0].top_k

    print("\n=== Stage 1: deterministic generation + ground-truth trace ===")
    traces: dict[str, GenerationTrace] = {}
    for i, p in enumerate(prompts, 1):
        require_memory_headroom(args.min_free_ram_gb, f"stage 1 prompt {p['id']!r}")
        for b in blocks:
            b.evict_all()
        shard_index.recycle()
        t0 = time.perf_counter()
        trace = generate_and_trace(
            model, blocks, tokenizer, p["id"], p["category"], p["text"],
            args.max_new_tokens, args.fingerprint_layer, top_k,
        )
        elapsed = time.perf_counter() - t0
        traces[p["id"]] = trace
        print(f"  [{i}/{len(prompts)}] {p['id']:<20} generated {len(trace.generated_token_ids)} tokens in {elapsed:.1f}s")

    print(f"\n=== Stage 2: replay under each policy, cache_budget={args.cache_gb}GB, scenario={args.cache_scenario} ===")
    events = []
    for policy in policies:
        if policy == "eai_coactivation_lfu":
            cache_policy = "lfu"
        elif policy in ("hybrid", "hybrid_adaptive"):
            cache_policy = "hybrid"
        elif policy.startswith("eai") or policy == "oracle" or policy.startswith("oracle_"):
            cache_policy = "lru"
        else:
            cache_policy = policy
        cache = None  # (re)built fresh per prompt in "cold"; built once and reused in "warm"

        for prompt_idx, p in enumerate(prompts):
            require_memory_headroom(args.min_free_ram_gb, f"stage 2 policy={policy!r} prompt={p['id']!r}")
            trace = traces[p["id"]]
            is_last_prompt = prompt_idx == len(prompts) - 1

            if args.cache_scenario == "cold" or cache is None:
                for b in blocks:
                    b.evict_all()
                shard_index.recycle()
                bench_stats = BenchmarkStats()
                cache = GlobalExpertCache(
                    shard_index=shard_index, budget_bytes=budget_bytes, policy=cache_policy,
                    dtype=dtype, stats=bench_stats, tensor_name_fn=tensor_name_fn, device=args.device,
                    hybrid_alpha=args.alpha, hybrid_beta=args.beta, hybrid_gamma=args.gamma,
                    hybrid_delta=args.delta, hybrid_lambda=args.lambda_,
                    admission_control=args.admission_control, admission_margin=args.admission_margin,
                )
                for b in blocks:
                    b.attach_global_cache(cache)
            else:
                # warm: same cache object, residency carries over - only the
                # per-prompt stats counter is reset, so each event still
                # reports THIS prompt's own numbers, not a running total.
                bench_stats = BenchmarkStats()
                cache.stats = bench_stats

            result = replay_policy(
                model, blocks, tokenizer, trace, policy, predictor, index, cache, bench_stats,
                cache_scenario=args.cache_scenario,
            )
            if args.cache_scenario == "cold" or is_last_prompt:
                cache.finalize()

            event = {
                "run_id": str(uuid.uuid4()),
                "prompt_id": p["id"],
                "policy": policy,
                "cache_budget_bytes": budget_bytes,
                "workload": args.workload,
                "cache_scenario": args.cache_scenario,
                "device": args.device,
                "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
                "delta": args.delta, "lambda": args.lambda_,
                "admission_control": args.admission_control, "admission_margin": args.admission_margin,
                **result,
                **bench_stats.as_dict(),
            }
            events.append(event)
            print(
                f"  policy={policy:<18} prompt={p['id']:<20} "
                f"tok/s={result['tokens_per_second']:6.2f}  hit_rate={bench_stats.as_dict()['hit_rate']*100:5.1f}%  "
                f"io_wait_ms={result['expert_io_wait_ms']:8.1f}  peak_gb={bench_stats.peak_resident_bytes/1e9:.2f}"
            )

    with open(out_path, "a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print(f"\nAppended {len(events)} events to {out_path}")


def replay_policy(model, blocks, tokenizer, trace: GenerationTrace, policy: str,
                   predictor: Predictor, index, cache: GlobalExpertCache, bench_stats: BenchmarkStats,
                   cache_scenario: str = "cold") -> dict:
    """Feeds `trace`'s exact token sequence through `model` one token at a
    time (real KV cache), with `cache` (already configured for this policy's
    eviction rule and budget) governing expert residency. `policy` decides
    whether/what to prefetch before each step - `cache.get()` (called from
    inside each block's forward()) handles the reactive path uniformly for
    every policy, prefetch or not.

    Index alignment (easy to get backwards, so spelled out): stage 1 records
    `selected_experts[i]` / `fingerprints[i]` as the REAL router experts /
    hidden state produced by the forward pass that GENERATED
    `generated_token_ids[i]` - i.e. the forward pass whose INPUT was
    `generated_token_ids[i-1]` (or the prompt, for i=0). So when this
    function feeds `generated_token_ids[step]` in as the next input token,
    the ground truth for THAT forward pass is `selected_experts[step+1]` /
    `fingerprints[step+1]`, not `[step]`. Only `num_steps - 1` such forward
    passes have recorded ground truth - the last generated token's own
    forward pass was never captured in stage 1, since generation stopped
    right after producing it - so this replays exactly `num_steps - 1`
    generation steps, not `num_steps`.
    """
    from transformers import DynamicCache

    device = next(model.parameters()).device
    num_steps = len(trace.generated_token_ids)
    num_replay_steps = max(0, num_steps - 1)
    top_k = blocks[0].top_k

    lookahead = 0
    if policy.startswith("eai_lookahead_"):
        lookahead = int(policy.rsplit("_", 1)[1])
    elif policy.startswith("oracle_"):
        lookahead = int(policy.rsplit("_", 1)[1]) - 1  # oracle_1 == plain "oracle" == lookahead 0 (current step only)
    # plain "oracle" (no suffix): lookahead stays 0, same as oracle_1

    def predict_for_step(ground_truth_idx: int):
        """Full PredictionResult (cluster_id + (num_layers, stored_top_k)
        predicted experts) for `trace.fingerprints[ground_truth_idx]`, via
        the real EAI predictor - timed, since predictor latency is part of
        what "does the predictor pay for itself" asks."""
        t0 = time.perf_counter()
        result = predictor.predict(trace.fingerprints[ground_truth_idx])
        bench_stats.eai_lookup_seconds += time.perf_counter() - t0
        return result

    # ---- prefill: chunked, matching generate_and_trace()'s stage-1 prefill ----
    # Must use the SAME chunking as stage 1, not just "however is fastest
    # here": bf16 matmul is not associative, so a single-shot prefill and a
    # chunked one can land on very slightly different logits (same root
    # cause as this project's known bf16 top-k tie-breaking non-determinism)
    # - occasionally enough to flip a close top-k router pick. Confirmed by
    # hitting exactly this: 17/240 correctness-gate mismatches on a real
    # Qwen3-30B run before this fix, 0 after matching the chunking.
    input_ids = torch.tensor([trace.prompt_token_ids], device=device)
    kv_cache = DynamicCache(config=model.config)
    t0 = time.perf_counter()
    for chunk_start in range(0, input_ids.shape[1], PREFILL_CHUNK_SIZE):
        chunk = input_ids[:, chunk_start : chunk_start + PREFILL_CHUNK_SIZE]
        with torch.no_grad():
            model(input_ids=chunk, past_key_values=kv_cache, use_cache=True)
    ttft_seconds = time.perf_counter() - t0

    step_latencies = []
    correctness_exact = 0
    correctness_mismatch = 0

    for step in range(num_replay_steps):
        gt_idx = step + 1  # ground truth for the forward pass about to run

        if policy == "oracle" or policy.startswith("oracle_"):
            for layer_idx, block in enumerate(blocks):
                window_experts = set(trace.selected_experts[gt_idx, layer_idx].tolist())
                for la in range(1, lookahead + 1):
                    future_idx = gt_idx + la
                    if future_idx < num_steps:
                        window_experts.update(trace.selected_experts[future_idx, layer_idx].tolist())
                cache.prefetch(layer_idx, sorted(window_experts))
        elif policy in ("hybrid", "hybrid_adaptive"):
            pred_result = predict_for_step(gt_idx)
            predicted = pred_result.top_experts_per_layer
            for layer_idx in range(len(blocks)):
                top1_pred = int(predicted[layer_idx, 0]) if predicted.shape[1] > 0 else -1
                real_set = set(trace.selected_experts[gt_idx, layer_idx].tolist())
                bench_stats.predictor_top1_total += 1
                if top1_pred in real_set:
                    bench_stats.predictor_top1_hits += 1

            if policy == "hybrid_adaptive":
                avg_expert_bytes = cache.stats.storage_bytes_read / max(1, cache.stats.unique_experts_loaded)
                model_working_set_bytes = top_k * len(blocks) * avg_expert_bytes
                weights = hybrid_adaptive_weights(cache.budget_bytes, model_working_set_bytes, cache_scenario)
                cache.hybrid_alpha, cache.hybrid_beta = weights["alpha"], weights["beta"]
                cache.hybrid_gamma, cache.hybrid_delta = weights["gamma"], weights["delta"]
                cache.hybrid_lambda = weights["lam"]
                if step == 0:
                    print(f"    [hybrid_adaptive] regime={weights['_regime']} alpha={weights['alpha']:.2f} beta={weights['beta']:.2f} gamma={weights['gamma']:.2f} delta={weights['delta']:.2f} lambda={weights['lam']:.2f}")

            for layer_idx, block in enumerate(blocks):
                window, predicted_probabilities, coactivation_scores = hybrid_select_with_scores(
                    index, pred_result.cluster_id, layer_idx, pred_result.probabilities_per_layer[layer_idx], target_width=top_k,
                )
                cache.prefetch(layer_idx, window, predicted_probabilities=predicted_probabilities, coactivation_scores=coactivation_scores)
        elif policy.startswith("eai"):
            pred_result = predict_for_step(gt_idx)
            predicted = pred_result.top_experts_per_layer  # (num_layers, stored_top_k)
            for layer_idx in range(len(blocks)):
                top1_pred = int(predicted[layer_idx, 0]) if predicted.shape[1] > 0 else -1
                real_set = set(trace.selected_experts[gt_idx, layer_idx].tolist())
                bench_stats.predictor_top1_total += 1
                if top1_pred in real_set:
                    bench_stats.predictor_top1_hits += 1

            if policy.startswith("eai_coactivation"):
                for layer_idx, block in enumerate(blocks):
                    window = coactivation_select(index, pred_result.cluster_id, layer_idx, target_width=top_k)
                    cache.prefetch(layer_idx, sorted(window))
            else:
                for layer_idx, block in enumerate(blocks):
                    window = set(int(e) for e in predicted[layer_idx] if e >= 0)
                    for la in range(1, lookahead + 1):
                        future_idx = gt_idx + la
                        if future_idx < num_steps:
                            future_pred = predict_for_step(future_idx).top_experts_per_layer
                            window.update(int(e) for e in future_pred[layer_idx] if e >= 0)
                    cache.prefetch(layer_idx, sorted(window))
        # reactive / lru / lfu: no prefetch at all - cache.get() (inside forward()) handles everything reactively

        next_input = torch.tensor([[trace.generated_token_ids[step]]], device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=next_input, past_key_values=kv_cache, use_cache=True)
        step_latencies.append(time.perf_counter() - t0)

        # Correctness gate: the cache/prefetch machinery must never change
        # what the real router selects, only when weights were resident.
        # Verified with evidence every single replay, not assumed correct.
        for layer_idx, block in enumerate(blocks):
            real = set(trace.selected_experts[gt_idx, layer_idx].tolist())
            replayed = set(block.last_selected[-1].tolist())
            if replayed == real:
                correctness_exact += 1
            else:
                correctness_mismatch += 1

    if correctness_mismatch:
        raise RuntimeError(
            f"replay_policy correctness check failed for policy={policy!r}: "
            f"{correctness_mismatch} (step, layer) router selections during replay "
            f"disagreed with stage-1 ground truth (out of {correctness_exact + correctness_mismatch} "
            f"checked). The cache/prefetch layer must never change router decisions - "
            f"this points at a real bug, not benchmark noise."
        )

    total_gen_seconds = sum(step_latencies)
    output_tokens = num_replay_steps
    lat_ms = np.array(step_latencies) * 1000

    return {
        "ttft_ms": ttft_seconds * 1000,
        "total_generation_ms": total_gen_seconds * 1000,
        "tokens_per_second": output_tokens / max(1e-9, total_gen_seconds),
        "output_tokens": output_tokens,
        "tpot_ms_mean": float(lat_ms.mean()) if len(lat_ms) else 0.0,
        "tpot_ms_p50": float(np.percentile(lat_ms, 50)) if len(lat_ms) else 0.0,
        "tpot_ms_p95": float(np.percentile(lat_ms, 95)) if len(lat_ms) else 0.0,
        "tpot_ms_p99": float(np.percentile(lat_ms, 99)) if len(lat_ms) else 0.0,
        "expert_io_wait_ms": bench_stats.expert_io_wait_seconds * 1000,
        "eai_lookup_ms": bench_stats.eai_lookup_seconds * 1000,
        "storage_bytes_per_output_token": bench_stats.storage_bytes_read / max(1, output_tokens),
        "compute_ms": total_gen_seconds * 1000 - bench_stats.expert_io_wait_seconds * 1000,
        "router_correctness_checks": correctness_exact + correctness_mismatch,
    }


def coactivation_select(index, cluster_id: int, layer_idx: int, target_width: int) -> list[int]:
    """Fase 7: an alternative to `eai`'s plain "prefetch every stored_top_k
    candidate" window. That plain window (validated in Fase 2/3) prefetches
    every one of the stored_top_k=16 individually-most-frequent experts for
    a cluster/layer - which, per the invariant-validation run on OLMoE,
    thrashes the cache badly (most of those 16 are not what actually fires
    together on a given token, just each individually common across the
    whole cluster) and ends up net-worse than plain reactive caching.

    This selects a much narrower, `target_width`-sized set (normally the
    model's real per-layer top_k) via greedy clique-building on the stored
    (stored_top_k x stored_top_k) coactivation matrix: start from the
    cluster's single most-frequent expert (slot 0), then repeatedly add
    whichever remaining stored slot has the highest average pairwise
    coactivation with what's already chosen - i.e. the experts that fire
    TOGETHER most often with the current picks, not just the ones that are
    each individually common in isolation.
    """
    coact = index.coactivation[cluster_id, layer_idx]  # (stored_top_k, stored_top_k)
    slots = index.top_experts[cluster_id, layer_idx]  # (stored_top_k,) expert ids, -1 = unused
    valid = [s for s in range(len(slots)) if slots[s] >= 0]
    if not valid:
        return []

    target_width = min(target_width, len(valid))
    selected = [valid[0]]  # slot 0 = highest marginal frequency, by construction of build_profiles()
    remaining = [s for s in valid if s != valid[0]]
    while len(selected) < target_width and remaining:
        best = max(remaining, key=lambda s: float(np.mean([coact[s, t] for t in selected])))
        selected.append(best)
        remaining.remove(best)

    return [int(slots[s]) for s in selected]


def hybrid_select_with_scores(
    index, cluster_id: int, layer_idx: int, probabilities_row: np.ndarray, target_width: int,
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    """Same greedy-clique candidate selection as `coactivation_select`, but
    also returns the per-expert scores GlobalExpertCache's "hybrid" eviction
    rule needs: `predicted_probability` (straight from the predictor's own
    stored per-slot probability) and `coactivation_score` (mean pairwise
    coactivation with the rest of the selected clique - the same quantity
    the greedy selection itself optimizes for, exposed here instead of
    thrown away). Returns (window, predicted_probabilities, coactivation_scores).
    """
    coact = index.coactivation[cluster_id, layer_idx]
    slots = index.top_experts[cluster_id, layer_idx]
    valid = [s for s in range(len(slots)) if slots[s] >= 0]
    if not valid:
        return [], {}, {}

    target_width = min(target_width, len(valid))
    selected = [valid[0]]
    remaining = [s for s in valid if s != valid[0]]
    while len(selected) < target_width and remaining:
        best = max(remaining, key=lambda s: float(np.mean([coact[s, t] for t in selected])))
        selected.append(best)
        remaining.remove(best)

    window = [int(slots[s]) for s in selected]
    predicted_probabilities = {int(slots[s]): float(probabilities_row[s]) for s in selected}
    coactivation_scores = {
        int(slots[s]): float(np.mean([coact[s, t] for t in selected if t != s])) if len(selected) > 1 else 0.0
        for s in selected
    }
    return window, predicted_probabilities, coactivation_scores


def hybrid_adaptive_weights(cache_budget_bytes: int, model_working_set_bytes: float, cache_scenario: str) -> dict[str, float]:
    """Fase 5: heuristic (not learned) weight selection based on the current
    regime - explicit rules, not ML, per spec. `model_working_set_bytes` is
    an estimate of what ONE step's real need costs (top_k * num_layers *
    avg_expert_bytes) - the natural yardstick for "is this budget tiny,
    medium, or large" (a fixed GB number means something different on
    OLMoE vs. Qwen3-30B). Returns the regime name (for logging - "Registrar
    no log qual regime foi usado") alongside the weights, via the dict's
    own "_regime" key.
    """
    if model_working_set_bytes <= 0:
        ratio = 1.0  # no estimate yet (very first step) - treat as "medium" until real data exists
    else:
        ratio = cache_budget_bytes / model_working_set_bytes

    if ratio < 1.5:
        # tiny cache: barely fits one step's own working set - prioritize the
        # immediate future hard, don't waste budget on frequency bookkeeping
        # that a cache this size can't afford to honor anyway.
        weights = dict(alpha=3.0, beta=2.0, gamma=0.2, delta=0.2, lam=0.5, _regime="tiny_cache")
    elif ratio < 4.0:
        # medium cache: balance prediction against accumulated usage stats.
        weights = dict(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, lam=1.0, _regime="medium_cache")
    else:
        # large cache: most of what's needed already fits - aggressive
        # prefetch mostly just adds eviction churn (see docs/benchmark_findings_current.md
        # #2/#7's warm-cache findings) - let accumulated frequency dominate,
        # keep prediction as a light tiebreaker only.
        weights = dict(alpha=0.3, beta=0.3, gamma=2.0, delta=1.0, lam=1.0, _regime="large_cache")

    if cache_scenario == "warm":
        # a warm cache has real cross-prompt history to lean on - the whole
        # lesson of docs/benchmark_findings_current.md #2's lfu-beats-oracle
        # finding: weight frequency up further when there's session-long
        # signal to actually exploit.
        weights["gamma"] *= 1.5
        weights["_regime"] += "+warm"
    else:
        # cold: no cross-prompt history yet to lean on - lean on
        # prediction/coactivation instead, frequency/recency are still
        # forming.
        weights["alpha"] *= 1.3
        weights["beta"] *= 1.3
        weights["_regime"] += "+cold"

    return weights


if __name__ == "__main__":
    main()
