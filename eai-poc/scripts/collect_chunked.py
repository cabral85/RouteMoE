#!/usr/bin/env python
"""Round 6: collect router ground truth for a model too large to load via
the normal eai.tracing.load_model path (accelerate/bitsandbytes) at all -
using eai.chunked_expert_loader instead, which never needs the full model
resident. This is what actually makes tracing a 30B-class model possible on
this machine (see README, "Attempting a 30B-class model").

Same output format as scripts/collect.py (artifacts/traces_{split}.npz),
so build_index.py and evaluate.py work unmodified on whatever this produces.

Memory is kept bounded by evicting each block's cache between prompts
(reactive-only: no EAI index exists yet for this model - that's what this
script's output builds) - same pattern proven safe in
scripts/streaming_eviction_experiment.py.

    python scripts/collect_chunked.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --out-dir artifacts/qwen3_30b
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.chunked_expert_loader import load_chunked_model
from eai.tracing import ModelHandle, PromptTrace, TraceSet, load_prompts_jsonl, trace_prompt_chunked

CHECKPOINT_EVERY = 10


def find_snapshot_dir(model_id: str) -> str:
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = model_id.split("/")
    pattern = os.path.join(cache_home, "hub", f"models--{org}--{name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no cached snapshot found for {model_id} under {pattern}")
    return matches[0]


def collect_split(model, blocks, shard_index, tokenizer, input_device, handle: ModelHandle,
                   prompts: list[dict], fingerprint_layer: int, split_name: str, out_path: Path) -> TraceSet:
    """Collects `prompts` into `out_path`, checkpointing every
    CHECKPOINT_EVERY prompts and resuming from whatever's already saved
    there. This collection run has twice been interrupted partway through by
    something external (no Python exception, no OOM, no watchdog kill - see
    README) that isn't fully root-caused; checkpointing means an
    interruption costs at most CHECKPOINT_EVERY prompts of lost work instead
    of the whole split, and reruns pick up where they left off automatically.
    """
    base_set: TraceSet | None = None
    already_done: set[str] = set()
    if out_path.exists():
        base_set = TraceSet.load(str(out_path))
        already_done = set(str(pid) for pid in base_set.prompt_ids)
        print(f"  [{split_name}] resuming: {len(already_done)} prompts already in {out_path}")

    remaining = [p for p in prompts if p["id"] not in already_done]
    if not remaining:
        print(f"  [{split_name}] nothing left to do - all {len(prompts)} prompts already collected")
        return base_set

    traces: list[PromptTrace] = []
    t_start = time.perf_counter()
    for i, p in enumerate(remaining, 1):
        for b in blocks:
            b.evict_all()  # keep our own cache bounded - reactive-only, no prediction exists yet to prefetch from
        shard_index.recycle()  # also release the OS-level mmap pages those evicted tensors were read from - see ExpertShardIndex.recycle
        tr = trace_prompt_chunked(
            model, blocks, tokenizer, input_device, p["id"], p["category"], p["text"],
            top_k=blocks[0].top_k, fingerprint_layer=fingerprint_layer,
        )
        traces.append(tr)
        print(
            f"  [{split_name}] {len(already_done) + i}/{len(prompts)} {p['id']:<20} cat={p['category']:<14} "
            f"tokens={tr.num_tokens:<4} fwd={tr.forward_pass_seconds * 1000:.0f}ms",
            flush=True,
        )
        if i % CHECKPOINT_EVERY == 0 or i == len(remaining):
            new_set = TraceSet.from_traces(traces, fingerprint_layer=fingerprint_layer, handle=handle)
            combined = TraceSet.concat([base_set, new_set]) if base_set is not None else new_set
            combined.save(str(out_path))
            print(f"  [{split_name}] checkpoint: {combined.num_prompts}/{len(prompts)} saved to {out_path}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"  [{split_name}] done: {len(traces)} new prompts in {elapsed:.1f}s ({elapsed/len(traces):.1f}s/prompt avg)")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", default="data/prompts_train.jsonl")
    parser.add_argument("--test", default="data/prompts_test.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fingerprint-layer", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--max-train", type=int, default=None, help="cap train prompts (for a quick first pass)")
    parser.add_argument("--max-test", type=int, default=None, help="cap test prompts (for a quick first pass)")
    parser.add_argument("--skip-train", action="store_true", help="don't re-collect train (e.g. an interrupted run already saved it)")
    parser.add_argument("--skip-test", action="store_true", help="don't collect test")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = find_snapshot_dir(args.model)
    print(f"Loading chunked model {args.model} from {model_dir} ...")
    t0 = time.perf_counter()
    model, blocks, shard_index, stats = load_chunked_model(args.model, model_dir, dtype=dtype)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s (backbone only - experts load on demand per prompt)")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_device = next(model.parameters()).device

    handle = ModelHandle(
        model=model, tokenizer=tokenizer, model_id=args.model, architecture=type(model).__name__,
        num_layers=len(blocks), num_experts=blocks[0].num_experts, top_k=blocks[0].top_k,
        hidden_size=config.hidden_size, quantization="none",
    )
    print(
        f"Model: layers={handle.num_layers} experts/layer={handle.num_experts} "
        f"top_k={handle.top_k} dtype={args.dtype}"
    )

    train_prompts = load_prompts_jsonl(args.train)[: args.max_train]
    test_prompts = load_prompts_jsonl(args.test)[: args.max_test]
    print(f"Train prompts: {len(train_prompts)}  Test prompts: {len(test_prompts)}")

    if not args.skip_train:
        train_out = out_dir / "traces_train.npz"
        collect_split(model, blocks, shard_index, tokenizer, input_device, handle, train_prompts, args.fingerprint_layer, "train", train_out)
        print(f"Saved {train_out}")
    else:
        print("Skipping train (--skip-train)")

    if not args.skip_test:
        test_out = out_dir / "traces_test.npz"
        collect_split(model, blocks, shard_index, tokenizer, input_device, handle, test_prompts, args.fingerprint_layer, "test", test_out)
        print(f"Saved {test_out}")
    else:
        print("Skipping test (--skip-test)")


if __name__ == "__main__":
    main()
