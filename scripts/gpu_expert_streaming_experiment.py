#!/usr/bin/env python
"""First real GPU measurement for this project: everything up to now
(the whole OLMoE/Qwen3-30B sweep) ran entirely on CPU/system RAM - this
script is the first to actually place the backbone and streamed experts on
CUDA, to answer directly: what does expert cold-start cost when the target
is VRAM instead of RAM, and does a small (consumer laptop) GPU change the
"30B on a small machine" story at all?

Two things measured, both against a CPU baseline for correctness AND speed:
  1. Backbone load time - non-expert weights placed directly on the GPU via
     load_state_dict(assign=True) from a meta-device model (same mechanism
     already used for CPU, see eai/chunked_expert_loader.py).
  2. Per-expert cold-start cost when residency is VRAM: disk -> host memory
     (seek()+readinto(), same as CPU) -> one .to("cuda") transfer over PCIe,
     measured via GlobalExpertCache's own read_seconds_total (now this
     includes the PCIe copy, not just the disk read).

Correctness check: the GPU run's output tokens must exactly match the CPU
run's (same prompt, same greedy decoding) - moving WHERE weights live must
never change WHAT the model computes.

    python scripts/gpu_expert_streaming_experiment.py --model allenai/OLMoE-1B-7B-0924 --cache-gb 2.0
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import psutil
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.chunked_expert_loader import load_chunked_model
from eai.expert_cache import BenchmarkStats, GlobalExpertCache


def find_snapshot_dir(model_id: str) -> str:
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = model_id.split("/")
    pattern = os.path.join(cache_home, "hub", f"models--{org}--{name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no cached snapshot for {model_id} under {pattern}")
    return matches[0]


def tensor_name_fn(layer_idx: int, expert_idx: int, proj: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}_proj.weight"


def run(model_id: str, model_dir: str, device: str, budget_bytes: int, prompt: str, max_new_tokens: int, tokenizer):
    from transformers import DynamicCache

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    model, blocks, shard_index, _ = load_chunked_model(model_id, model_dir, dtype=torch.bfloat16, device=device)
    load_seconds = time.perf_counter() - t0

    stats = BenchmarkStats()
    cache = GlobalExpertCache(
        shard_index=shard_index, budget_bytes=budget_bytes, policy="lru",
        dtype=torch.bfloat16, stats=stats, tensor_name_fn=tensor_name_fn, device=device,
    )
    for b in blocks:
        b.attach_global_cache(cache)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    kv_cache = DynamicCache(config=model.config)

    proc = psutil.Process(os.getpid())
    cpu_rss_before = proc.memory_info().private

    generated = []
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=kv_cache, use_cache=True)
    ttft = time.perf_counter() - t0

    next_id = out.logits[0, -1].argmax().item()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            generated.append(next_id)
            next_input = torch.tensor([[next_id]], device=device)
            out = model(input_ids=next_input, past_key_values=kv_cache, use_cache=True)
            next_id = out.logits[0, -1].argmax().item()
    gen_seconds = time.perf_counter() - t0

    cpu_rss_after = proc.memory_info().private
    result = dict(
        device=device, load_seconds=load_seconds, ttft_seconds=ttft,
        tokens_per_second=max_new_tokens / gen_seconds, generated=generated,
        expert_io_wait_seconds=stats.expert_io_wait_seconds,
        storage_bytes_read=stats.storage_bytes_read, reads_count=stats.reads_count,
        avg_read_ms=stats.read_seconds_total / max(1, stats.reads_count) * 1000,
        cache_misses=stats.cache_misses, hit_rate=stats.as_dict()["hit_rate"],
        cpu_private_delta_gb=(cpu_rss_after - cpu_rss_before) / 1e9,
    )
    if device == "cuda":
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del model, blocks, cache, kv_cache
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--prompt", default="Escreva uma função Python que verifica se um número é primo.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-gb", type=float, default=2.0, help="expert cache budget, applied identically on both CPU and GPU runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device available in this environment")

    from transformers import AutoTokenizer

    model_dir = find_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    budget_bytes = int(args.cache_gb * 1e9)

    print(f"model={args.model}  cache_budget={args.cache_gb}GB  max_new_tokens={args.max_new_tokens}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB VRAM)")

    print("\n=== CPU baseline ===")
    cpu = run(args.model, model_dir, "cpu", budget_bytes, args.prompt, args.max_new_tokens, tokenizer)
    for k, v in cpu.items():
        if k != "generated":
            print(f"  {k}: {v}")

    print("\n=== GPU (CUDA) ===")
    gpu = run(args.model, model_dir, "cuda", budget_bytes, args.prompt, args.max_new_tokens, tokenizer)
    for k, v in gpu.items():
        if k != "generated":
            print(f"  {k}: {v}")

    match = cpu["generated"] == gpu["generated"]
    print(f"\n=== comparison ===")
    print(f"  identical output tokens: {match}  (must be True)")
    print(f"  tok/s:              cpu={cpu['tokens_per_second']:.2f}  gpu={gpu['tokens_per_second']:.2f}  ({gpu['tokens_per_second']/cpu['tokens_per_second']*100:.1f}% )")
    print(f"  avg read+transfer:  cpu={cpu['avg_read_ms']:.3f}ms  gpu={gpu['avg_read_ms']:.3f}ms")
    print(f"  peak VRAM (gpu run): {gpu.get('peak_vram_gb', float('nan')):.3f}GB")
    if not match:
        raise SystemExit("FAILED: GPU run produced different tokens than CPU - device placement must never change model output")


if __name__ == "__main__":
    main()
