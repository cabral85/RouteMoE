#!/usr/bin/env python
"""Phase 2: run the model over the train/test prompt sets and record router ground truth.

    python scripts/collect.py

Writes artifacts/traces_train.npz and artifacts/traces_test.npz. These are
transient/regenerable intermediate files (raw per-token traces) - not the
final .eai index, and not meant to be kept forever; delete and re-run any time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai import DEFAULT_MODEL_ID
from eai.tracing import PromptTrace, TraceSet, load_model, load_prompts_jsonl, trace_prompt


def collect_split(handle, prompts: list[dict], fingerprint_layer: int, split_name: str) -> TraceSet:
    traces: list[PromptTrace] = []
    t_start = time.perf_counter()
    for i, p in enumerate(prompts, 1):
        tr = trace_prompt(handle, p["id"], p["category"], p["text"], fingerprint_layer=fingerprint_layer)
        traces.append(tr)
        print(
            f"  [{split_name}] {i}/{len(prompts)} {p['id']:<14} cat={p['category']:<12} "
            f"tokens={tr.num_tokens:<4} fwd={tr.forward_pass_seconds * 1000:.0f}ms"
        )
    elapsed = time.perf_counter() - t_start
    print(f"  [{split_name}] done: {len(traces)} prompts in {elapsed:.1f}s")
    return TraceSet.from_traces(traces, fingerprint_layer=fingerprint_layer, handle=handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None, help="force a device, e.g. cpu. Default: auto GPU/CPU split")
    parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"],
                         help="load quantized (only for models too large for full bf16 on this machine); "
                              "perturbs router ground truth, so traces are labeled accordingly")
    parser.add_argument("--offload-folder", default=None,
                         help="disk directory for accelerate to spill weights to when they exceed "
                              "--max-memory-cpu-gb; keeps full bf16 fidelity (no quantization) at the cost "
                              "of disk I/O per layer. Use for large models on RAM-constrained machines.")
    parser.add_argument("--max-memory-cpu-gb", type=float, default=None,
                         help="cap on CPU RAM accelerate will use before offloading the rest to --offload-folder")
    parser.add_argument("--train", default="data/prompts_train.jsonl")
    parser.add_argument("--test", default="data/prompts_test.jsonl")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--fingerprint-layer", type=int, default=2, help="hidden_states index pooled into the fingerprint")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_memory = {"cpu": f"{args.max_memory_cpu_gb}GiB"} if args.max_memory_cpu_gb else None
    if args.offload_folder:
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    t0 = time.perf_counter()
    handle = load_model(
        args.model, device=args.device, quantize=args.quantize,
        offload_folder=args.offload_folder, max_memory=max_memory,
    )
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    print(
        f"Model: layers={handle.num_layers} experts/layer={handle.num_experts} "
        f"top_k={handle.top_k} quantization={handle.quantization}"
    )

    train_prompts = load_prompts_jsonl(args.train)
    test_prompts = load_prompts_jsonl(args.test)
    print(f"Train prompts: {len(train_prompts)}  Test prompts: {len(test_prompts)}")

    train_traces = collect_split(handle, train_prompts, args.fingerprint_layer, "train")
    train_out = out_dir / "traces_train.npz"
    train_traces.save(str(train_out))
    print(f"Saved {train_out}")

    test_traces = collect_split(handle, test_prompts, args.fingerprint_layer, "test")
    test_out = out_dir / "traces_test.npz"
    test_traces.save(str(test_out))
    print(f"Saved {test_out}")


if __name__ == "__main__":
    main()
