#!/usr/bin/env python
"""Measures the real cost of layer-by-layer streaming on a DENSE (non-MoE)
model - the concrete answer to "does the chunked-loading idea generalize
beyond MoE experts?" Dense models have no sparsity to exploit: every layer
runs on every single token, unconditionally, so there's nothing for a
predictor like EAI to guess - you always need every layer, in order. This
script builds the same load/use/evict cycle chunked_expert_loader.py uses
for MoE experts, but applied to whole transformer BLOCKS of a small dense
model, and measures what it actually costs: how much peak memory it saves,
and how much throughput it gives up, with no prediction to hide the I/O
behind.

Two runs, same model, same prompt, same token count:
  baseline  - the whole model resident in memory the entire time (normal
              from_pretrained + generate).
  streaming - built on torch.device("meta") (zero weights resident at
              construction); each block loads its OWN weights fresh from
              the safetensors checkpoint right before its forward() runs,
              then evicts them back to empty meta tensors immediately after
              - so at any instant, only ONE block's weights (plus the
              always-resident embeddings/lm_head) are actually in memory.

    python scripts/dense_layer_streaming_experiment.py --model gpt2 --max-new-tokens 40
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import psutil
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.chunked_expert_loader import ExpertShardIndex


def find_snapshot_dir(model_id: str) -> str:
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org_name = model_id.replace("/", "--")
    pattern = os.path.join(cache_home, "hub", f"models--{org_name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no cached snapshot for {model_id} under {pattern} - run once with an internet connection to populate the HF cache")
    return matches[0]


class PeakPrivateMemSampler:
    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.peak = self.proc.memory_info().private

    def sample(self) -> int:
        p = self.proc.memory_info().private
        self.peak = max(self.peak, p)
        return p


def run_baseline(model_id: str, input_ids: torch.Tensor, max_new_tokens: int) -> dict:
    from transformers import AutoModelForCausalLM

    sampler = PeakPrivateMemSampler()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    sampler.sample()
    load_seconds = time.perf_counter() - t0

    ids = input_ids.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=ids)
            next_id = out.logits[0, -1].argmax().view(1, 1)
            ids = torch.cat([ids, next_id], dim=1)
            sampler.sample()
    gen_seconds = time.perf_counter() - t0

    return dict(
        mode="baseline (fully resident)", load_seconds=load_seconds, gen_seconds=gen_seconds,
        tokens_per_second=max_new_tokens / gen_seconds, peak_private_gb=sampler.peak / 1e9,
        output_ids=ids,
    )


class StreamingBlock(torch.nn.Module):
    """Wraps one transformer block, structurally on the meta device: every
    forward() call loads exactly this block's own weights from the shard
    index, runs the block, then evicts them back to meta - no residency
    between calls, mirroring ChunkedExpertBlock's per-call load/use/evict
    cycle but for a whole dense layer instead of one MoE expert."""

    def __init__(self, block: torch.nn.Module, shard_index: ExpertShardIndex, prefix: str, stats: dict, recycle_every: int, checkpoint_strip_prefix: str = ""):
        super().__init__()
        self._block = block
        self._shard_index = shard_index
        self._stats = stats
        self._recycle_every = recycle_every
        # captured once, from the block's own (meta) state_dict - shapes are
        # known even on the meta device, and capturing them now (rather than
        # via get_parameter() later) works uniformly for both parameters and
        # buffers (e.g. GPT2's causal-mask buffer isn't an nn.Parameter).
        meta_state = block.state_dict()
        self._shapes = {local: t.shape for local, t in meta_state.items()}
        full_names = {
            local: (prefix + local).removeprefix(checkpoint_strip_prefix) if checkpoint_strip_prefix else prefix + local
            for local in self._shapes
        }
        # only names the checkpoint actually has - a buffer like a causal
        # mask that's regenerated at init time rather than learned/stored
        # simply stays whatever the meta model already set it to.
        self._full_names = {local: full for local, full in full_names.items() if shard_index.has(full)}

    def _load(self) -> None:
        t0 = time.perf_counter()
        state = {}
        nbytes = 0
        for local, full in self._full_names.items():
            t = self._shard_index.get_tensor(full)  # already a real, independently-owned copy (pread-based, see ExpertShardIndex)
            state[local] = t
            nbytes += t.numel() * t.element_size()
        self._block.load_state_dict(state, strict=False, assign=True)
        self._stats["load_seconds"] += time.perf_counter() - t0
        self._stats["bytes_loaded"] += nbytes
        self._stats["loads"] += 1
        if self._recycle_every and self._stats["loads"] % self._recycle_every == 0:
            self._shard_index.recycle()

    def _evict(self) -> None:
        meta_state = {local: torch.empty(shape, device="meta") for local, shape in self._shapes.items() if local in self._full_names}
        self._block.load_state_dict(meta_state, strict=False, assign=True)

    def forward(self, *args, **kwargs):
        self._load()
        out = self._block(*args, **kwargs)
        self._evict()
        return out


def run_streaming(model_id: str, model_dir: str, input_ids: torch.Tensor, max_new_tokens: int, block_attr: str, prefix_fmt: str, checkpoint_strip_prefix: str) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM

    sampler = PeakPrivateMemSampler()
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.eval()
    model.tie_weights()  # e.g. gpt2 ties lm_head.weight to wte.weight and drops the duplicate from the checkpoint entirely - must be set up before we skip "missing" tied names below

    shard_index = ExpertShardIndex.load(model_dir)
    stats = dict(load_seconds=0.0, bytes_loaded=0, loads=0)

    def to_checkpoint_name(name: str) -> str:
        return name.removeprefix(checkpoint_strip_prefix) if checkpoint_strip_prefix else name

    # everything EXCEPT the transformer blocks stays permanently resident -
    # embeddings/lm_head/final norm are needed exactly once per token
    # regardless, so streaming them buys nothing and would only add
    # overhead. This mirrors chunked_expert_loader.py's own design: only
    # the part of the model with real sparsity/reuse structure is chunked.
    blocks_container = model
    for attr in block_attr.split("."):
        blocks_container = getattr(blocks_container, attr)
    num_layers = len(blocks_container)

    # state_dict(), not just named_parameters() - buffers outside any block
    # (e.g. Qwen2Model.rotary_emb.inv_freq) need loading too, same as every
    # StreamingBlock already does for its own buffers below.
    non_block_names = [n for n in model.state_dict() if not any(n.startswith(prefix_fmt.format(i=i)) for i in range(num_layers))]
    non_block_state = {
        n: shard_index.get_tensor(to_checkpoint_name(n))
        for n in non_block_names if shard_index.has(to_checkpoint_name(n))
        # names absent from the checkpoint (e.g. lm_head.weight when tied to
        # wte.weight, or a buffer regenerated at init rather than stored)
        # are skipped - the former is fixed by the tie_weights() call below,
        # the latter is already correctly set on the meta model.
    }
    model.load_state_dict(non_block_state, strict=False, assign=True)
    model.tie_weights()  # assign=True replaces the wte parameter object outright, breaking the earlier tie - must re-tie so lm_head follows the newly-loaded real tensor, not the stale meta one

    # Buffers that are COMPUTED at __init__ time rather than loaded from the
    # checkpoint (e.g. rotary embeddings' inv_freq) are still on the meta
    # device here - the checkpoint never had them to load in the first
    # place. Rebuild any such submodule fresh, on a real device, rather than
    # trying to materialize meaningless "empty" data for them.
    for name, module in model.named_modules():
        if any(b.is_meta for b in module.buffers(recurse=False)) and hasattr(module, "config"):
            parent_path, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attr, type(module)(module.config, device="cpu"))

    for i in range(num_layers):
        prefix = prefix_fmt.format(i=i)
        blocks_container[i] = StreamingBlock(blocks_container[i], shard_index, prefix, stats, recycle_every=8, checkpoint_strip_prefix=checkpoint_strip_prefix)

    sampler.sample()
    load_seconds = time.perf_counter() - t0

    ids = input_ids.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=ids)
            next_id = out.logits[0, -1].argmax().view(1, 1)
            ids = torch.cat([ids, next_id], dim=1)
            sampler.sample()
    gen_seconds = time.perf_counter() - t0

    return dict(
        mode="streaming (one block resident at a time)", load_seconds=load_seconds, gen_seconds=gen_seconds,
        tokens_per_second=max_new_tokens / gen_seconds, peak_private_gb=sampler.peak / 1e9,
        output_ids=ids, block_loads=stats["loads"], bytes_loaded_gb=stats["bytes_loaded"] / 1e9,
        block_load_seconds=stats["load_seconds"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--block-attr", default="transformer.h", help="dotted path to the list of transformer blocks (gpt2: transformer.h)")
    parser.add_argument("--prefix-fmt", default="transformer.h.{i}.", help="parameter-name prefix format for block i, in the MODEL's own named_parameters() namespace")
    parser.add_argument("--checkpoint-strip-prefix", default="transformer.", help="prefix present in named_parameters() but ABSENT from the checkpoint's own tensor names (gpt2 saves relative to the base model, e.g. 'h.0...' not 'transformer.h.0...') - empty string if the checkpoint uses the same names as named_parameters()")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_dir = find_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    print(f"model={args.model}  prompt={args.prompt!r}  max_new_tokens={args.max_new_tokens}")

    print("\n=== baseline (fully resident) ===")
    baseline = run_baseline(args.model, input_ids, args.max_new_tokens)
    print(f"  load: {baseline['load_seconds']:.2f}s  gen: {baseline['gen_seconds']:.2f}s  "
          f"tok/s: {baseline['tokens_per_second']:.2f}  peak_private: {baseline['peak_private_gb']:.3f}GB")
    print("  output:", tokenizer.decode(baseline["output_ids"][0]))

    print("\n=== streaming (one block resident at a time) ===")
    streaming = run_streaming(args.model, model_dir, input_ids, args.max_new_tokens, args.block_attr, args.prefix_fmt, args.checkpoint_strip_prefix)
    print(f"  load: {streaming['load_seconds']:.2f}s  gen: {streaming['gen_seconds']:.2f}s  "
          f"tok/s: {streaming['tokens_per_second']:.2f}  peak_private: {streaming['peak_private_gb']:.3f}GB")
    print(f"  block_loads={streaming['block_loads']}  bytes_loaded={streaming['bytes_loaded_gb']:.3f}GB  block_load_seconds={streaming['block_load_seconds']:.2f}s")
    print("  output:", tokenizer.decode(streaming["output_ids"][0]))

    match = torch.equal(baseline["output_ids"], streaming["output_ids"])
    print(f"\n=== comparison ===")
    print(f"  identical output tokens: {match}  (must be True - streaming must never change what the model computes, only when weights are resident)")
    print(f"  peak memory:  baseline={baseline['peak_private_gb']:.3f}GB  streaming={streaming['peak_private_gb']:.3f}GB  "
          f"({(1 - streaming['peak_private_gb']/baseline['peak_private_gb'])*100:.1f}% lower)")
    print(f"  throughput:   baseline={baseline['tokens_per_second']:.2f} tok/s  streaming={streaming['tokens_per_second']:.2f} tok/s  "
          f"({streaming['tokens_per_second']/baseline['tokens_per_second']*100:.1f}% of baseline)")
    if not match:
        raise SystemExit("FAILED: streaming changed the model's output - this must never happen, it's a correctness bug, not a performance tradeoff")


if __name__ == "__main__":
    main()
