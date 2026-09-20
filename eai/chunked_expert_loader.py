"""Load an MoE model with expert weights fetched on demand, chunk by chunk,
instead of materializing every expert into memory at load time.

This is the mechanism the rest of the PoC only *simulated*: eai/predictor.py
scores whether we *could* have guessed the right experts ahead of time, but
never actually changed what got loaded into memory. This module acts on that
prediction - it's the first piece of the project that changes real resource
usage, not just a metric.

Backbone weights (embeddings, attention, norms, router gates, lm_head) are
always resident - needed regardless of which experts fire, and tiny relative
to expert weights (~1GB of OLMoE's ~14GB total, see README). Expert FFN
weights are the opposite: 64 experts/layer here, only 8 fire per token, and
unlike the backbone we can *predict* which ones we'll need before running the
model at all - that's exactly what eai/predictor.py is for.

Why this bypasses AutoModelForCausalLM rather than patching it: the installed
Transformers version stores all of a layer's experts as one dense
`(num_experts, ...)` tensor (`OlmoeExperts.gate_up_proj`/`down_proj`), built
from the checkpoint's actual per-expert tensors via an internal conversion
step (`@use_experts_implementation`) that isn't a public, stable hook to
intercept safely. A single dense tensor also can't have "some experts
resident, some not" - PyTorch tensors are contiguous. So instead of fighting
that internal machinery, this reads the checkpoint's original, separately-
named per-expert tensors directly (`model.layers.{L}.mlp.experts.{E}.
{gate,up,down}_proj.weight` - confirmed present in the raw safetensors shards
by inspection) and replaces each layer's MoE block wholesale with
`ChunkedExpertBlock`, a small module backed by a plain dict cache instead of
one big tensor. Everything else (attention, norms, embeddings, rotary,
lm_head) reuses the real Transformers modules unmodified, loaded normally -
they're not the expensive part and reimplementing them would only add risk
without addressing the actual question.

Correctness, not just memory, is the point: the router math and expert FFN
math here are bit-for-bit the same as the reference model (see
OlmoeSparseMoeBlock/OlmoeExperts in modeling_olmoe.py) - only *when* each
expert's weights get read off disk changes. scripts/chunked_inference_experiment.py
validates this by comparing selected experts against the ground truth traces
already collected in Phase 2 for the same prompts, not just by assuming it.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

# safetensors' own dtype strings -> torch dtypes, for parsing shard headers
# directly (see ExpertShardIndex's seek+read-based reads below).
_SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}


@dataclass
class ExpertShardIndex:
    """Maps every tensor name to the shard file that holds it, and gives
    direct-by-name reads - built once from the checkpoint's own
    model.safetensors.index.json, the same manifest `from_pretrained` uses.

    Reads via plain seek()+read() on a regular buffered file handle, NOT
    `safetensors.safe_open`/mmap. Why: mmap keeps every touched page
    resident in the process's own working set for as long as the file
    mapping stays open, regardless of whether our own cache still
    references the tensor - confirmed empirically (private memory didn't
    drop after evicting our own tensors AND gc.collect(); it only dropped
    once the mmap'd file handle itself was closed). A plain seek()+read()
    copies bytes straight into a buffer we own outright: once we drop that
    buffer, it's ordinary Python/torch memory, freed exactly like any other
    allocation - no page-cache accumulation to periodically "recycle" away.
    This is the same fix [kimi-k3-in-c](https://github.com/josesilva05/kimi-k3-in-c)
    uses (O_DIRECT) for the identical reason; this uses plain buffered reads
    rather than true O_DIRECT (which needs sector-aligned reads and
    platform-specific flags, and POSIX pread isn't even available on
    Windows) since the actual problem here was mmap's resident-working-set
    behavior, not the OS page cache itself - buffered reads still benefit
    from the OS cache on repeat reads, they just don't count that cache
    against *our* process's private memory the way an active mapping does.
    """

    model_dir: Path
    weight_map: dict[str, str]
    _headers: dict[str, dict[str, tuple[torch.dtype, tuple[int, ...], int, int]]] = field(default_factory=dict, repr=False)
    _fds: dict[str, object] = field(default_factory=dict, repr=False)

    @staticmethod
    def load(model_dir: str) -> "ExpertShardIndex":
        model_dir = Path(model_dir)
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            return ExpertShardIndex(model_dir=model_dir, weight_map=index["weight_map"])
        # Small models often ship as a single unsharded model.safetensors
        # with no index.json (e.g. gpt2) - build an equivalent weight_map by
        # reading that one file's own header, so every other caller of this
        # class works identically either way.
        single_file = model_dir / "model.safetensors"
        if not single_file.exists():
            raise FileNotFoundError(f"neither model.safetensors.index.json nor model.safetensors found under {model_dir}")
        with safe_open(str(single_file), framework="pt") as f:
            weight_map = {name: "model.safetensors" for name in f.keys()}
        return ExpertShardIndex(model_dir=model_dir, weight_map=weight_map)

    def _header(self, shard_name: str) -> dict[str, tuple[torch.dtype, tuple[int, ...], int, int]]:
        if shard_name not in self._headers:
            path = self.model_dir / shard_name
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                raw_header = json.loads(f.read(header_len))
            base = 8 + header_len
            entries = {}
            for name, meta in raw_header.items():
                if name == "__metadata__":
                    continue
                dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
                if dtype is None:
                    raise ValueError(f"unsupported safetensors dtype {meta['dtype']!r} for tensor {name!r} in {path}")
                start, end = meta["data_offsets"]
                entries[name] = (dtype, tuple(meta["shape"]), base + start, base + end)
            self._headers[shard_name] = entries
        return self._headers[shard_name]

    def _fd(self, shard_name: str):
        # A plain buffered file object, not os.open()/os.pread() - pread is
        # POSIX-only (no os.pread on Windows). seek()+readinto() on a
        # regular file object is portable and gives the identical property
        # that matters here: bytes copied into a buffer we own, no mmap.
        # Default buffering (not 0/unbuffered) - unbuffered forced every
        # read into its own raw OS syscall with no readahead, which measured
        # slower than default buffering for this access pattern.
        if shard_name not in self._fds:
            self._fds[shard_name] = open(self.model_dir / shard_name, "rb")
        return self._fds[shard_name]

    def get_tensor(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map[name]
        dtype, shape, start, end = self._header(shard_name)[name]
        f = self._fd(shard_name)
        f.seek(start)
        buf = bytearray(end - start)
        f.readinto(buf)  # reads straight into a buffer we own - one fewer copy than read()+bytearray(), still no mmap
        # torch.frombuffer is a view over `buf`; .clone() makes the final
        # tensor fully independent of buf's lifetime, removing any doubt
        # about ownership.
        return torch.frombuffer(buf, dtype=dtype).reshape(shape).clone()

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def close(self) -> None:
        for f in self._fds.values():
            f.close()
        self._fds.clear()

    def recycle(self) -> None:
        """Kept for interface compatibility with every existing caller (they
        call this between prompts/at intervals) - now just closes file
        descriptors for hygiene (avoiding fd-count growth on a very long
        run). Unlike the old mmap-backed version, this is no longer load-
        bearing for memory safety: seek()+read() never leaves resident pages
        behind the way an open mmap did, so skipping this doesn't
        reintroduce the original problem - it's cheap insurance, not a
        required fix."""
        self.close()


@dataclass
class ExpertLoadStats:
    """Counters a run accumulates across all layers, for reporting afterward."""

    prefetched: int = 0
    hits: int = 0  # router needed an expert that prefetch already loaded
    misses: int = 0  # router needed an expert that had to be fetched reactively, mid-forward-pass
    evictions: int = 0
    bytes_loaded: int = 0
    fetch_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "prefetched": self.prefetched,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, self.hits + self.misses),
            "evictions": self.evictions,
            "bytes_loaded": self.bytes_loaded,
            "fetch_seconds": self.fetch_seconds,
        }


class _ExpertCache(torch.nn.Module):
    """Shared caching/eviction mechanics for one MoE layer's routed experts -
    load-on-demand, prefetch, and three eviction policies (all, exact-set,
    LRU-by-budget). Architecture-specific subclasses (OLMoE, Qwen2Moe, ...)
    only need to supply `_tensor_name` and `forward`'s FFN/router math; the
    cache bookkeeping is identical across all of them.

    `global_cache` (optional): when set, this block's residency is delegated
    entirely to a shared `eai.expert_cache.GlobalExpertCache` spanning every
    layer under one byte budget, instead of this block's own unbounded
    per-layer dict - see scripts/benchmark_streaming.py. When left `None`
    (every caller through Round 6), behavior is unchanged from before this
    parameter existed - zero regression risk to already-published results.
    """

    def __init__(
        self, layer_idx: int, num_experts: int, shard_index: ExpertShardIndex, stats: ExpertLoadStats,
        dtype: torch.dtype = torch.bfloat16, global_cache=None, device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self._shard_index = shard_index
        self._stats = stats
        self._dtype = dtype
        self._device = device
        self._global_cache = global_cache
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._last_used_step: dict[int, int] = {}  # expert_idx -> step counter, for LRU eviction
        self._step = 0
        self.last_selected: torch.Tensor | None = None  # (num_tokens, top_k) int - this layer's actual router pick from the most recent forward() call, for external correctness checks
        self.last_selected_weights: torch.Tensor | None = None  # (num_tokens, top_k) float - router probabilities for last_selected, for trace collection (see eai/tracing.py's trace_prompt_chunked)

    def get_expert(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The one method forward() actually needs: resolve one expert's
        weights, whichever backing store is in play. When delegating to a
        GlobalExpertCache this IS the hit/miss/load path (no separate
        `_note_needed` bookkeeping needed - the global cache does its own).
        """
        if self._global_cache is not None:
            return self._global_cache.get(self.layer_idx, expert_idx)
        return self._cache[expert_idx]

    def _tensor_name(self, expert_idx: int, proj: str) -> str:
        raise NotImplementedError

    def attach_global_cache(self, global_cache) -> None:
        """Switch this block over to a shared `GlobalExpertCache` after
        loading - the pattern scripts/benchmark_streaming.py uses: load the
        model normally first (`load_chunked_model`, unchanged), then attach
        one shared cache to every block. Kept separate from the constructor
        parameter (which still exists, for symmetry) because the cache needs
        this block's own `_tensor_name` naming scheme, which isn't known
        until the block itself is constructed."""
        self._global_cache = global_cache

    def materialize(self, expert_idx: int) -> None:
        """Load one expert's weights off disk, if not already resident."""
        if expert_idx in self._cache:
            return
        t0 = time.perf_counter()
        gate = self._shard_index.get_tensor(self._tensor_name(expert_idx, "gate")).to(dtype=self._dtype, device=self._device)
        up = self._shard_index.get_tensor(self._tensor_name(expert_idx, "up")).to(dtype=self._dtype, device=self._device)
        down = self._shard_index.get_tensor(self._tensor_name(expert_idx, "down")).to(dtype=self._dtype, device=self._device)
        self._cache[expert_idx] = (gate, up, down)
        self._stats.bytes_loaded += sum(t.numel() * t.element_size() for t in (gate, up, down))
        self._stats.fetch_seconds += time.perf_counter() - t0

    def prefetch(self, expert_ids: list[int]) -> None:
        """Warm the cache with a predicted set of experts before a forward pass -
        this is where eai/predictor.py's output plugs in."""
        for e in expert_ids:
            if e not in self._cache:
                self.materialize(e)
                self._stats.prefetched += 1

    def evict_all(self) -> None:
        self._cache.clear()
        self._last_used_step.clear()

    def evict_except(self, keep_ids: set[int] | list[int]) -> None:
        """Drop every resident expert not in `keep_ids` - the 'unload' half of
        load -> process -> unload. Used between generation steps to shrink the
        cache down to (ideally) just what the next token needs, instead of
        letting a whole prompt's worth of distinct experts accumulate."""
        keep = set(keep_ids)
        to_drop = [e for e in self._cache if e not in keep]
        for e in to_drop:
            del self._cache[e]
            self._last_used_step.pop(e, None)
            self._stats.evictions += 1

    def evict_lru(self, max_resident: int) -> None:
        """Evict least-recently-used experts until at most `max_resident` remain -
        the plain-recency baseline (no prediction at all), for comparison
        against EAI-predicted prefetch. Mirrors the LRU-with-size-budget
        approach real streaming MoE inference engines use as their fallback
        when no predictor is available."""
        if len(self._cache) <= max_resident:
            return
        by_recency = sorted(self._cache.keys(), key=lambda e: self._last_used_step.get(e, -1))
        to_drop = by_recency[: len(self._cache) - max_resident]
        for e in to_drop:
            del self._cache[e]
            self._last_used_step.pop(e, None)
            self._stats.evictions += 1

    @property
    def resident_experts(self) -> set[int]:
        if self._global_cache is not None:
            return {e for (l, e) in self._global_cache.resident_keys if l == self.layer_idx}
        return set(self._cache.keys())

    @property
    def resident_bytes(self) -> int:
        if self._global_cache is not None:
            # this block's share of the shared cache's residency - the global
            # cache's own .resident_bytes is the number that matters for
            # budget/peak-memory reporting; this per-layer view exists for
            # API parity with the non-delegating path.
            return self._global_cache.resident_bytes_for_layer(self.layer_idx)
        return sum(t.numel() * t.element_size() for triplet in self._cache.values() for t in triplet)

    def _note_needed(self, expert_ids: list[int]) -> None:
        """Bump hit/miss stats and the LRU clock for a set of experts the
        router just asked for - shared by every subclass's forward().

        A no-op when delegating to a GlobalExpertCache: that cache's own
        `get()` (called via `get_expert()`) already records hits/misses/
        reloads/eviction bookkeeping itself - doing it here too would
        double-count against the wrong stats object (this block's local
        `ExpertLoadStats`, not the benchmark's `BenchmarkStats`).
        """
        if self._global_cache is not None:
            return
        self._step += 1
        for e in expert_ids:
            if e in self._cache:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
                self.materialize(e)
            self._last_used_step[e] = self._step


class ChunkedExpertBlock(_ExpertCache):
    """Drop-in replacement for one OLMoE layer's *whole* MoE block (router +
    experts - OLMoE has no shared expert, so there's nothing else to leave
    stock), backed by a lazily-populated dict cache instead of one dense
    (num_experts, ...) tensor.

    Numerically identical to the reference OlmoeSparseMoeBlock: same router
    math (linear -> softmax -> top-k, optionally renormalized), same expert
    FFN math (silu(gate(x)) * up(x), then down(.), weighted by router score).
    The only difference from the reference is *when* each expert's weights
    get read off disk - never what gets computed.
    """

    def __init__(
        self, layer_idx: int, config, shard_index: ExpertShardIndex, stats: ExpertLoadStats,
        dtype: torch.dtype = torch.bfloat16, global_cache=None, device: str | torch.device = "cpu",
    ):
        super().__init__(layer_idx, config.num_experts, shard_index, stats, dtype, global_cache=global_cache, device=device)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.gate_weight: torch.nn.Parameter | None = None  # router weight - set by the loader, always resident

    def _tensor_name(self, expert_idx: int, proj: str) -> str:
        return f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.{proj}_proj.weight"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(x, self.gate_weight)
        router_probs = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
        top_k_weights, top_k_index = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(router_logits.dtype)
        self.last_selected = top_k_index.detach().clone()
        self.last_selected_weights = top_k_weights.detach().clone()

        needed = torch.unique(top_k_index).tolist()
        self._note_needed(needed)

        final = torch.zeros_like(x)
        for e in needed:
            gate_w, up_w, down_w = self.get_expert(e)
            token_idx, slot_idx = torch.where(top_k_index == e)
            current = x[token_idx]
            gate = F.linear(current, gate_w)
            up = F.linear(current, up_w)
            hidden = F.silu(gate) * up
            out = F.linear(hidden, down_w)
            out = out * top_k_weights[token_idx, slot_idx, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        return final.view(batch_size, seq_len, hidden_dim)


class ChunkedQwen2MoeExperts(_ExpertCache):
    """Surgical replacement for just `Qwen2MoeSparseMoeBlock.experts`
    (the `Qwen2MoeExperts` submodule) - NOT the whole MoE block. Unlike
    OLMoE, Qwen1.5-MoE has an always-active shared expert
    (`shared_expert`/`shared_expert_gate`) blended into every token's output
    regardless of routing; that stays the real, stock Transformers module,
    loaded normally (it's small and every token needs it, so there's nothing
    to chunk there). Only the *routed*, sparse experts - the part where
    "which ones am I going to need" is actually a meaningful question - get
    replaced. `Qwen2MoeSparseMoeBlock.forward` itself is untouched: it still
    computes the router and shared-expert output normally, and calls
    `self.experts(hidden_states, selected_experts, routing_weights)` exactly
    as before - `selected_experts`/`routing_weights` still come from the
    real, unmodified router, this class only supplies what said call now
    resolves to.
    """

    def __init__(
        self, layer_idx: int, config, shard_index: ExpertShardIndex, stats: ExpertLoadStats,
        dtype: torch.dtype = torch.bfloat16, global_cache=None, device: str | torch.device = "cpu",
    ):
        super().__init__(layer_idx, config.num_experts, shard_index, stats, dtype, global_cache=global_cache, device=device)
        self.top_k = config.num_experts_per_tok  # not used internally (the stock router applies top-k), kept for API parity with ChunkedExpertBlock

    def _tensor_name(self, expert_idx: int, proj: str) -> str:
        return f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.{proj}_proj.weight"

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        needed = torch.unique(top_k_index).tolist()
        self.last_selected = top_k_index.detach().clone()
        self.last_selected_weights = top_k_weights.detach().clone()
        self._note_needed(needed)

        for e in needed:
            gate_w, up_w, down_w = self.get_expert(e)
            token_idx, slot_idx = torch.where(top_k_index == e)
            current = hidden_states[token_idx]
            gate = F.linear(current, gate_w)
            up = F.linear(current, up_w)
            hidden = F.silu(gate) * up
            out = F.linear(hidden, down_w)
            out = out * top_k_weights[token_idx, slot_idx, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        return final


def _load_chunked_whole_block(model_id: str, model_dir: str, dtype: torch.dtype, rotary_embedding_cls, device: str | torch.device = "cpu"):
    """Shared implementation behind every architecture whose MoE block is
    *entirely* routed experts (no always-active shared expert) - so the whole
    `.mlp` submodule can be swapped for `ChunkedExpertBlock` wholesale, not
    just its `.experts` piece. OLMoE and Qwen3-MoE both fit this shape and
    use bit-for-bit identical router/FFN math (confirmed by reading both
    modeling files side by side) - only the rotary embedding class differs,
    passed in by the caller.

    `model_dir` is the local snapshot directory already on disk (from
    huggingface_hub's cache) - this never re-downloads, only reads what's
    already there.

    `dtype` defaults to bf16 (matching the checkpoint and the rest of this
    PoC). float32 is available for debugging numerical drift: bf16 has only
    ~3 decimal digits of precision, a fused gate_up_proj matmul (reference
    model) vs. two separate gate_proj/up_proj matmuls (this module, reading
    the checkpoint's original separate tensors) can round differently, and
    that tiny difference compounds across layers - occasionally enough to
    flip a close top-k router decision. float32 shrinks that gap by several
    orders of magnitude, which is how this was diagnosed as expected
    floating-point non-associativity rather than a logic bug (see README).

    Returns (model, chunked_blocks, shard_index, stats). Call
    `block.prefetch([...])` on each entry in `chunked_blocks` before running
    the model, seeded from eai/predictor.py's per-layer predictions.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    shard_index = ExpertShardIndex.load(model_dir)
    stats = ExpertLoadStats()

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    # Materialize every real (non-expert) parameter from the checkpoint.
    to_assign = {}
    def is_expert_weight(name: str) -> bool:
        return ".mlp.experts." in name

    for name, _ in model.named_parameters():
        if is_expert_weight(name):
            continue
        if shard_index.has(name):
            to_assign[name] = shard_index.get_tensor(name).to(dtype=dtype, device=device)
    missing, unexpected = model.load_state_dict(to_assign, strict=False, assign=True)
    # `missing` should only ever be the expert tensors we deliberately skipped
    # (mlp.experts.gate_up_proj/down_proj) - anything else missing means the
    # backbone didn't load correctly and downstream results would be silently wrong.
    real_missing = [m for m in missing if not is_expert_weight(m)]
    if real_missing:
        raise RuntimeError(f"backbone load left unexpected parameters uninitialized: {real_missing[:10]}")

    # Rotary embeddings are computed at __init__ time (not stored in the
    # checkpoint), so under the meta-device context above they never got real
    # values - cheap to just build a fresh one normally rather than replicate
    # the frequency formula here. Must land on the same device as the hidden
    # states it'll be applied to.
    model.model.rotary_emb = rotary_embedding_cls(config=config).to(device)

    chunked_blocks: list[ChunkedExpertBlock] = []
    for i, layer in enumerate(model.model.layers):
        block = ChunkedExpertBlock(i, config, shard_index, stats, dtype=dtype, device=device)
        gate_name = f"model.layers.{i}.mlp.gate.weight"
        block.gate_weight = torch.nn.Parameter(shard_index.get_tensor(gate_name).to(dtype=dtype, device=device), requires_grad=False)
        layer.mlp = block
        chunked_blocks.append(block)

    # Safety net, not a formality: assign=True only replaces what we
    # explicitly listed above. Any parameter this loop didn't know to look
    # for (a future transformers version renaming something, say) would
    # otherwise stay silently on the meta device - "works" until the first
    # forward pass hits it with an opaque "Cannot copy out of meta tensor"
    # error deep in a library, or worse, silently no-ops. Fail loud, here,
    # with the exact parameter name, instead.
    stray_meta = [n for n, p in model.named_parameters() if p.is_meta and not is_expert_weight(n)]
    if stray_meta:
        raise RuntimeError(f"model still has non-expert parameters on the meta device after loading: {stray_meta[:10]}")

    model.eval()
    return model, chunked_blocks, shard_index, stats


def load_chunked_olmoe(model_id: str, model_dir: str, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu"):
    """OLMoE: see `_load_chunked_whole_block`."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding

    return _load_chunked_whole_block(model_id, model_dir, dtype, OlmoeRotaryEmbedding, device=device)


def load_chunked_qwen3moe(model_id: str, model_dir: str, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu"):
    """Qwen3-MoE (e.g. Qwen3-30B-A3B): same router/FFN math as OLMoE, no
    shared expert - see `_load_chunked_whole_block`. The whole point of
    building this: with zero experts resident at load time and only ever
    materializing a prediction-sized subset, this never needs the ~60GB
    (bf16) a normal `AutoModelForCausalLM.from_pretrained` load requires -
    see README's "Attempting a 30B-class model" for why that normal path
    reliably failed on this machine, independent of quantization or OS.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

    return _load_chunked_whole_block(model_id, model_dir, dtype, Qwen3MoeRotaryEmbedding, device=device)


def load_chunked_qwen2moe(model_id: str, model_dir: str, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu"):
    """Same idea as `load_chunked_olmoe`, generalized to a second, unrelated
    architecture (Qwen1.5-MoE) - proof the approach isn't OLMoE-specific.

    The one real architectural difference: Qwen1.5-MoE has an always-active
    `shared_expert` (blended into every token's output via a sigmoid gate,
    alongside the sparsely-routed experts). That module is small, needed by
    every token regardless of routing, and left as the real, unmodified
    Transformers module - only `Qwen2MoeSparseMoeBlock.experts` (the
    genuinely sparse, genuinely chunkable part) gets replaced, via
    `ChunkedQwen2MoeExperts` - see its docstring for why this is a surgical
    swap rather than replacing the whole `.mlp` block the way OLMoE's loader
    does.

    Returns (model, chunked_blocks, shard_index, stats), same contract as
    `load_chunked_olmoe`.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeRotaryEmbedding

    config = AutoConfig.from_pretrained(model_id)
    shard_index = ExpertShardIndex.load(model_dir)
    stats = ExpertLoadStats()

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    def is_expert_weight(name: str) -> bool:
        return ".mlp.experts." in name

    to_assign = {}
    for name, _ in model.named_parameters():
        if is_expert_weight(name):
            continue
        if shard_index.has(name):
            to_assign[name] = shard_index.get_tensor(name).to(dtype=dtype, device=device)
    missing, unexpected = model.load_state_dict(to_assign, strict=False, assign=True)
    real_missing = [m for m in missing if not is_expert_weight(m)]
    if real_missing:
        raise RuntimeError(f"backbone load left unexpected parameters uninitialized: {real_missing[:10]}")

    model.model.rotary_emb = Qwen2MoeRotaryEmbedding(config=config).to(device)

    chunked_blocks: list[ChunkedQwen2MoeExperts] = []
    for i, layer in enumerate(model.model.layers):
        moe_block = layer.mlp
        if not hasattr(moe_block, "experts"):
            # a dense (mlp_only_layers) layer for this config - nothing to chunk
            continue
        experts = ChunkedQwen2MoeExperts(i, config, shard_index, stats, dtype=dtype, device=device)
        moe_block.experts = experts
        chunked_blocks.append(experts)

    stray_meta = [n for n, p in model.named_parameters() if p.is_meta and not is_expert_weight(n)]
    if stray_meta:
        raise RuntimeError(f"model still has non-expert parameters on the meta device after loading: {stray_meta[:10]}")

    model.eval()
    return model, chunked_blocks, shard_index, stats


_LOADERS_BY_MODEL_TYPE = {
    "olmoe": load_chunked_olmoe,
    "qwen2_moe": load_chunked_qwen2moe,
    "qwen3_moe": load_chunked_qwen3moe,
}


def load_chunked_model(model_id: str, model_dir: str, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu"):
    """Dispatch to the right architecture-specific chunked loader based on
    the checkpoint's own `model_type` - so callers (experiment scripts) don't
    need to know or care which one they're driving. Add a new architecture
    by writing `load_chunked_<name>` the same way the two existing ones are
    built, then registering it here.

    `device`: where the BACKBONE (attention, norms, embeddings, router
    gates) and any cached expert weights live. "cuda" moves everything
    except not-yet-materialized experts onto the GPU; experts still stream
    from disk on demand exactly as on CPU, just landing in VRAM instead of
    system RAM - see eai/expert_cache.py's `device` parameter for the
    GlobalExpertCache side of this.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    loader = _LOADERS_BY_MODEL_TYPE.get(config.model_type)
    if loader is None:
        raise ValueError(
            f"no chunked loader registered for model_type={config.model_type!r} "
            f"(known: {sorted(_LOADERS_BY_MODEL_TYPE)}) - see the two existing loaders "
            "for the pattern to follow when adding a new architecture."
        )
    return loader(model_id, model_dir, dtype=dtype, device=device)
