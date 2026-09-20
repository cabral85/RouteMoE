"""A global, byte-budgeted expert cache shared across every MoE layer, with
pluggable eviction policies - the foundation for
scripts/benchmark_streaming.py's policy comparison.

Why this is a separate module from the per-layer caches in
eai/chunked_expert_loader.py: those (`ChunkedExpertBlock`,
`ChunkedQwen2MoeExperts`) each own an independent, per-layer, unbounded-until-
told-otherwise dict cache - correct for everything validated through Round 6
(correctness, whole-prompt prefetch, token-level streaming eviction), but not
what a *fair, budget-controlled policy comparison* needs. A fixed
`expert_cache_budget_bytes` has to be enforced *globally*, across all layers
at once (an 8GB budget means 8GB total, not 8GB per layer) - and "which
expert gets evicted when full" has to be a pluggable, swappable rule
(reactive/LRU/LFU/prediction-driven), not hardcoded into the forward pass.

`GlobalExpertCache` is that shared, swappable cache. The existing
`ChunkedExpertBlock`/`ChunkedQwen2MoeExperts` classes are extended (see
chunked_expert_loader.py) to *optionally* delegate to one of these instead of
their own dict - when no global cache is supplied, they behave exactly as
before (zero regression risk to Rounds 4-6's already-published results).

Policy design note: `reactive` and `lru` both never prefetch, so the only
thing that can distinguish them under a byte budget is *which* expert gets
evicted when the budget is exceeded. `reactive` uses FIFO (oldest-loaded,
no notion of recency or frequency at all) - the "a cache exists but nobody
made it smart" baseline. `lru`/`lfu` are the deliberate policies one step up
the sophistication ladder. This is a judgment call the spec that requested
this module left implicit; documented here so it's a stated design decision,
not an accident.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch

EvictionPolicyName = Literal["fifo", "lru", "lfu"]
ExpertKey = tuple[int, int]  # (layer_idx, expert_idx)


@dataclass
class ExpertEntry:
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    size_bytes: int
    last_used_step: int
    access_count: int = 0
    load_order: int = 0  # monotonic counter at load time, for FIFO
    is_pending_prefetch: bool = False  # loaded via prefetch(), not yet confirmed used by a real get()


@dataclass
class BenchmarkStats:
    """Every counter scripts/benchmark_streaming.py's JSONL schema needs,
    accumulated during one (policy, cache_budget, prompt) run. Field names
    match the spec's JSONL example directly where there's a 1:1 mapping."""

    # cache
    expert_accesses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expert_reloads: int = 0  # a miss on a key that had been resident before (evicted, then needed again) - distinct from a first-ever cold miss

    # prefetch
    prefetched_experts: int = 0
    useful_prefetches: int = 0
    wasted_prefetches: int = 0
    bytes_prefetched: int = 0
    bytes_prefetched_used: int = 0
    bytes_prefetched_unused: int = 0

    # I/O (measured, not estimated: real get_tensor calls, real wall-clock time)
    storage_bytes_read: int = 0
    reads_count: int = 0
    read_seconds_total: float = 0.0
    expert_io_wait_seconds: float = 0.0  # wall-clock time spent in a *reactive* (on-the-critical-path) load; prefetch time is tracked separately since it's not necessarily blocking

    # predictor (filled in by the caller, not the cache itself)
    eai_lookup_seconds: float = 0.0
    predictor_top1_hits: int = 0
    predictor_top1_total: int = 0

    # memory (peak tracked by the cache; RSS by the caller)
    peak_resident_bytes: int = 0

    def as_dict(self) -> dict:
        hit_denom = max(1, self.cache_hits + self.cache_misses)
        prefetch_denom = max(1, self.bytes_prefetched)
        return {
            "expert_accesses": self.expert_accesses,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / hit_denom,
            "evictions": self.evictions,
            "expert_reloads": self.expert_reloads,
            "prefetched_experts": self.prefetched_experts,
            "useful_prefetches": self.useful_prefetches,
            "wasted_prefetches": self.wasted_prefetches,
            "prefetch_precision": self.useful_prefetches / max(1, self.useful_prefetches + self.wasted_prefetches),
            "bytes_prefetched": self.bytes_prefetched,
            "bytes_prefetched_used": self.bytes_prefetched_used,
            "bytes_prefetched_unused": self.bytes_prefetched_unused,
            "useful_prefetch_ratio": self.bytes_prefetched_used / prefetch_denom,
            "storage_bytes_read": self.storage_bytes_read,
            "reads_count": self.reads_count,
            "avg_read_size_bytes": self.storage_bytes_read / max(1, self.reads_count),
            "read_seconds_total": self.read_seconds_total,
            "expert_io_wait_seconds": self.expert_io_wait_seconds,
            "eai_lookup_seconds": self.eai_lookup_seconds,
            "predictor_top1_hits": self.predictor_top1_hits,
            "predictor_top1_total": self.predictor_top1_total,
            "predictor_top1_accuracy": self.predictor_top1_hits / max(1, self.predictor_top1_total),
            "peak_resident_bytes": self.peak_resident_bytes,
        }


class GlobalExpertCache:
    """One shared, byte-budgeted cache spanning every MoE layer.

    Correctness guarantee unchanged from the rest of this PoC: this class
    only ever decides *when* an expert's weights are resident in memory. It
    never changes what the router selects (that's computed upstream, by the
    real, unmodified router modules) or what gets computed once weights are
    available - eviction and prefetch are purely a memory/timing decision.
    """

    def __init__(
        self,
        shard_index,  # eai.chunked_expert_loader.ExpertShardIndex
        budget_bytes: int,
        policy: EvictionPolicyName,
        dtype: torch.dtype,
        stats: BenchmarkStats,
        tensor_name_fn,  # (layer_idx, expert_idx, proj) -> checkpoint tensor name
        recycle_every_n_loads: int = 512,
        # Tradeoff, not a free fix: closing+reopening shard file handles costs
        # real time (verified: expert_io_wait_seconds rose noticeably in a
        # real benchmark run after this was added), so this trades some
        # throughput for a hard bound on the mmap page-cache growth this
        # class would otherwise cause across a long run's many expert loads.
        # Set higher for more throughput at the cost of higher peak memory
        # (or 0/None to disable and rely only on the caller's own recycle()
        # calls at prompt/policy boundaries, matching pre-fix behavior).
        device: str | torch.device = "cpu",
        # "cuda" makes `budget_bytes` a VRAM budget instead of a RAM one -
        # every resident expert lands on the GPU, loaded via the shard
        # index's disk read (always CPU/host memory - that's what seek()+
        # read() produces) followed by one .to(device) transfer over PCIe.
        # See scripts/gpu_expert_streaming_experiment.py for real measured
        # numbers on this machine's GPU.
    ):
        self.budget_bytes = budget_bytes
        self.policy: EvictionPolicyName = policy
        self.dtype = dtype
        self.stats = stats
        self._tensor_name_fn = tensor_name_fn
        self._shard_index = shard_index
        self._recycle_every_n_loads = recycle_every_n_loads
        self.device = device

        self._resident: dict[ExpertKey, ExpertEntry] = {}
        self._ever_loaded: set[ExpertKey] = set()
        self._step = 0
        self._load_counter = 0

    # ---- internal ----

    def _load_entry(self, key: ExpertKey) -> ExpertEntry:
        layer_idx, expert_idx = key
        t0 = time.perf_counter()
        # ExpertShardIndex.get_tensor() (see chunked_expert_loader.py) now
        # reads via seek()+readinto() and already returns a real,
        # independently-owned CPU tensor - no aliasing risk, so plain
        # .to(dtype, device) is safe (a no-op when both already match, a
        # real cast/transfer when they don't) without needing copy=True's
        # extra, now-redundant memcpy on top. When device="cuda" this is the
        # actual disk->host->VRAM transfer, timed the same as the CPU path.
        gate = self._shard_index.get_tensor(self._tensor_name_fn(layer_idx, expert_idx, "gate")).to(dtype=self.dtype, device=self.device)
        up = self._shard_index.get_tensor(self._tensor_name_fn(layer_idx, expert_idx, "up")).to(dtype=self.dtype, device=self.device)
        down = self._shard_index.get_tensor(self._tensor_name_fn(layer_idx, expert_idx, "down")).to(dtype=self.dtype, device=self.device)
        elapsed = time.perf_counter() - t0

        size = sum(t.numel() * t.element_size() for t in (gate, up, down))
        self.stats.storage_bytes_read += size
        self.stats.reads_count += 3  # one get_tensor call per projection
        self.stats.read_seconds_total += elapsed

        self._load_counter += 1
        if self._recycle_every_n_loads and self._load_counter % self._recycle_every_n_loads == 0:
            # Now just file-descriptor hygiene (see ExpertShardIndex.recycle's
            # docstring) - pread-based reads never left resident pages behind
            # the way the old mmap-backed reads did, so this is cheap
            # insurance against fd-count growth on a very long run, not a
            # required memory-safety fix anymore.
            self._shard_index.recycle()
        return ExpertEntry(weights=(gate, up, down), size_bytes=size, last_used_step=self._step, load_order=self._load_counter)

    def _choose_victim(self) -> ExpertKey:
        if self.policy == "lru":
            return min(self._resident, key=lambda k: self._resident[k].last_used_step)
        if self.policy == "lfu":
            # tie-break by recency so LFU doesn't get stuck thrashing between
            # equally-rare experts
            return min(self._resident, key=lambda k: (self._resident[k].access_count, self._resident[k].last_used_step))
        # fifo (== "reactive"): oldest-loaded, no notion of recency/frequency at all
        return min(self._resident, key=lambda k: self._resident[k].load_order)

    def _evict_one(self) -> None:
        victim = self._choose_victim()
        entry = self._resident.pop(victim)
        if entry.is_pending_prefetch:
            # evicted before ever being used - a wasted prefetch, resolved now
            self.stats.wasted_prefetches += 1
            self.stats.bytes_prefetched_unused += entry.size_bytes
        self.stats.evictions += 1

    def _admit(self, key: ExpertKey, entry: ExpertEntry) -> None:
        self._resident[key] = entry
        while self.resident_bytes > self.budget_bytes and len(self._resident) > 1:
            self._evict_one()
        self.stats.peak_resident_bytes = max(self.stats.peak_resident_bytes, self.resident_bytes)

    # ---- public API ----

    def get(self, layer_idx: int, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The router needs this expert *right now* - on the critical path.
        A miss here is real, measured I/O wait, not a scheduling choice."""
        key = (layer_idx, expert_idx)
        self._step += 1
        self.stats.expert_accesses += 1

        entry = self._resident.get(key)
        if entry is not None:
            entry.last_used_step = self._step
            entry.access_count += 1
            self.stats.cache_hits += 1
            if entry.is_pending_prefetch:
                entry.is_pending_prefetch = False
                self.stats.useful_prefetches += 1
                self.stats.bytes_prefetched_used += entry.size_bytes
            return entry.weights

        self.stats.cache_misses += 1
        if key in self._ever_loaded:
            self.stats.expert_reloads += 1
        self._ever_loaded.add(key)

        t0 = time.perf_counter()
        entry = self._load_entry(key)
        self.stats.expert_io_wait_seconds += time.perf_counter() - t0
        entry.access_count = 1
        self._admit(key, entry)
        return entry.weights

    def prefetch(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Speculative, off-the-critical-path load - may turn out useful or
        wasted, resolved later (on use or eviction)."""
        for expert_idx in expert_ids:
            key = (layer_idx, expert_idx)
            if key in self._resident:
                continue  # already resident (from a hit or an earlier prefetch) - not a new prefetch
            entry = self._load_entry(key)
            entry.is_pending_prefetch = True
            self._ever_loaded.add(key)
            self.stats.prefetched_experts += 1
            self.stats.bytes_prefetched += entry.size_bytes
            self._admit(key, entry)

    def finalize(self) -> None:
        """Call once at the end of a run: any prefetch still pending (loaded
        speculatively, evicted or not, but never actually used) is a wasted
        prefetch by definition - resolve it so the run's totals are complete
        rather than leaving it in limbo."""
        for entry in self._resident.values():
            if entry.is_pending_prefetch:
                entry.is_pending_prefetch = False
                self.stats.wasted_prefetches += 1
                self.stats.bytes_prefetched_unused += entry.size_bytes

    def evict_all(self) -> None:
        self._resident.clear()

    @property
    def resident_bytes(self) -> int:
        return sum(e.size_bytes for e in self._resident.values())

    @property
    def resident_keys(self) -> set[ExpertKey]:
        return set(self._resident.keys())

    def resident_bytes_for_layer(self, layer_idx: int) -> int:
        return sum(e.size_bytes for (l, _), e in self._resident.items() if l == layer_idx)
