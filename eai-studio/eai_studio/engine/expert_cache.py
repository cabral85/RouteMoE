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

EvictionPolicyName = Literal["fifo", "lru", "lfu", "hybrid"]
ExpertKey = tuple[int, int]  # (layer_idx, expert_idx)


@dataclass
class ExpertEntry:
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    size_bytes: int
    last_used_step: int
    access_count: int = 0
    load_order: int = 0  # monotonic counter at load time, for FIFO
    is_pending_prefetch: bool = False  # loaded via prefetch(), not yet confirmed used by a real get()
    # Set externally (by the caller's predictor/coactivation logic, via
    # GlobalExpertCache.update_scores()) - the cache itself never computes
    # these, it only reads them for the "hybrid" eviction rule. Decay toward
    # 0 is the caller's responsibility (a stale high score from many steps
    # ago shouldn't protect an entry forever); this class just stores
    # whatever it was last told.
    predicted_probability: float = 0.0
    coactivation_score: float = 0.0


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

    # admission control (see GlobalExpertCache.prefetch's admission gate) -
    # 0/0 for every policy that doesn't use it (admission_control=False),
    # not just missing - so a report can tell "not applicable" from "always
    # admitted" at a glance.
    admission_checks: int = 0
    admission_rejected: int = 0

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

    # distinct (layer, expert) keys ever loaded (reactive or prefetch) -
    # feeds reload_amplification below. Not the same as prefetched_experts
    # or cache_misses: this counts UNIQUE keys, those count events.
    unique_experts_loaded: int = 0

    def as_dict(self) -> dict:
        hit_denom = max(1, self.cache_hits + self.cache_misses)
        prefetch_denom = max(1, self.bytes_prefetched)
        total_loads = self.cache_misses + self.prefetched_experts  # every _load_entry() call, reactive or speculative
        return {
            "expert_accesses": self.expert_accesses,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / hit_denom,
            "evictions": self.evictions,
            "expert_reloads": self.expert_reloads,
            "eviction_churn": self.evictions / max(1, self.expert_accesses),
            "prefetched_experts": self.prefetched_experts,
            "useful_prefetches": self.useful_prefetches,
            "wasted_prefetches": self.wasted_prefetches,
            "prefetch_precision": self.useful_prefetches / max(1, self.useful_prefetches + self.wasted_prefetches),
            "bytes_prefetched": self.bytes_prefetched,
            "bytes_prefetched_used": self.bytes_prefetched_used,
            "bytes_prefetched_unused": self.bytes_prefetched_unused,
            "useful_prefetch_ratio": self.bytes_prefetched_used / prefetch_denom,
            "unique_experts_loaded": self.unique_experts_loaded,
            "reload_amplification": total_loads / max(1, self.unique_experts_loaded),
            "useful_io_ratio": (self.storage_bytes_read - self.bytes_prefetched_unused) / max(1, self.storage_bytes_read),
            "admission_checks": self.admission_checks,
            "admission_rejected": self.admission_rejected,
            "admission_rejection_rate": self.admission_rejected / max(1, self.admission_checks),
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
        hybrid_alpha: float = 1.0,   # weight on predicted_probability
        hybrid_beta: float = 1.0,    # weight on coactivation_score
        hybrid_gamma: float = 1.0,   # weight on normalized access frequency
        hybrid_delta: float = 1.0,   # weight on normalized recency
        hybrid_lambda: float = 1.0,  # weight SUBTRACTED for normalized load cost (bigger/costlier experts score lower, all else equal)
        # Only read when policy="hybrid" - score(expert) = alpha*predicted_probability
        # + beta*coactivation_score + gamma*norm_frequency + delta*norm_recency
        # - lambda*norm_load_cost. Lowest score among resident entries is
        # evicted first. Deliberately NOT auto-tuned here - see
        # scripts/benchmark_streaming.py's --alpha/--beta/--gamma/--delta/--lambda
        # flags and docs/benchmark_findings_current.md for the sweep that
        # picks real values instead of guessing.
        admission_control: bool = False,
        # When True, a prefetch is only admitted if it clears a benefit-vs-
        # cost gate (see should_prefetch()) instead of being admitted
        # unconditionally whenever prefetch() is called - see that method's
        # docstring for the (deliberately simple, not "mathematically
        # perfect") formula.
        admission_margin: float = 1.0,
        # Multiplies the cost side of the admission gate - >1.0 makes
        # admission stricter (fewer, more confident prefetches), <1.0 looser.
    ):
        self.budget_bytes = budget_bytes
        self.policy: EvictionPolicyName = policy
        self.dtype = dtype
        self.stats = stats
        self._tensor_name_fn = tensor_name_fn
        self._shard_index = shard_index
        self._recycle_every_n_loads = recycle_every_n_loads
        self.device = device
        self.hybrid_alpha = hybrid_alpha
        self.hybrid_beta = hybrid_beta
        self.hybrid_gamma = hybrid_gamma
        self.hybrid_delta = hybrid_delta
        self.hybrid_lambda = hybrid_lambda
        self.admission_control = admission_control
        self.admission_margin = admission_margin

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
        if self.policy == "hybrid":
            # Normalization maxima computed ONCE here, not once per candidate
            # inside a per-key scoring function - min(..., key=fn) calls fn
            # once per resident entry, so recomputing three max()-over-all-
            # residents scans inside it made eviction O(n^2) instead of
            # O(n). Measured impact was real, not theoretical: a full
            # benchmark run with this cache-thrashing budget dropped to
            # ~0.6-0.9 tok/s under "hybrid" vs. ~2.6-3.1 tok/s for every
            # other policy on an identical workload, before this fix.
            max_access = max((e.access_count for e in self._resident.values()), default=1) or 1
            max_recency = max((e.last_used_step for e in self._resident.values()), default=1) or 1
            max_size = max((e.size_bytes for e in self._resident.values()), default=1) or 1
            return min(self._resident, key=lambda k: self._hybrid_score(k, max_access, max_recency, max_size))
        # fifo (== "reactive"): oldest-loaded, no notion of recency/frequency at all
        return min(self._resident, key=lambda k: self._resident[k].load_order)

    def _hybrid_score(self, key: ExpertKey, max_access: int, max_recency: int, max_size: int) -> float:
        """score(expert) = alpha*predicted_probability + beta*coactivation_score
        + gamma*norm_frequency + delta*norm_recency - lambda*norm_load_cost.
        Higher = more worth keeping; _choose_victim() picks the MINIMUM, so
        the lowest-scoring resident entry is evicted first. Frequency/
        recency/cost are normalized against the CURRENT resident set (not a
        fixed constant, passed in by the caller so it's computed once per
        eviction decision, not once per candidate) so the weights stay
        meaningful across very different cache sizes and models - an
        access_count of 5 means something different in a 4-expert cache
        than a 400-expert one.
        """
        entry = self._resident[key]
        norm_frequency = entry.access_count / max_access
        norm_recency = entry.last_used_step / max_recency
        norm_load_cost = entry.size_bytes / max_size
        return (
            self.hybrid_alpha * entry.predicted_probability
            + self.hybrid_beta * entry.coactivation_score
            + self.hybrid_gamma * norm_frequency
            + self.hybrid_delta * norm_recency
            - self.hybrid_lambda * norm_load_cost
        )

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
        self._note_ever_loaded(key)

        t0 = time.perf_counter()
        entry = self._load_entry(key)
        self.stats.expert_io_wait_seconds += time.perf_counter() - t0
        entry.access_count = 1
        # Same class of bug as prefetch's access_count fix above, for the
        # "hybrid" policy specifically: a reactive miss means the real
        # router CONFIRMED it needs this expert right now - that's stronger
        # evidence than any prediction, not the ExpertEntry dataclass's 0.0
        # default (which "hybrid" would read as "definitely not needed",
        # the worst possible score, making a just-loaded-because-it-was-
        # NEEDED entry the first thing evicted next). Caught by a unit test
        # before this ever ran on a real model.
        entry.predicted_probability = 1.0
        self._admit(key, entry)
        return entry.weights

    def _note_ever_loaded(self, key: ExpertKey) -> None:
        if key not in self._ever_loaded:
            self._ever_loaded.add(key)
            self.stats.unique_experts_loaded += 1

    def _estimate_avg_expert_bytes(self) -> float:
        if self.stats.unique_experts_loaded == 0:
            return 0.0
        return self.stats.storage_bytes_read / self.stats.unique_experts_loaded

    def _estimate_avg_load_seconds(self) -> float:
        single_loads = max(1, self.stats.reads_count // 3)  # 3 reads (gate/up/down) per expert load
        return self.stats.read_seconds_total / single_loads

    def should_prefetch(self, key: ExpertKey, predicted_probability: float) -> bool:
        """Admission gate (Fase 4 - "não carregar um expert se o benefício
        previsto não justificar uma eviction melhor"). Deliberately simple,
        not mathematically perfect - a real, tunable first version, not a
        placeholder:

          benefit = predicted_probability * avg_observed_reactive_load_seconds
          cost    = (avg_load_seconds again, as the eviction penalty - the
                     cost of having to reload whatever gets evicted to make
                     room, if the cache is already full) + this prefetch's
                     own load cost (avg_load_seconds)
          prefetch only if benefit > cost * admission_margin

        Everything is in the same unit (seconds this run has actually
        measured, not a guess) so the comparison is at least dimensionally
        honest. Before any history exists (first few loads), always admits -
        nothing to judge benefit against yet.
        """
        avg_load_seconds = self._estimate_avg_load_seconds()
        if self.stats.reads_count == 0:
            return True
        benefit = predicted_probability * avg_load_seconds
        avg_expert_bytes = self._estimate_avg_expert_bytes()
        would_evict = bool(self._resident) and (self.resident_bytes + avg_expert_bytes > self.budget_bytes)
        eviction_penalty = avg_load_seconds if would_evict else 0.0
        cost = eviction_penalty + avg_load_seconds
        return benefit > cost * self.admission_margin

    def prefetch(
        self, layer_idx: int, expert_ids: list[int],
        predicted_probabilities: dict[int, float] | None = None,
        coactivation_scores: dict[int, float] | None = None,
    ) -> None:
        """Speculative, off-the-critical-path load - may turn out useful or
        wasted, resolved later (on use or eviction). `predicted_probabilities`/
        `coactivation_scores` (optional, expert_idx -> score) feed the
        "hybrid" eviction rule (see _hybrid_score) and the admission-control
        gate (see should_prefetch) - every other policy ignores them, so
        passing None/{} (the default) preserves the exact prior behavior:
        unconditional admission, no per-entry scores tracked.
        """
        for expert_idx in expert_ids:
            key = (layer_idx, expert_idx)
            predicted_probability = (predicted_probabilities or {}).get(expert_idx, 1.0)
            if key in self._resident:
                # already resident - still worth refreshing its scores, so a
                # "hybrid" eviction decision made a few steps from now uses
                # the current prediction, not a stale one from when this
                # expert first entered the cache.
                if self.policy == "hybrid":
                    entry = self._resident[key]
                    entry.predicted_probability = predicted_probability
                    entry.coactivation_score = (coactivation_scores or {}).get(expert_idx, entry.coactivation_score)
                continue  # already resident - not a new prefetch
            if self.admission_control:
                self.stats.admission_checks += 1
                if not self.should_prefetch(key, predicted_probability):
                    self.stats.admission_rejected += 1
                    continue
            entry = self._load_entry(key)
            entry.is_pending_prefetch = True
            # Matches get()'s baseline, not the ExpertEntry dataclass's 0
            # default: without this, under LFU a fresh prefetch has the
            # lowest possible access_count and becomes the FIRST eviction
            # candidate, before it's ever had a chance to be used - verified
            # empirically (eai_coactivation_lfu: 144/144 prefetches wasted,
            # 0 useful, byte-for-byte identical hit_rate to plain lfu with
            # no prefetching at all - LFU was evicting every single
            # prefetch immediately). A prefetch shouldn't start out WORSE
            # than a reactively-loaded entry just because of how it entered
            # the cache.
            entry.access_count = 1
            if self.policy == "hybrid":
                entry.predicted_probability = predicted_probability
                entry.coactivation_score = (coactivation_scores or {}).get(expert_idx, 0.0)
            self._note_ever_loaded(key)
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
