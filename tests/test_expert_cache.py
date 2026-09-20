#!/usr/bin/env python
"""Unit tests for eai/expert_cache.py, isolated from any real model - a
synthetic shard index stands in for the checkpoint, so this validates the
cache/eviction *logic* (budget enforcement, FIFO/LRU/LFU correctness,
prefetch useful/wasted bookkeeping) in milliseconds, before ever touching a
real model. Per the phase plan: don't trust a metric until it's validated at
the smallest unit that can test it.

    python tests/test_expert_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.expert_cache import BenchmarkStats, GlobalExpertCache

EXPERT_BYTES = 3 * 4 * 4 * 2  # 3 tensors x (4x4) x bf16(2 bytes) = 96 bytes/expert, tiny and exact


class FakeShardIndex:
    """Every expert's weights are a distinct (4,4) tensor - real bytes, real
    tensor identity, but no disk I/O and no model needed."""

    def get_tensor(self, name: str) -> torch.Tensor:
        # name encodes (layer, expert, proj) - unique per call, deterministic
        return torch.zeros(4, 4, dtype=torch.bfloat16) + hash(name) % 7


def tensor_name_fn(layer_idx: int, expert_idx: int, proj: str) -> str:
    return f"L{layer_idx}.E{expert_idx}.{proj}"


def make_cache(policy: str, budget_experts: int) -> GlobalExpertCache:
    stats = BenchmarkStats()
    return GlobalExpertCache(
        shard_index=FakeShardIndex(),
        budget_bytes=budget_experts * EXPERT_BYTES,
        policy=policy,
        dtype=torch.bfloat16,
        stats=stats,
        tensor_name_fn=tensor_name_fn,
    )


def test_budget_enforced():
    cache = make_cache("lru", budget_experts=2)
    cache.get(0, 0)
    cache.get(0, 1)
    assert len(cache.resident_keys) == 2, "budget of 2 experts should hold exactly 2"
    cache.get(0, 2)
    assert len(cache.resident_keys) == 2, "adding a 3rd expert over a 2-expert budget must evict one"
    assert cache.resident_bytes <= cache.budget_bytes
    print("test_budget_enforced: OK")


def test_fifo_evicts_oldest_loaded():
    cache = make_cache("fifo", budget_experts=2)
    cache.get(0, 0)  # loaded 1st
    cache.get(0, 1)  # loaded 2nd
    cache.get(0, 0)  # re-access - FIFO does NOT care about recency, only load order
    cache.get(0, 2)  # loaded 3rd -> should evict expert 0 (oldest LOAD, despite being just re-accessed)
    assert (0, 0) not in cache.resident_keys, "FIFO must evict by load order, ignoring recent access"
    assert (0, 1) in cache.resident_keys
    assert (0, 2) in cache.resident_keys
    print("test_fifo_evicts_oldest_loaded: OK")


def test_lru_evicts_least_recently_used():
    cache = make_cache("lru", budget_experts=2)
    cache.get(0, 0)
    cache.get(0, 1)
    cache.get(0, 0)  # touch expert 0 again -> now expert 1 is least-recently-used
    cache.get(0, 2)  # should evict expert 1, not expert 0
    assert (0, 1) not in cache.resident_keys, "LRU must evict the least-recently-touched expert"
    assert (0, 0) in cache.resident_keys
    assert (0, 2) in cache.resident_keys
    print("test_lru_evicts_least_recently_used: OK")


def test_lfu_evicts_least_frequently_used():
    cache = make_cache("lfu", budget_experts=2)
    cache.get(0, 0)
    cache.get(0, 1)
    cache.get(0, 0)
    cache.get(0, 0)  # expert 0 now has access_count=3, expert 1 has access_count=1
    cache.get(0, 2)  # should evict expert 1 (fewest accesses), not expert 0
    assert (0, 1) not in cache.resident_keys, "LFU must evict the least-frequently-accessed expert"
    assert (0, 0) in cache.resident_keys
    print("test_lfu_evicts_least_frequently_used: OK")


def test_prefetch_useful_when_subsequently_used():
    cache = make_cache("lru", budget_experts=4)
    cache.prefetch(0, [5])
    assert cache.stats.prefetched_experts == 1
    assert cache.stats.useful_prefetches == 0
    cache.get(0, 5)  # the prefetch pays off
    assert cache.stats.useful_prefetches == 1
    assert cache.stats.wasted_prefetches == 0
    assert cache.stats.cache_hits == 1  # the get() was a hit, since prefetch already loaded it
    print("test_prefetch_useful_when_subsequently_used: OK")


def test_prefetch_wasted_when_evicted_unused():
    cache = make_cache("lru", budget_experts=1)
    cache.prefetch(0, [5])  # loaded speculatively
    cache.get(0, 6)  # a different expert needed - budget=1 forces eviction of expert 5, never used
    assert cache.stats.wasted_prefetches == 1
    assert cache.stats.useful_prefetches == 0
    print("test_prefetch_wasted_when_evicted_unused: OK")


def test_prefetch_pending_at_end_counts_as_wasted_after_finalize():
    cache = make_cache("lru", budget_experts=4)
    cache.prefetch(0, [5])  # never actually used, never evicted either - run just ends
    assert cache.stats.wasted_prefetches == 0  # not yet resolved
    cache.finalize()
    assert cache.stats.wasted_prefetches == 1
    print("test_prefetch_pending_at_end_counts_as_wasted_after_finalize: OK")


def test_global_budget_spans_layers():
    """A 2-expert-worth budget must be shared ACROSS layers, not per-layer -
    the whole point of this module vs. the old per-layer dict caches."""
    cache = make_cache("lru", budget_experts=2)
    cache.get(0, 0)  # layer 0
    cache.get(1, 0)  # layer 1 - different layer, same expert_idx=0, must count as a DIFFERENT key
    assert len(cache.resident_keys) == 2
    cache.get(2, 0)  # layer 2 - a 3rd distinct key, over budget -> must evict
    assert len(cache.resident_keys) == 2, "budget must be enforced globally across layers, not per-layer"
    print("test_global_budget_spans_layers: OK")


def test_reload_counted_distinct_from_cold_miss():
    cache = make_cache("lru", budget_experts=1)
    cache.get(0, 0)  # cold miss - first time ever
    assert cache.stats.expert_reloads == 0
    cache.get(0, 1)  # evicts expert 0 (budget=1)
    cache.get(0, 0)  # expert 0 needed again - this is a RELOAD, not a fresh cold miss
    assert cache.stats.expert_reloads == 1
    assert cache.stats.cache_misses == 3  # all three get() calls were misses (nothing was ever a hit here)
    print("test_reload_counted_distinct_from_cold_miss: OK")


if __name__ == "__main__":
    test_budget_enforced()
    test_fifo_evicts_oldest_loaded()
    test_lru_evicts_least_recently_used()
    test_lfu_evicts_least_frequently_used()
    test_prefetch_useful_when_subsequently_used()
    test_prefetch_wasted_when_evicted_unused()
    test_prefetch_pending_at_end_counts_as_wasted_after_finalize()
    test_global_budget_spans_layers()
    test_reload_counted_distinct_from_cold_miss()
    print("\nOK - all expert_cache unit tests passed")
