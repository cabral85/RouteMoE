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


def make_cache(policy: str, budget_experts: int, **kwargs) -> GlobalExpertCache:
    stats = BenchmarkStats()
    return GlobalExpertCache(
        shard_index=FakeShardIndex(),
        budget_bytes=budget_experts * EXPERT_BYTES,
        policy=policy,
        dtype=torch.bfloat16,
        stats=stats,
        tensor_name_fn=tensor_name_fn,
        **kwargs,
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


def test_lfu_prefetch_not_immediately_self_evicted():
    """Regression test for a real bug: prefetch() didn't set access_count,
    leaving it at the ExpertEntry dataclass's 0 default - the lowest
    possible value, meaning LFU picked every fresh prefetch as its own
    first eviction victim before it could ever be used. Caught by a real
    benchmark run: eai_coactivation_lfu had 144/144 prefetches wasted, 0
    useful, byte-identical hit_rate to plain lfu with no prefetching at
    all."""
    cache = make_cache("lfu", budget_experts=2)
    cache.get(0, 0)  # access_count=1, one real use
    cache.prefetch(0, [1])  # access_count must be >= 1, not 0, or this is evicted next
    cache.get(0, 2)  # a 3rd distinct key over budget=2 forces one eviction
    assert (0, 1) in cache.resident_keys, "a fresh prefetch must not be the FIRST thing LFU evicts, before it's ever had a chance to be used"
    print("test_lfu_prefetch_not_immediately_self_evicted: OK")


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


def test_hybrid_prefers_high_predicted_probability_when_alpha_dominates():
    """With alpha >> every other weight, hybrid should evict the entry with
    the LOWEST predicted_probability first, regardless of frequency/recency."""
    cache = make_cache("hybrid", budget_experts=2, hybrid_alpha=100.0, hybrid_beta=0.0, hybrid_gamma=0.0, hybrid_delta=0.0, hybrid_lambda=0.0)
    cache.get(0, 0)
    cache.get(0, 1)
    # expert 0 accessed many more times than expert 1 (frequency/recency would favor keeping 0)...
    cache.get(0, 0)
    cache.get(0, 0)
    cache.get(0, 0)
    # ...but expert 1 is predicted far more likely to be needed again - alpha dominates, so 1 survives, 0 is evicted
    cache.prefetch(0, [0, 1], predicted_probabilities={0: 0.01, 1: 0.99})
    cache.get(0, 2)  # 3rd distinct key over budget=2, forces one eviction
    assert (0, 1) in cache.resident_keys, "high predicted_probability must protect an entry when alpha dominates the score"
    assert (0, 0) not in cache.resident_keys
    print("test_hybrid_prefers_high_predicted_probability_when_alpha_dominates: OK")


def test_hybrid_penalizes_load_cost_when_lambda_dominates():
    """With lambda >> every other weight, hybrid should evict the BIGGER
    (costlier-to-reload) entry first, regardless of everything else."""
    cache = make_cache("hybrid", budget_experts=3, hybrid_alpha=0.0, hybrid_beta=0.0, hybrid_gamma=0.0, hybrid_delta=0.0, hybrid_lambda=100.0)

    class VariableSizeShardIndex:
        def get_tensor(self, name: str) -> torch.Tensor:
            # expert 1's tensors are deliberately much bigger than 0's or 2's
            size = 8 if ".E1." in name else 4
            return torch.zeros(size, size, dtype=torch.bfloat16)

    cache._shard_index = VariableSizeShardIndex()
    cache.get(0, 0)
    cache.get(0, 1)  # bigger - higher load cost
    cache.get(0, 2)
    # all three fit (budget_experts=3 sized off the SMALL EXPERT_BYTES constant,
    # so the bigger entry may already have forced pressure - budget is generous
    # enough here that eviction only happens on the 4th key)
    cache.get(0, 3)
    assert (0, 1) not in cache.resident_keys, "the biggest (costliest-to-reload) entry must be evicted first when lambda dominates"
    print("test_hybrid_penalizes_load_cost_when_lambda_dominates: OK")


def test_admission_control_rejects_low_benefit_prefetch():
    """Once there's read-time history and the cache is under real pressure,
    a very-low-probability prefetch should be rejected outright rather than
    silently admitted (the whole point of Fase 4's should_prefetch gate)."""
    cache = make_cache("lru", budget_experts=1, admission_control=True, admission_margin=1.0)
    cache.get(0, 0)  # establishes real read-time history (reads_count > 0), and fills the 1-expert budget
    cache.prefetch(0, [1], predicted_probabilities={1: 0.0})  # ~zero predicted benefit, but WOULD cost an eviction to admit
    assert cache.stats.admission_checks == 1
    assert cache.stats.admission_rejected == 1
    assert (0, 1) not in cache.resident_keys, "a near-zero-benefit prefetch under real cache pressure should be rejected, not admitted"
    print("test_admission_control_rejects_low_benefit_prefetch: OK")


def test_admission_control_off_by_default_matches_prior_behavior():
    """admission_control defaults to False - every prefetch is unconditionally
    admitted exactly like before this feature existed, zero regression risk."""
    cache = make_cache("lru", budget_experts=1)
    cache.get(0, 0)
    cache.prefetch(0, [1], predicted_probabilities={1: 0.0})  # would be rejected if admission_control were on
    assert cache.stats.admission_checks == 0
    assert (0, 1) in cache.resident_keys, "with admission_control=False (default), prefetch must remain unconditional"
    print("test_admission_control_off_by_default_matches_prior_behavior: OK")


if __name__ == "__main__":
    test_budget_enforced()
    test_fifo_evicts_oldest_loaded()
    test_lru_evicts_least_recently_used()
    test_lfu_evicts_least_frequently_used()
    test_prefetch_useful_when_subsequently_used()
    test_prefetch_wasted_when_evicted_unused()
    test_prefetch_pending_at_end_counts_as_wasted_after_finalize()
    test_lfu_prefetch_not_immediately_self_evicted()
    test_global_budget_spans_layers()
    test_reload_counted_distinct_from_cold_miss()
    test_hybrid_prefers_high_predicted_probability_when_alpha_dominates()
    test_hybrid_penalizes_load_cost_when_lambda_dominates()
    test_admission_control_rejects_low_benefit_prefetch()
    test_admission_control_off_by_default_matches_prior_behavior()
    print("\nOK - all expert_cache unit tests passed")
