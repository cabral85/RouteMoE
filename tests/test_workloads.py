#!/usr/bin/env python
"""Unit tests for eai/workloads.py - pure list-reordering logic, no model
needed. Checks the structural invariant each workload exists to guarantee
(contiguous category runs for domain_clustered, no consecutive repeats for
adversarial_shift, etc.), not just "it runs without crashing".

    python tests/test_workloads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai.workloads import (
    adversarial_shift_workload,
    conversational_workload,
    domain_clustered_workload,
    random_workload,
)

CATEGORIES = ["python", "sql", "csharp", "matematica", "traducao"]
PROMPTS = [
    {"id": f"{c}-{i}", "category": c, "text": f"prompt {c} {i}"}
    for c in CATEGORIES
    for i in range(4)
]


def same_multiset(a: list[dict], b: list[dict]) -> bool:
    return sorted(p["id"] for p in a) == sorted(p["id"] for p in b)


def test_random_is_a_permutation():
    out = random_workload(PROMPTS, seed=1)
    assert same_multiset(out, PROMPTS)
    assert len(out) == len(PROMPTS)
    print("test_random_is_a_permutation: OK")


def test_domain_clustered_runs_are_contiguous():
    out = domain_clustered_workload(PROMPTS, seed=1)
    assert same_multiset(out, PROMPTS)
    seen_categories: list[str] = []
    for p in out:
        if not seen_categories or seen_categories[-1] != p["category"]:
            assert p["category"] not in seen_categories, (
                f"category {p['category']!r} reappeared non-contiguously - domain_clustered must "
                f"keep every prompt from one category in a single unbroken run"
            )
            seen_categories.append(p["category"])
    assert len(seen_categories) == len(CATEGORIES)
    print("test_domain_clustered_runs_are_contiguous: OK")


def test_adversarial_shift_never_repeats_consecutively():
    out = adversarial_shift_workload(PROMPTS, seed=1)
    assert same_multiset(out, PROMPTS)
    for a, b in zip(out, out[1:]):
        assert a["category"] != b["category"], (
            f"adversarial_shift produced consecutive same-category prompts ({a['id']}, {b['id']}) - "
            f"it must cycle through every category before any repeats"
        )
    print("test_adversarial_shift_never_repeats_consecutively: OK")


def test_conversational_is_a_permutation_with_interleaved_sessions():
    # Checked across many seeds, not just one: a single fixed seed could
    # accidentally land on the edge case below and hide it.
    for seed in range(20):
        out = conversational_workload(PROMPTS, session_length=2, num_sessions_interleaved=3, seed=seed)
        assert same_multiset(out, PROMPTS), f"seed={seed} lost or duplicated prompts"

    out = conversational_workload(PROMPTS, session_length=2, num_sessions_interleaved=3, seed=1)
    # Distinct from domain_clustered: at least one category must be split
    # across multiple non-adjacent runs (that's the whole point of
    # interleaving several sessions instead of one long per-category block).
    # NOTE: an individual session's own run is capped at session_length, but
    # two DIFFERENT sessions of the same category can still land back-to-back
    # by chance when one session's slot is refilled right after another
    # slot's same-category session just finished - that's a real, accepted
    # trait of round-robin interleaving, not a bug, so this test does not
    # assert a strict global run-length bound.
    positions: dict[str, list[int]] = {}
    for i, p in enumerate(out):
        positions.setdefault(p["category"], []).append(i)
    non_contiguous = [c for c, idxs in positions.items() if any(b - a > 1 for a, b in zip(idxs, idxs[1:]))]
    assert non_contiguous, "expected at least one category's prompts to be split across non-adjacent runs"
    print("test_conversational_is_a_permutation_with_interleaved_sessions: OK")


def test_workloads_are_deterministic_given_a_seed():
    a = domain_clustered_workload(PROMPTS, seed=7)
    b = domain_clustered_workload(PROMPTS, seed=7)
    assert [p["id"] for p in a] == [p["id"] for p in b], "same seed must produce the same ordering"
    print("test_workloads_are_deterministic_given_a_seed: OK")


if __name__ == "__main__":
    test_random_is_a_permutation()
    test_domain_clustered_runs_are_contiguous()
    test_adversarial_shift_never_repeats_consecutively()
    test_conversational_is_a_permutation_with_interleaved_sessions()
    test_workloads_are_deterministic_given_a_seed()
    print("\nOK - all workload unit tests passed")
