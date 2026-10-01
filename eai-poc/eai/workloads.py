"""Fase 6: workload orderings over the existing prompt set. A cache or
predictor's real-world value depends heavily on HOW prompts arrive, not just
which ones - a predictor that looks great when 20 python prompts arrive back
to back can still be useless on a real, bursty multi-tenant traffic pattern.
Pure list reordering: no model, no I/O, deterministic given a fixed seed -
safe and fast to validate on its own, independent of anything that touches
GPU/CPU memory.

Four required patterns:
  random             - no structure at all; the null hypothesis for locality.
  domain_clustered    - every prompt from one category runs back-to-back
                        before the next category starts - maximum locality,
                        the easiest case for any cluster/coactivation-based
                        predictor to look good on.
  conversational      - several short "conversations" (session_length
                        prompts from the same category) interleaved
                        round-robin - local runs, but a session goes cold
                        for a while before coming back, unlike
                        domain_clustered's one long uninterrupted run.
  adversarial_shift    - never repeats a category on consecutive prompts,
                        cycling through every category before any repeats -
                        the worst case for a locality-assuming cache or
                        predictor, by construction.
"""

from __future__ import annotations

import random

WORKLOAD_NAMES = ["random", "domain_clustered", "conversational", "adversarial_shift"]


def _group_by_category(prompts: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for p in prompts:
        groups.setdefault(p["category"], []).append(p)
    return groups


def random_workload(prompts: list[dict], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = list(prompts)
    rng.shuffle(out)
    return out


def domain_clustered_workload(prompts: list[dict], seed: int = 0) -> list[dict]:
    groups = _group_by_category(prompts)
    categories = list(groups.keys())
    random.Random(seed).shuffle(categories)  # category ORDER randomized; contents within a category keep file order
    out: list[dict] = []
    for c in categories:
        out.extend(groups[c])
    return out


def conversational_workload(
    prompts: list[dict], session_length: int = 2, num_sessions_interleaved: int = 3, seed: int = 0,
) -> list[dict]:
    groups = _group_by_category(prompts)
    rng = random.Random(seed)
    categories = list(groups.keys())
    rng.shuffle(categories)

    sessions: list[list[dict]] = []
    for c in categories:
        items = groups[c]
        for i in range(0, len(items), session_length):
            sessions.append(items[i:i + session_length])
    rng.shuffle(sessions)

    pending = list(sessions)
    active: list[list[dict] | None] = [pending.pop(0) for _ in range(min(num_sessions_interleaved, len(pending)))]
    cursors = [0] * len(active)

    out: list[dict] = []
    while True:
        progressed = False
        for i in range(len(active)):
            if active[i] is None:
                if not pending:
                    continue
                active[i] = pending.pop(0)
                cursors[i] = 0
            out.append(active[i][cursors[i]])
            cursors[i] += 1
            progressed = True
            if cursors[i] >= len(active[i]):
                active[i] = None
        if not progressed:
            break
    return out


def adversarial_shift_workload(prompts: list[dict], seed: int = 0) -> list[dict]:
    groups = _group_by_category(prompts)
    categories = list(groups.keys())
    random.Random(seed).shuffle(categories)
    queues = {c: list(items) for c, items in groups.items()}
    out: list[dict] = []
    while any(queues.values()):
        for c in categories:
            if queues[c]:
                out.append(queues[c].pop(0))
    return out


def build_workload(name: str, prompts: list[dict], seed: int = 0) -> list[dict]:
    if name == "random":
        return random_workload(prompts, seed=seed)
    if name == "domain_clustered":
        return domain_clustered_workload(prompts, seed=seed)
    if name == "conversational":
        return conversational_workload(prompts, seed=seed)
    if name == "adversarial_shift":
        return adversarial_shift_workload(prompts, seed=seed)
    raise ValueError(f"unknown workload {name!r} (choices: {WORKLOAD_NAMES})")
