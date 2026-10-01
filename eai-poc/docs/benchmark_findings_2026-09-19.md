# Benchmark findings — 2026-09-19 session (archived)

This document consolidates everything learned during the first full pass at
building and running `scripts/benchmark_streaming.py` (the EAI vs. simple-
caching vs. Oracle comparison harness), before a mid-session fix to the
expert-loading I/O layer changed the timing characteristics of every measurement
taken up to that point. All raw data referenced below now lives under
`artifacts/benchmark/archive_2026-09-19_pre_io_fix/` (gitignored, kept locally
for reference — **not republished/reused as-is**, see "Why this is archived"
at the bottom).

**Read this if**: you want the reasoning behind a design decision made this
session, or want to know what was already tried before extending the
benchmark further. **Don't read this as**: current, trustworthy performance
numbers — the timing-dependent ones (tok/s, io_wait_ms) are stale (see below).

---

## 1. What was built

- `eai/expert_cache.py` — `GlobalExpertCache`: one byte-budgeted cache shared
  across every MoE layer, pluggable eviction policy (fifo/lru/lfu),
  prefetch bookkeeping (useful vs. wasted), full instrumentation
  (`BenchmarkStats`).
- `eai/chunked_expert_loader.py` — `ExpertShardIndex` extended so
  `ChunkedExpertBlock`/`ChunkedQwen2MoeExperts` can optionally delegate
  residency to a `GlobalExpertCache` instead of their own unbounded
  per-layer dict (zero regression risk to earlier rounds' results).
- `scripts/benchmark_streaming.py` — the two-stage (trace, then replay)
  benchmark runner: 8 policies (`reactive`, `lru`, `lfu`, `eai`,
  `eai_lookahead_{1,2,4,8}`, `eai_coactivation`, `oracle`), fixed cache
  budgets, 4 workload orderings, warm/cold cache scenarios, a
  memory-headroom guard, a per-step router-correctness gate.
- `eai/workloads.py` — `random`, `domain_clustered`, `conversational`,
  `adversarial_shift` prompt orderings.
- `scripts/analyze_benchmark.py` — tables + 8 plot types from a JSONL event
  file, with structural rules baked in (never compares across budgets, labels
  Oracle as an upper bound not a candidate, flags negative results, flags
  timing that looks contaminated by concurrent machine load).
- `scripts/validate_benchmark_invariants.py` — automated gate: router-
  correctness, budget-never-exceeded, Oracle-never-worse-than-baseline,
  bare-Oracle-zero-waste, run before trusting any sweep.
- `scripts/run_full_sweep.py` — drives the full (budget × workload ×
  scenario) grid via subprocess, with `--resume` support.
- `scripts/dense_layer_streaming_experiment.py` — the same load/use/evict
  cycle applied to a DENSE model's transformer blocks, to test whether the
  MoE-expert chunking idea generalizes.

## 2. Core bug fixed before any of this data was trustworthy

`replay_policy()`'s stage-1→stage-2 index alignment was off by one:
`selected_experts[i]` records the experts used to *generate* token `i`
(i.e. from processing token `i-1`), not the experts that will be used when
token `i` is fed back in. Fixed to align on `[step+1]` and bound replay to
`num_steps - 1` steps. A router-correctness gate (compares real router
selections during replay against the stage-1 ground truth every single step,
raises `RuntimeError` on any mismatch) was added specifically so this class
of bug can never again produce silently-wrong numbers — it ran on every
result below with **zero mismatches** in every run, including at Qwen3-30B's
full 48-layer/128-expert scale.

## 3. Findings that remain valid (mechanism-level, not timing-dependent)

These depend only on which experts were resident when, not on wall-clock
speed — unaffected by the I/O fix below.

- **The router is provably never altered by any caching policy.**
  `expert_accesses` was identical across every policy for the same
  (prompt, budget) in every run — the cache/prefetch layer only ever changes
  *when* weights are resident, never what the router picks.
- **Oracle never underperforms a non-prefetching policy at the same budget**
  (hit_rate), and a bare (lookahead=0) Oracle always has exactly zero wasted
  prefetches — both hold by construction, verified as hard invariants across
  every (budget, prompt) group tested.
- **Plain `eai` (raw `stored_top_k=16` candidate window) is net-negative**
  vs. `reactive`/`lru`/`lfu` at small-to-mid budgets on OLMoE. Root cause:
  the predictor's candidate window (16/layer) is 2x the real router's
  `top_k=8`, so prefetching all of it evicts things that would've been
  reused naturally, for low precision (~9-20% useful).
- **`eai_coactivation` (Fase 7) fixes a meaningful chunk of that**: instead
  of prefetching all `stored_top_k` candidates, it greedily builds a
  `top_k`-sized clique from the cluster's `(stored_top_k × stored_top_k)`
  coactivation matrix (experts that fire *together*, not just individually
  frequent). At 4GB budget it beat plain `eai` by a wide margin (e.g.
  54.1%/50.7%/56.6% hit_rate vs. `eai`'s 44.4%/48.9%/37.5% on the same three
  prompts) and became competitive with `lru`/`lfu`. At 2GB it was still weak
  (10-18%) — the benefit is budget-dependent, headroom-limited.
- **The oracle-vs-reactive hit_rate gap shrinks as budget approaches the
  full model size** (OLMoE total ≈ 13.8GB), confirmed with real sweep data:

  | budget | reactive | oracle | gap |
  |---|---|---|---|
  | 2GB | 24.4% | 53.9% | 29.5pp |
  | 4GB | 40.8% | 65.5% | 24.7pp |
  | 6GB | 50.5% | 66.3% | 15.9pp |
  | 8GB | 57.7% | 66.2% | 8.6pp |

  This is the concrete evidence behind the conclusion: **EAI only has value
  in the under-provisioned regime** (budget < model size). Once the budget
  comfortably fits the model, every policy converges and smart caching adds
  nothing (predictor lookup becomes pure overhead).
- **Qwen3-30B validated at real scale** (48 layers, 128 experts): harness
  runs correctly end to end, `oracle` beat `reactive` on both hit_rate
  (45.7% vs 20.8%) and tok/s (2.55 vs 2.10) at a 6GB budget, 1 prompt. Too
  small a sample to trust the magnitude, but proves the mechanism works at
  target scale, not just on the small OLMoE test model.
- **Workload orderings behave as designed**: `domain_clustered` produces
  clean 4-prompt-per-category blocks, `adversarial_shift` never repeats a
  category consecutively, `conversational` interleaves session-length-2
  runs across 3 concurrent sessions, all verified with unit tests (one real
  bug found and fixed: `conversational_workload` was losing prompts when all
  active sessions exhausted in the same round — fixed by restructuring the
  refill loop).
- **Dense-model streaming does NOT show the same win as MoE** (see §5) —
  this finding doesn't depend on I/O speed, it's about relative memory
  behavior, and stands as-is.

## 4. The I/O fix that made everything above's TIMING numbers stale

Investigated two questions from the user: (a) are we actually disposing of
memory after use, given the whole point is streaming; (b) does the MoE
chunked-loading idea generalize to a dense model partitioned like a database
table.

**Root cause found (empirically, not assumed)**: `safetensors.safe_open`
reads via mmap. Even after our own cache correctly drops its Python
reference to an evicted tensor (`gc.collect()` included), the underlying
memory-mapped pages stay resident in the process's *private* working set for
as long as the shard file handle stays open — confirmed with real
measurements (loading+touching 64 experts, then evicting + gc.collect() +
even calling the existing `recycle()`: private memory did NOT drop in one
test run). This is the exact problem
[kimi-k3-in-c](https://github.com/josesilva05/kimi-k3-in-c) sidesteps with
`O_DIRECT`.

Two fixes applied, in order:

1. **`.to(dtype, copy=True)`** in `GlobalExpertCache._load_entry` — turned
   out `.to(dtype)` was a no-op alias (not a copy) whenever the checkpoint's
   own dtype already matched (true for OLMoE/Qwen3, both stored in bf16),
   meaning "evicted" entries were never really our own memory. Combined with
   periodic (`recycle_every_n_loads`) shard-index recycling. This worked but
   needed careful tuning of the recycle frequency, and cost real
   `expert_io_wait_ms` (measured: ~200-300ms → ~3000-6600ms for the same
   OLMoE smoke config) — a genuine, not-hidden cost of doing real disposal
   instead of a free-but-unsafe alias.
2. **Replaced mmap entirely** with plain `seek()`+`readinto()` on a regular
   buffered file handle (`ExpertShardIndex` in `chunked_expert_loader.py`).
   `os.pread` was tried first and rejected — it's POSIX-only, not available
   on Windows. Verified: memory now stays flat (~1.5GB private) across 512
   loads **with periodic recycling fully disabled** — strictly more robust
   than fix #1 (no tuning, no dependency on callers remembering to recycle
   at the right cadence, portable). Speed is in the same ballpark as fix #1
   (~3500-8900ms io_wait for the same smoke config) — real disk-backed
   copies cost real time regardless of which safe mechanism does them; the
   ~15-30x slowdown vs. the original *unsafe* zero-copy mmap alias is the
   honest price of correctness, not a regression introduced by this fix.

**Consequence**: every `io_wait_ms`/`tokens_per_second` number collected
before this fix (i.e. everything in `archive_2026-09-19_pre_io_fix/`) was
measured under an I/O path that was silently NOT disposing memory the way
the benchmark's own budget accounting assumed. Hit-rate/correctness/cache-
mechanism numbers are unaffected (they don't depend on how fast a byte gets
copied) and remain valid per §3. Timing numbers are not comparable to
anything collected after this fix and must not be mixed with new data.

## 5. Dense-model streaming experiment (negative-but-informative)

`scripts/dense_layer_streaming_experiment.py`: same load/use/evict cycle as
the MoE expert loader, applied to whole transformer blocks (`meta` device +
`load_state_dict(assign=True)` per block per forward call), tested on GPT-2
(124M) and Qwen2.5-1.5B-Instruct. Correctness held both times (identical
output tokens to the fully-resident baseline). But:

| model | baseline peak | streaming peak | baseline tok/s | streaming tok/s |
|---|---|---|---|---|
| GPT-2 (124M) | 2.41GB | 2.61GB (+8%) | 16.7 | 2.35 (14%) |
| Qwen2.5 (1.5B) | 9.33GB | 12.93GB (+39%) | 1.84 | 0.69 (38%) |

Streaming used **more** memory in both cases, not less, and cost 2.6-7x
throughput. Cause: this predates the I/O fix in §4 — every block reload was
still going through the same mmap-accumulation problem, at a much higher
rate (every layer reloaded every single generated token, no sparsity to
skip anything). Unlike MoE, a dense model has no predictable subset to
prefetch — every layer is needed every token, unconditionally — so there's
no EAI-style win available even in principle, only the raw
load/use/evict mechanics, which (before the I/O fix) actively hurt. This
experiment has NOT been re-run since the I/O fix; the *conceptual*
conclusion (no sparsity to exploit → no EAI-style win possible, at best a
memory/speed tradeoff) stands regardless, but the exact numbers above should
be re-measured before being cited, since the I/O fix might change them
materially (worth doing before writing off dense streaming as unusable).

## 6. What was running when this was archived

A fresh, full OLMoE sweep (5 budgets × 4 workloads × 2 scenarios = 40 runs,
under the fixed I/O layer) had just started and completed exactly 1/40
combinations (`2GB / random / cold`, 42 clean events) before being killed
per explicit instruction, to reset and start the next phase cleanly. That
one combo's data is archived alongside everything else — not enough on its
own to be useful, but it's the first clean (post-fix) data point collected.

## Why this is archived, not deleted

Every mechanism-level finding in §3 is still true and should inform the next
sweep's design (don't waste budget re-litigating whether Oracle dominates,
whether `eai_coactivation` beats plain `eai`, etc. — build on it). But the
absolute timing numbers throughout are pre-fix and would corrupt any
apples-to-apples comparison if mixed with post-fix data. Keeping the raw
JSONL locally (gitignored, not published) preserves full traceability
without risking silent contamination of the next report.
