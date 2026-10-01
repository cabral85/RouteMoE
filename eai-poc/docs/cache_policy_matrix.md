# Cache policy matrix — cache-policy-optimization phase

Status: **in progress**. OLMoE-validated (small scale, 3-4 prompts, single
budget points). Not yet run at Qwen3-30B scale, not yet run across the full
budget/workload/scenario grid this phase's spec asks for (Partes 6-9). This
document is the running scoreboard for that larger effort; see
[oracle_horizon_findings.md](oracle_horizon_findings.md) for the Oracle-N
deep dive and [benchmark_findings_current.md](benchmark_findings_current.md)
for everything from the prior (Round 7) phase this one builds on.

## What was built this phase

- **`oracle_1`/`oracle_2`/`oracle_4`/`oracle_8`** (`scripts/benchmark_streaming.py`)
  - N future steps' ground-truth experts, prefetched as one window. Plain
    `oracle` kept as an alias for `oracle_1`.
- **`hybrid`** (`eai/expert_cache.py`'s new `"hybrid"` eviction policy +
  `scripts/benchmark_streaming.py`'s matching prefetch branch) - scores
  every resident expert as `alpha*predicted_probability +
  beta*coactivation_score + gamma*norm_frequency + delta*norm_recency -
  lambda*norm_load_cost`, evicts the lowest-scoring one. Weights are real
  CLI flags (`--alpha --beta --gamma --delta --lambda`), default 1.0 each,
  not tuned in this doc yet beyond the 5-point sweep below.
- **`hybrid_adaptive`** - same scoring, but `alpha/beta/gamma/delta/lambda`
  are recomputed every step from the current (cache_budget vs. one-step
  working-set) ratio and cold/warm scenario, via explicit if/else rules
  (no ML) in `hybrid_adaptive_weights()`. Regime logged at step 0 of every
  replay (`[hybrid_adaptive] regime=... alpha=... ...`).
- **Admission control** (`GlobalExpertCache.should_prefetch()`,
  `--admission-control`/`--admission-margin`) - `benefit = predicted_probability
  * avg_observed_load_seconds`, `cost = (eviction_penalty if the cache is
  full) + this prefetch's own load cost`, admits only if `benefit > cost *
  admission_margin`. Off by default (`admission_control=False`) - zero
  behavior change unless explicitly turned on.
- New metrics, standardized into `BenchmarkStats`/every JSONL event:
  `eviction_churn`, `reload_amplification`, `useful_io_ratio`,
  `unique_experts_loaded`, `admission_checks`/`admission_rejected`/
  `admission_rejection_rate`, and (this run) `alpha`/`beta`/`gamma`/`delta`/
  `lambda`/`admission_control`/`admission_margin` themselves, so a sweep's
  own weight combination is traceable from the output file alone.

## Two real bugs found and fixed while validating this (before any of the
   above numbers could be trusted)

1. **`hybrid`'s eviction scoring was O(n²) per eviction**, recomputing
   frequency/recency/cost normalization maxima once per *candidate* inside
   the `min(...)` scan instead of once per eviction decision. Measured
   impact was severe, not theoretical: a real benchmark run dropped
   `hybrid` to ~0.6-0.9 tok/s vs. ~2.5-3.1 tok/s for every other policy on
   an identical workload, before the fix (computing the maxima once,
   outside the per-candidate scoring function).
2. **A reactively-loaded expert (via `get()`, not `prefetch()`) kept the
   `ExpertEntry` dataclass's `predicted_probability=0.0` default** - under
   `hybrid`, that's the worst possible score, meaning an expert the real
   router just confirmed it needs `RIGHT NOW` was the *first* eviction
   candidate the very next time the cache had to make room. Caught by a
   unit test before ever running on a real model (same class of bug as the
   prior phase's LFU+prefetch `access_count` bug -
   [benchmark_findings_current.md](benchmark_findings_current.md) §7 - a
   reactive/confirmed access should never start out scored worse than a
   speculative one). Fixed: a reactive miss now sets
   `predicted_probability=1.0` (a confirmed need, stronger evidence than
   any prediction).

## Results so far (OLMoE, 3 prompts, 4GB budget, cold, default weights unless noted)

| policy | hit_rate | tok/s |
|---|---|---|
| reactive | 40.8% | 2.54 |
| lru | 45.1% | 2.82 |
| lfu | 44.0% | 2.80 |
| eai_coactivation | 45.3% | 2.89 |
| hybrid (alpha=beta=gamma=delta=lambda=1.0) | 42.8% | 2.54 |
| hybrid_adaptive | 39.2% | 2.35 |
| oracle_1 (= oracle) | 61.5% | 6.00-6.39 |
| oracle_2 | 61.4% | 6.25 |
| oracle_4 | 56.3% | 4.69 |
| oracle_8 | 45.3% | 2.91 |

**Default-weight `hybrid` does not yet beat `lru`/`lfu`/`eai_coactivation`**
at this one budget/workload point - consistent with the spec's own expected
outcome ("Não fixe arbitrariamente os pesos sem benchmark" was right to
warn against trusting 1.0-each). `hybrid_adaptive`'s heuristic regime
selection (see log: `tiny_cache+cold` for this budget, since 4GB is close
to what one step's own working set needs for OLMoE) currently does *worse*
than plain `hybrid` - its tiny-cache weights (alpha=3.9, beta=2.6 - lean
hard into prediction) run straight into the same "prediction-heavy loses"
pattern the weight sweep below found. Worth revisiting the tiny-cache
regime's weight choices specifically before trusting `hybrid_adaptive`'s
current heuristic.

## Small weight sweep (5 points, same OLMoE/3-prompt/4GB config)

| combo | alpha | beta | gamma | delta | lambda | hit_rate (mean) | tok/s (mean) |
|---|---|---|---|---|---|---|---|
| default | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 42.8% | 2.54 |
| freq_dominant | 0.2 | 0.2 | 3.0 | 1.0 | 0.5 | 45.8% | 2.79 |
| predict_dominant | 3.0 | 2.0 | 0.2 | 0.2 | 0.5 | **39.1%** | **2.35** |
| cost_penalized | 1.0 | 1.0 | 1.0 | 1.0 | 3.0 | 42.8% | 2.57 |
| **recency_dominant** | 0.2 | 0.2 | 0.5 | 3.0 | 0.5 | **46.1%** | **2.90** |

**Consistent with every other finding in this whole project**: weighting
toward *passive, observed-usage signals* (recency, frequency) beats
weighting toward *prediction* (predicted_probability, coactivation_score).
`predict_dominant` is the worst combination tested, `recency_dominant` the
best - at this one budget/workload point. `cost_penalized` barely moves
anything, plausibly because OLMoE's experts are all nearly the same size
(load-cost normalization has little to differentiate) - worth re-testing on
a model with more expert-size variance, or just accepting cost weighting
matters less for same-shape-expert architectures.

## Answering this phase's central questions (partial - small-scale only so far)

1. **Hybrid vence LFU?** Not yet, at default weights, at this one point.
   `recency_dominant` (0.2/0.2/0.5/3.0/0.5) comes closest (46.1% vs. lfu's
   44.0%) but this is one budget/workload/prompt-count combination, not
   enough to call it a real win.
2. **Em quais budgets?** Untested beyond 4GB.
3. **Em cold ou warm?** Only cold tested so far - warm is exactly where the
   prior phase's biggest finding lives (lfu beating Oracle), so this is the
   most important untested cell.
4. **Oracle multi-step vence LFU?** No - see
   [oracle_horizon_findings.md](oracle_horizon_findings.md): a wider Oracle
   horizon makes it *worse* under a fixed budget (oracle_8 nearly collapses
   to reactive's hit_rate, wasting ~75% of its own prefetches), not better.
5. **Quanto horizonte futuro é suficiente?** 1-2 steps is already near the
   ceiling for this budget/model; 4+ actively hurts.
6. **Admission control reduz churn?** Built and unit-tested (rejects
   near-zero-benefit prefetches under real cache pressure), not yet
   measured on a real model run.
7. **Coactivation continua útil no 30B?** Not re-tested with `hybrid` at
   that scale yet.
8-12. **Not yet answered** - need the larger sweeps (Partes 6-9 of this
   phase's plan) this document doesn't cover yet.

## What's not yet done (this phase's remaining scope)

- Warm-scenario hybrid/hybrid_adaptive runs (the single most informative
  untested cell, given where the prior phase's headline finding lives).
- Qwen3-30B validation of everything above.
- A real, larger weight/admission-control sweep (this was 5 points, one
  budget, cold-only, 3 prompts - a start, not a conclusion).
- Fase 6 (larger profiling set), Fase 9 (full Qwen3-30B benchmark), Fase 10
  (async I/O), Fase 11 (storage layout experiment) - not started.
