# Benchmark findings — current round (post I/O fix)

Status: **both sweeps complete.** OLMoE (40/40 combinations, 1680 events,
~5.5h) and Qwen3-30B (4/4 combinations, 32 events, ~32min). Collected under
the corrected I/O layer (`ExpertShardIndex` reading via
`seek()`+`readinto()`, not mmap — see
[docs/benchmark_findings_2026-09-19.md](benchmark_findings_2026-09-19.md) §4
for why that fix happened). Everything in the archived document's §3
(mechanism-level findings: router correctness, `eai` vs `eai_coactivation`,
workload orderings) still holds and isn't re-derived here.

## Raw data

`artifacts/benchmark/sweep_olmoe.jsonl` — 1680 events, 5 budgets (2/4/6/8/10
GB) × 4 workloads (random/domain_clustered/conversational/adversarial_shift)
× 2 scenarios (cold/warm) × 7 policies × 6 prompts. Gitignored; regenerate
via `python scripts/run_full_sweep.py --config olmoe`. Report tables/plots:
`artifacts/benchmark/report_full/`.

## Validation

Full invariant sweep (see `scripts/validate_benchmark_invariants.py`'s
documented rules) checked against all 1680 events:

- `router_correctness_checks > 0` and no correctness mismatch anywhere
  (would have raised `RuntimeError` and aborted the run) — **0 failures**.
- `peak_resident_bytes <= cache_budget_bytes` in every single event —
  **0 failures**, budget enforcement holds under the new I/O path too.
- `hit_rate`/etc. in `[0,1]` — **0 failures**.
- Bare-Oracle-zero-wasted-prefetches — **0 failures**.
- `expert_accesses` identical across every policy within a (prompt, budget,
  workload, scenario) group — **0 failures**, router is provably
  policy-independent throughout.
- Oracle hit_rate >= reactive/lru/lfu: **0/360 violations in `cold`**,
  **37/360 (10.3%) violations in `warm`** — see §2, this is a genuine
  finding, not a bug (root-caused below).

## 1. The budget-vs-gap pattern, confirmed at full scale

(Combined across all workloads/scenarios; see `report_full/` for
workload-specific breakdowns.)

| budget | reactive hit_rate | oracle hit_rate | gap |
|---|---|---|---|
| 2GB | 23.3% | 54.0% | 30.7pp |
| 4GB | 40.4% | 65.5% | 25.1pp |
| 6GB | 53.1% | 68.4% | 15.3pp |
| 8GB | 59.1% | 70.9% | 11.8pp |
| 10GB | 70.4% | 75.2% | 4.8pp |

Same shape as the archived (stale-timing) data, now backed by the full grid
instead of 3-6 sample points: **the gap between "no prediction at all" and
"perfect prediction" collapses as the budget approaches the model's full
size** (OLMoE ≈ 13.8GB). Confirms: EAI-style prediction only has room to add
value in the under-provisioned regime.

## 2. NEW finding: a warm cache changes which policy wins, and by how much

This is the biggest new result from the full sweep — the smaller earlier
sample never had enough warm-scenario coverage to see it clearly.

| scenario | budget | reactive | lru | lfu | eai_coactivation | oracle |
|---|---|---|---|---|---|---|
| cold | 4GB | 39.0% | 41.9% | 42.2% | 43.2% | 64.4% |
| warm | 4GB | 41.8% | 44.0% | **54.2%** | 45.2% | 66.5% |
| cold | 6GB | 48.7% | 51.2% | 51.0% | 51.8% | 65.5% |
| warm | 6GB | 57.5% | 57.0% | **67.4%** | 57.6% | 71.3% |
| cold | 8GB | 56.3% | 57.3% | 57.3% | 58.1% | 65.5% |
| warm | 8GB | 61.8% | 68.2% | **77.4%** | 69.9% | **76.3%** |

At 8GB warm, **`lfu` (77.4%) actually beats Oracle (76.3%)** — the simplest
possible frequency-counting policy, with zero prediction, outperforms a
policy that gets to see the immediate future perfectly.

**Root cause** (verified with actual counters, not just hit_rate): in a
warm run, the same cache persists across all 6 prompts for a given policy.
`lfu`'s eviction rule never evicts proactively — it only evicts reactively,
under real memory pressure, and always picks the least-frequently-used
victim. Anything that's stayed popular across the whole run (all 6 prompts)
is naturally protected. Oracle, by contrast, only knows the very next step's
real need (`lookahead=0`) and prefetches exactly that every step — which
means it keeps evicting things to make room for what's needed *right now*,
even when what gets evicted would have been useful again a few prompts
later. Confirmed on one specific violating case
(`matematica-test-04`, 10GB, `random`, `warm`):

| policy | hit_rate | evictions | cache_misses | prefetched |
|---|---|---|---|---|
| reactive | 88.4% | 246 | 246 | 0 |
| lfu | 92.9% | 151 | 151 | 0 |
| oracle | 85.9% | **367** | 299 | 68 |

Oracle evicted 2.4x more than `lfu` on the exact same prompt sequence,
despite having "perfect" per-step knowledge — its knowledge is scoped to one
step at a time, not to the whole session, so it has no way to know an expert
it's about to evict will be needed again three prompts later. `lfu`'s
frequency count is implicitly a (crude, but apparently effective) long-
horizon memory that Oracle-as-implemented-here (lookahead 0-4, always
prompt-relative) doesn't have.

**Practical implication**: in a realistic serving setting (a session that
keeps running across many requests, i.e. warm, not a cold cache reset per
request), a well-tuned LFU-style cache may already capture most of the
achievable benefit, and EAI's marginal value over *just caching well* shrinks
further than the cold-scenario numbers alone would suggest. This is a
genuinely important scoping result for where EAI is (and isn't) worth
building.

## 3. `eai` / `eai_coactivation` at full scale

Consistent with the earlier smaller sample: plain `eai` is still net-negative
vs. `reactive` at low budgets (2GB: 8.2% vs 23.3% cold; 4GB: 32.3% vs 39.0%
cold), crosses over to positive only at higher budgets (8-10GB, where every
policy is converging toward the ceiling anyway). `eai_coactivation`
consistently beats plain `eai` at every budget tested, and is competitive
with (sometimes ahead of) `lru`/`lfu` in the cold scenario - but doesn't
show the same dramatic warm-scenario jump `lfu` does, since coactivation's
prefetching has the same "evicts to serve right now" property as oracle,
just less aggressively.

## 4. First GPU (CUDA) measurement

Everything above ran entirely on CPU/system RAM - `eai/chunked_expert_loader.py`
and `eai/expert_cache.py` gained a `device` parameter (`"cpu"`/`"cuda"`);
`scripts/gpu_expert_streaming_experiment.py` is the first script to actually
place the backbone and streamed experts on a GPU, tested on this machine's
RTX 5060 Laptop (8.5GB VRAM). OLMoE, 2GB expert budget, 16 tokens:

| | CPU | GPU |
|---|---|---|
| avg read+transfer per tensor | 2.237ms | 2.225ms |
| tokens/second | 2.08 | 1.83 (88%) |
| peak memory | 2.20GB (RAM) | **2.99GB VRAM** |

Correctness: identical output tokens on both runs.

**The disk-read cost is the same whether the destination is RAM or VRAM**
(2.237ms vs 2.225ms, within noise) - confirms the prediction from reasoning
alone: PCIe transfer bandwidth (multiple GB/s) is nowhere near the
bottleneck compared to disk read speed (~2.6GB/s measured earlier), so
adding a GPU tier doesn't change the fundamental "cold start per expert"
cost at all - it's still disk-bound. **GPU tok/s was actually slightly
worse than CPU (88%)** for this specific workload (batch=1, single-token
decode, tiny 1B-active-param model) - there isn't enough parallel work per
step to amortize CUDA kernel-launch/sync overhead, and since the bottleneck
is disk I/O either way, GPU compute speed was never going to be what
decides this comparison. Peak VRAM (2.99GB) fit comfortably in 8.5GB,
confirming the " backbone + a modest expert budget" story holds for GPU too.

**Interpretation for the original question** ("30B on a 12GB GPU"): moving
the destination from RAM to VRAM doesn't change the memory-fit story
(same byte-budget logic applies, `GlobalExpertCache` already supports it
via `device="cuda"`) and doesn't meaningfully change per-expert cold-start
latency either (disk-bound in both cases) - the place a GPU's speed WOULD
matter is the actual matmul/compute time once a token's needed experts are
resident, which this tiny-model, tiny-batch measurement wasn't big enough
to show clearly. Worth re-measuring on a larger model (Qwen1.5-MoE or
Qwen3-30B, VRAM allowing) and/or a larger batch, where GPU compute
parallelism has more work to actually pay for itself.

## 5. Qwen3-30B sweep - the real target-scale numbers

`artifacts/benchmark/sweep_qwen3_30b.jsonl` - 32 events, 2 budgets (6/10GB)
x 2 workloads (domain_clustered/adversarial_shift), cold only, 4 policies
(reactive/lru/eai_coactivation/oracle), 2 prompts. All invariants held
(0 failures): router correctness, budget enforcement, oracle >= reactive/lru
hit_rate at both budgets, bare-oracle zero waste.

| budget | reactive | lru | eai_coactivation | oracle |
|---|---|---|---|---|
| 6GB | 11.9% / 0.62 tok/s | 14.2% / 0.69 | 13.0% / 0.63 | 23.9% / 1.64 |
| 10GB | 19.1% / 0.86 tok/s | 20.8% / 0.95 | 20.5% / 0.89 | 27.9% / 2.42 |

(cells are hit_rate / tokens-per-second)

Same qualitative shape as OLMoE - oracle clearly ahead of every real policy,
gap narrowing somewhat from 6GB to 10GB - but at a much smaller absolute
scale (Qwen3-30B has 48 layers x 128 experts vs. OLMoE's 16x64, so the same
byte budget covers a much smaller fraction of the full model: even 10GB is
small relative to the ~57GB full expert weight size). `eai_coactivation`
sits between `reactive` and `lru` here rather than clearly ahead of both, on
this small (2-prompt) sample - not enough data yet to call this a real
divergence from the OLMoE pattern vs. sampling noise; would need more
prompts to say confidently. tok/s at this scale is far slower than OLMoE
(sub-1 to ~2.4 tok/s vs. OLMoE's several tok/s) - expected: ~3x the layers,
~2x the experts/layer/token to fetch on every miss, same disk speed.

### Getting this sweep to run at all required two real engineering fixes

The first two attempts aborted (memory guard correctly stopped them, not a
crash) - root-caused with actual instrumentation, not assumption, before
fixing:

1. **Stage 1's per-block cache was unbounded WITHIN one prompt's
   generation** (a design choice that was fine for OLMoE's 1024-expert
   space, never revisited for a 6144-expert model). Measured directly: one
   Qwen3-30B prefill alone - before generating a single token - touched
   2013 distinct experts and pulled 19GB resident, growing to 31GB by the
   6th generated token. Fixed: `_ExpertCache.evict_lru(max_resident_per_layer)`
   (already existed, just never called here) now runs after every forward
   pass in `generate_and_trace()`. Confirmed safe for correctness because
   eviction never touches what the real router computes (see §3 of the
   archived doc) - the resulting trace is bit-for-bit the policy question,
   unaffected. Result: `resident_bytes` flat at 10.87GB across a full
   6-step generation, instead of growing without bound.
2. **A long prompt still spiked memory before the fix above could act**,
   because eviction only runs BETWEEN forward calls and prefill was one
   single call covering the whole prompt - a 27-word prompt alone dropped
   free RAM from ~38GB to ~7.6GB, worse than a 10-word prompt's ~14GB, i.e.
   the peak scaled with prompt length as expected. Fixed with chunked
   prefill (`PREFILL_CHUNK_SIZE = 8` tokens at a time through the same
   growing KV cache, evicting between chunks - the same technique real
   inference engines call "chunked prefill"). Verified numerically
   equivalent to single-shot prefill first (OLMoE, both bf16 and float32 -
   float32 max abs logit difference ~1e-5, pure floating-point
   non-associativity, argmax always identical - same class of noise this
   project already documented for other batching differences, not a new
   bug). Result: same prompt's post-prefill free RAM improved to ~18.5GB
   and stayed flat through generation.

**One real bug found and fixed in the process**: after adding chunked
prefill to stage 1 (`generate_and_trace`) but not stage 2 (`replay_policy`),
the router-correctness gate correctly caught a real inconsistency - 17/240
mismatches, because the two stages were now computing prefill differently
(chunked vs. single-shot), and bf16's non-associativity was occasionally
enough to flip a close top-k pick between the two paths. Fixed by unifying
both stages on the same `PREFILL_CHUNK_SIZE` module constant, so they can
never drift apart again. Re-ran clean: 0/240 mismatches.

## 6. First GPU (CUDA) support added

`eai/chunked_expert_loader.py` and `eai/expert_cache.py` gained a `device`
parameter (`"cpu"`/`"cuda"`) - see §4 above for the first real measurement
(OLMoE, RTX 5060 Laptop 8.5GB VRAM): disk read cost identical whether the
destination is RAM or VRAM, GPU tok/s actually slightly worse than CPU for
this tiny/batch=1 workload (not enough parallel work to amortize kernel
overhead), correctness verified identical to CPU. Not yet wired into
`scripts/benchmark_streaming.py`'s own CLI (`--device` flag) or extended to
a 3-tier VRAM->RAM->disk fallback - both real, larger follow-ups, not done
yet.

## What's next

- Extend the GPU experiment to a bigger model / batch size (Qwen1.5-MoE or
  Qwen3-30B, VRAM allowing), where GPU compute parallelism has enough work
  to show a real speed difference instead of being dominated by disk I/O
  either way.
- Wire `--device` into `scripts/benchmark_streaming.py` itself so the full
  policy sweep can run on GPU, not just the standalone experiment script.
- More Qwen3-30B prompts before treating §5's `eai_coactivation` vs.
  `lru`/`reactive` ordering as a real finding rather than small-sample noise.
- Consider adding an `eai_coactivation` variant that ALSO tracks long-run
  frequency (blend prediction with an LFU-style floor) given §2's finding -
  worth a follow-up experiment before writing this off as "EAI only helps
  cold caches".
- `scripts/validate_benchmark_invariants.py`'s docstring now documents the
  cold-only scoping of the oracle-hit_rate invariant; the script itself only
  ever tests cold scenario, so no code change was needed there.
