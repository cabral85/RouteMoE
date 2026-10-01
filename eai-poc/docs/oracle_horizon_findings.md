# Oracle horizon findings

Status: OLMoE validated (3 prompts, 4GB budget); Qwen3-30B and a real weight
sweep not yet run. Answers the question this phase's spec posed directly:

> LFU bateu Oracle porque prediction é inútil ou porque Oracle-1 tem
> horizonte curto demais?

## What was built

`oracle_1`/`oracle_2`/`oracle_4`/`oracle_8` in `scripts/benchmark_streaming.py`
- N = how many future steps' real (ground-truth) expert selections the
policy is allowed to see and prefetch, all in one window, every step. Plain
`oracle` is kept as an exact alias for `oracle_1` (same lookahead=0 meaning,
same prefetch calls) - every prior sweep's "oracle" rows stay meaningful
unchanged. Still benchmark-only: this only ever changes cache/prefetch
timing, never the router's real decision or the generated tokens (same
per-step correctness gate as every other policy, 0 mismatches across every
run below).

**A pre-existing gap this surfaced**: the codebase already had the
`lookahead` mechanism built (used by `eai_lookahead_N`), but plain `oracle`
never actually used it - the string match only fired for
`"eai_lookahead_"`, so `oracle` has *always* run at lookahead=0 in practice,
despite the surrounding code structure looking multi-step-ready. `oracle_N`
finally wires that up for real.

## Results (OLMoE, 3 prompts, 4GB budget, cold)

| policy | hit_rate | tok/s | evictions | prefetched | wasted prefetches |
|---|---|---|---|---|---|
| reactive | 40.8% | 2.54 | 1535 | 0 | 0 |
| oracle (= oracle_1) | 61.5% | 6.00-6.39 | 1434 | 546 | **0** |
| oracle_2 | 61.4% | 6.25 | 1501 | 610 | 0 |
| oracle_4 | 56.3% | 4.69 | 2079 | 1027 | **488** |
| oracle_8 | 45.3% | 2.91 | 4409 | 3009 | **2249** |

## The answer

**Not "prediction is useless" - "a wider Oracle horizon makes it worse under
a fixed budget, not better."** oracle_2 barely differs from oracle_1
(61.4% vs 61.5%). oracle_4 already loses ground (56.3%, first time a bare
Oracle variant has *any* wasted prefetches at all - 488). oracle_8 nearly
collapses to reactive's own hit_rate (45.3% vs 40.8%) while moving 3x the
evictions and wasting **~75% of its own prefetches** (2249/3009).

Mechanism, same shape as every other "aggressive prefetch under a fixed
budget" finding this project keeps running into: a wider window means the
Oracle prefetches the *union* of what N future steps will need, all at
once, before running any of them - under a fixed budget, that union
competes with *itself* for space long before the real router ever asks for
any of it. By step 8's window, most of what got prefetched gets evicted
(to make room for the rest of that same window) before its own turn
arrives - the Oracle is thrashing against its own foresight, not against
prediction error (there is none - it's ground truth by construction).

**Direct implication for the original warm-cache/`lfu` question**: this
rules out "Oracle-1's horizon is too short" as the explanation for
`lfu` beating it in a warm cache
([docs/benchmark_findings_current.md](benchmark_findings_current.md) #2).
A *longer* horizon doesn't recover ground against `lfu` - it loses more,
faster, for the same budget-competition reason that made the 1-step
Oracle's own prefetching evict things `lfu` would have kept. The fix
`lfu` gets almost for free (never evicting proactively, only under real
pressure) isn't a horizon problem prediction can out-plan its way past -
it's a structural tension between "prefetch what you know you'll need" and
"a fixed budget can't hold everything you know you'll need at once."

## What's not yet done

- Same sweep at Qwen3-30B scale (48 layers, 128 experts - the tension
  described above should be even sharper there, given more layers over
  which a wide window's union can accumulate).
- A real weight/budget sweep (only one 4GB budget point tested here).
- Combining `oracle_N` with the `hybrid` policy's admission control (Fase
  4) - would a benefit/cost gate on the Oracle's own prefetches recover
  some of oracle_4/oracle_8's lost ground by refusing to admit prefetches
  the budget can't actually afford to keep? Untested, plausible next step.
