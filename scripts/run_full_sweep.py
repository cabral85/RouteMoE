#!/usr/bin/env python
"""Fase 6: the full sweep - every (cache_budget, workload, cache_scenario)
combination, each internally covering every policy across the same fixed
set of prompts (so policies within one combination are always compared
fairly, on identical inputs, per scripts/benchmark_streaming.py's own
design). Drives scripts/benchmark_streaming.py via subprocess, exactly like
scripts/validate_benchmark_invariants.py does, so every run gets the same
memory-guard protection and unbuffered live logging.

Two named configs are built in:
  olmoe      - full grid (5 budgets x 4 workloads x 2 scenarios = 40 runs),
               cheap enough to actually run in full.
  qwen3_30b  - deliberately narrower (fewer budgets/workloads/policies,
               cold only) - Qwen3-30B is ~10-20x slower per token than
               OLMoE under chunked loading, so the full grid there would
               take many hours; this config exists to get real evidence at
               the actual target scale without an unbounded time commitment.

    python scripts/run_full_sweep.py --config olmoe
    python scripts/run_full_sweep.py --config qwen3_30b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "olmoe": dict(
        model="allenai/OLMoE-1B-7B-0924",
        index="artifacts/model_token128.eai",
        budgets_gb=[2, 4, 6, 8, 10],
        workloads=["random", "domain_clustered", "conversational", "adversarial_shift"],
        scenarios=["cold", "warm"],
        policies="reactive,lru,lfu,eai,eai_lookahead_2,eai_coactivation,oracle",
        num_prompts=6,
        max_new_tokens=12,
        min_free_ram_gb=4.0,
        out="artifacts/benchmark/sweep_olmoe.jsonl",
    ),
    "qwen3_30b": dict(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        index="artifacts/qwen3_30b/model_token128.eai",
        budgets_gb=[6, 10],
        workloads=["domain_clustered", "adversarial_shift"],
        scenarios=["cold"],
        policies="reactive,lru,eai_coactivation,oracle",
        num_prompts=2,
        max_new_tokens=6,
        min_free_ram_gb=12.0,
        # Raised from 8.0 after a real abort: stage 1's trace cache is
        # deliberately unbounded WITHIN one prompt's generation (never
        # evicts mid-prompt, by design - it's the ground truth every policy
        # gets compared against), and the memory-headroom guard only checks
        # BETWEEN prompts - so a 48-layer model's per-prompt working set can
        # push available RAM lower than a boundary check anticipates, before
        # the next check even runs. More margin, not a smaller model or
        # fewer tokens, since that unbounded-within-a-prompt cache is
        # intentional (see scripts/benchmark_streaming.py's
        # generate_and_trace docstring), not something to shrink away.
        out="artifacts/benchmark/sweep_qwen3_30b.jsonl",
    ),
}


def completed_combos(out_path: Path) -> set[tuple[float, str, str]]:
    """(budget_gb, workload, cache_scenario) combos already fully present in
    an existing output file - so a sweep interrupted partway (killed by an
    external low-memory watchdog, say) can pick up where it left off instead
    of re-running (and re-spending the time/risk on) work that's already
    good, validated data sitting on disk."""
    if not out_path.exists():
        return set()
    done: set[tuple[float, str, str]] = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            done.add((e["cache_budget_bytes"] / 1e9, e["workload"], e["cache_scenario"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--dry-run", action="store_true", help="print the run matrix without executing anything")
    parser.add_argument("--resume", action="store_true", help="skip (budget, workload, scenario) combos already fully present in the output file, instead of starting over")
    parser.add_argument("--min-free-ram-gb", type=float, default=None, help="override the config's default per-invocation memory floor")
    args = parser.parse_args()
    cfg = CONFIGS[args.config]
    min_free_ram_gb = args.min_free_ram_gb if args.min_free_ram_gb is not None else cfg["min_free_ram_gb"]

    out_path = ROOT / cfg["out"]
    all_runs = [
        (budget, workload, scenario)
        for budget in cfg["budgets_gb"]
        for workload in cfg["workloads"]
        for scenario in cfg["scenarios"]
    ]

    already_done: set[tuple[float, str, str]] = set()
    if args.resume:
        already_done = completed_combos(out_path)
        runs = [r for r in all_runs if r not in already_done]
        print(f"config={args.config}  --resume: {len(already_done)}/{len(all_runs)} combos already in {out_path}, {len(runs)} remaining")
    else:
        runs = all_runs
        print(f"config={args.config}  {len(runs)} runs  ->  {out_path}")

    for i, (budget, workload, scenario) in enumerate(runs, 1):
        print(f"  [{i}/{len(runs)}] budget={budget}GB workload={workload} scenario={scenario}")
    if args.dry_run:
        return
    if not runs:
        print("nothing left to run")
        return

    if out_path.exists() and not args.resume:
        backup = out_path.with_suffix(out_path.suffix + f".bak.{int(time.time())}")
        out_path.rename(backup)
        print(f"existing {out_path} moved to {backup}")

    t_start = time.perf_counter()
    for i, (budget, workload, scenario) in enumerate(runs, 1):
        cmd = [
            sys.executable, str(ROOT / "scripts" / "benchmark_streaming.py"),
            "--model", cfg["model"],
            "--index", cfg["index"],
            "--num-prompts", str(cfg["num_prompts"]),
            "--max-new-tokens", str(cfg["max_new_tokens"]),
            "--cache-gb", str(budget),
            "--policies", cfg["policies"],
            "--workload", workload,
            "--cache-scenario", scenario,
            "--min-free-ram-gb", str(min_free_ram_gb),
            "--out", str(out_path),
        ]
        elapsed = time.perf_counter() - t_start
        print(f"\n=== run {i}/{len(runs)} (elapsed {elapsed/60:.1f}min) budget={budget}GB workload={workload} scenario={scenario} ===")
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"!! run {i}/{len(runs)} exited with code {result.returncode} - stopping sweep so a bad result doesn't get lost in a huge log. Fix and re-run (already-completed runs stay in {out_path}).")
            sys.exit(result.returncode)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nSweep '{args.config}' complete: {len(runs)} runs in {total_elapsed/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
