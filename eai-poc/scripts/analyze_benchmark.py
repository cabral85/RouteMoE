#!/usr/bin/env python
"""Fase 8: turns scripts/benchmark_streaming.py's JSONL events into the
tables and graphs that actually answer the benchmark's question - "quanto
ganho real o EAI entrega sobre caching simples, e quao perto chega de um
oracle perfeito?" Pure data analysis: no model, no GPU/CPU load, safe to run
any time regardless of what else is using system memory.

Rules this script enforces structurally, not just by convention (per the
spec's explicit "don't do these things" list):

  - Never compares policies across DIFFERENT cache budgets. Every table and
    every plot facets by cache_budget_bytes first; a policy is only ever
    ranked against other policies that ran under the identical budget.
  - Oracle is always labeled as an upper bound, never as a candidate
    "winner" - it answers "how far is the gap to close", not "what to ship".
  - Never ranks policies by peak_resident_bytes alone - memory is reported
    as a diagnostic column (and a plot of its own), not a scoring axis.
  - Negative results are not filtered out or softened: a policy's delta vs.
    `reactive` (the simplest possible baseline) is printed for every policy,
    including when it's negative.

    python scripts/analyze_benchmark.py --events artifacts/benchmark/events.jsonl
    python scripts/analyze_benchmark.py --events artifacts/benchmark/*.jsonl --out-dir artifacts/benchmark/report
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORACLE_LABEL = "oracle (upper bound - not a deployable policy)"
BASELINE_POLICY = "reactive"

METRIC_COLUMNS = [
    "tokens_per_second", "hit_rate", "expert_io_wait_ms", "compute_ms",
    "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
    "prefetch_precision", "useful_prefetch_ratio",
    "predictor_top1_accuracy", "eai_lookup_ms",
    "storage_bytes_per_output_token", "peak_resident_bytes",
]


def load_events(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no events found matching {patterns}")
    df = pd.DataFrame(rows)
    df["cache_budget_gb"] = df["cache_budget_bytes"] / 1e9
    return df


def lookahead_depth(policy: str) -> int | None:
    if policy == "eai":
        return 0
    if policy.startswith("eai_lookahead_"):
        return int(policy.rsplit("_", 1)[1])
    return None


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per (policy, cache_budget_gb) across prompts - the unit every
    table/plot below actually compares."""
    agg = df.groupby(["policy", "cache_budget_gb"])[METRIC_COLUMNS].mean().reset_index()
    agg["peak_resident_gb"] = agg["peak_resident_bytes"] / 1e9

    baseline = agg[agg["policy"] == BASELINE_POLICY].set_index("cache_budget_gb")["tokens_per_second"]
    oracle = agg[agg["policy"] == "oracle"].set_index("cache_budget_gb")["tokens_per_second"]

    def gain_fraction(row) -> float:
        b, o = baseline.get(row["cache_budget_gb"]), oracle.get(row["cache_budget_gb"])
        if b is None or o is None or abs(o - b) < 1e-9:
            return float("nan")
        return (row["tokens_per_second"] - b) / (o - b)

    def delta_vs_baseline(row) -> float:
        b = baseline.get(row["cache_budget_gb"])
        return float("nan") if b is None else row["tokens_per_second"] - b

    agg["fraction_of_oracle_gain"] = agg.apply(gain_fraction, axis=1)
    agg["tokens_per_second_delta_vs_reactive"] = agg.apply(delta_vs_baseline, axis=1)
    return agg


def print_tables(agg: pd.DataFrame) -> None:
    for budget in sorted(agg["cache_budget_gb"].unique()):
        sub = agg[agg["cache_budget_gb"] == budget].sort_values("tokens_per_second", ascending=False)
        print(f"\n=== cache_budget = {budget:.2f} GB (policies below ran under THIS budget only) ===")
        cols = ["policy", "tokens_per_second", "tokens_per_second_delta_vs_reactive", "hit_rate",
                "fraction_of_oracle_gain", "prefetch_precision", "predictor_top1_accuracy", "peak_resident_gb"]
        display = sub[cols].copy()
        display["policy"] = display["policy"].replace({"oracle": ORACLE_LABEL})
        for c in ["tokens_per_second", "tokens_per_second_delta_vs_reactive", "peak_resident_gb"]:
            display[c] = display[c].map(lambda v: f"{v:.2f}")
        for c in ["hit_rate", "fraction_of_oracle_gain", "prefetch_precision", "predictor_top1_accuracy"]:
            display[c] = display[c].map(lambda v: "n/a" if pd.isna(v) else f"{v*100:.1f}%")
        print(display.to_string(index=False))

        negative = sub[sub["tokens_per_second_delta_vs_reactive"] < 0]
        if not negative.empty:
            names = ", ".join(negative["policy"].tolist())
            print(f"  NOTE: at this budget, {names} performed WORSE than the '{BASELINE_POLICY}' baseline (negative result, reported as-is).")

        oracle_row = sub[sub["policy"] == "oracle"]
        reactive_row = sub[sub["policy"] == BASELINE_POLICY]
        if not oracle_row.empty and not reactive_row.empty:
            oracle_tps = oracle_row["tokens_per_second"].iloc[0]
            reactive_tps = reactive_row["tokens_per_second"].iloc[0]
            if oracle_tps < reactive_tps:
                print(
                    f"  WARNING: oracle tok/s ({oracle_tps:.2f}) < reactive tok/s ({reactive_tps:.2f}) at this budget, "
                    f"despite oracle having a >= hit_rate by construction. This means wall-clock timing at this budget "
                    f"is likely contaminated by concurrent load on the machine this run happened on, not a real result - "
                    f"the 'fraction_of_oracle_gain' column above is NOT trustworthy for this budget. Hit_rate and "
                    f"prefetch_precision are still valid (they don't depend on wall-clock timing). Re-run on a quieter "
                    f"machine (or with repeated trials) before drawing timing conclusions at this budget."
                )


def savefig(fig, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_grouped_bar(agg: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, filename: str, as_pct: bool = False) -> None:
    budgets = sorted(agg["cache_budget_gb"].unique())
    policies = sorted(agg["policy"].unique())
    fig, ax = plt.subplots(figsize=(max(6, len(policies) * 1.1), 4.5))
    width = 0.8 / max(1, len(budgets))
    x = np.arange(len(policies))
    for i, budget in enumerate(budgets):
        sub = agg[agg["cache_budget_gb"] == budget].set_index("policy")
        values = [sub[metric].get(p, np.nan) for p in policies]
        if as_pct:
            values = [v * 100 if not pd.isna(v) else np.nan for v in values]
        ax.bar(x + i * width, values, width, label=f"{budget:.1f}GB")
    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by policy (faceted by cache budget - never compared across budgets)")
    ax.legend(title="cache budget")
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, out_dir, filename)


def plot_io_vs_compute(agg: pd.DataFrame, out_dir: Path) -> None:
    for budget in sorted(agg["cache_budget_gb"].unique()):
        sub = agg[agg["cache_budget_gb"] == budget].sort_values("tokens_per_second", ascending=False)
        fig, ax = plt.subplots(figsize=(max(6, len(sub) * 1.1), 4.5))
        x = np.arange(len(sub))
        ax.bar(x, sub["compute_ms"], label="compute")
        ax.bar(x, sub["expert_io_wait_ms"], bottom=sub["compute_ms"], label="expert I/O wait")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy"], rotation=30, ha="right")
        ax.set_ylabel("ms per prompt (generation phase only)")
        ax.set_title(f"Compute vs. expert I/O wait - cache_budget={budget:.1f}GB")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        savefig(fig, out_dir, f"io_vs_compute_{budget:.1f}gb.png")


def plot_prefetch_waste(df: pd.DataFrame, out_dir: Path) -> None:
    prefetching = df[df["prefetched_experts"] > 0]
    if prefetching.empty:
        return
    for budget in sorted(prefetching["cache_budget_gb"].unique()):
        sub = prefetching[prefetching["cache_budget_gb"] == budget].groupby("policy")[
            ["bytes_prefetched_used", "bytes_prefetched_unused"]
        ].mean().reset_index()
        fig, ax = plt.subplots(figsize=(max(6, len(sub) * 1.1), 4.5))
        x = np.arange(len(sub))
        ax.bar(x, sub["bytes_prefetched_used"] / 1e6, label="useful (bytes)")
        ax.bar(x, sub["bytes_prefetched_unused"] / 1e6, bottom=sub["bytes_prefetched_used"] / 1e6, label="wasted (bytes)")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy"], rotation=30, ha="right")
        ax.set_ylabel("MB prefetched")
        ax.set_title(f"Prefetch: useful vs. wasted bytes - cache_budget={budget:.1f}GB")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        savefig(fig, out_dir, f"prefetch_waste_{budget:.1f}gb.png")


def plot_lookahead_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    df["lookahead"] = df["policy"].map(lookahead_depth)
    sweep = df.dropna(subset=["lookahead"])
    if sweep["lookahead"].nunique() < 2:
        return
    for budget in sorted(sweep["cache_budget_gb"].unique()):
        sub = sweep[sweep["cache_budget_gb"] == budget].groupby("lookahead")[["hit_rate", "predictor_top1_accuracy"]].mean().reset_index().sort_values("lookahead")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(sub["lookahead"], sub["hit_rate"] * 100, marker="o", label="hit_rate")
        ax.plot(sub["lookahead"], sub["predictor_top1_accuracy"] * 100, marker="s", label="predictor_top1_accuracy")
        ax.set_xlabel("lookahead depth (future steps prefetched)")
        ax.set_ylabel("%")
        ax.set_title(f"EAI lookahead sweep - cache_budget={budget:.1f}GB")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, out_dir, f"lookahead_sweep_{budget:.1f}gb.png")


def plot_peak_memory(agg: pd.DataFrame, out_dir: Path) -> None:
    """Diagnostic only - peak_resident_bytes vs. requested budget, to check
    the byte-budget cap is actually being respected. NOT a ranking plot:
    the x-axis is the budget itself, not a policy comparison."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for policy in sorted(agg["policy"].unique()):
        sub = agg[agg["policy"] == policy].sort_values("cache_budget_gb")
        ax.plot(sub["cache_budget_gb"], sub["peak_resident_gb"], marker="o", label=policy)
    lims = [agg["cache_budget_gb"].min(), agg["cache_budget_gb"].max()]
    ax.plot(lims, lims, "k--", alpha=0.4, label="budget (y=x)")
    ax.set_xlabel("requested cache_budget_bytes (GB)")
    ax.set_ylabel("measured peak_resident_bytes (GB)")
    ax.set_title("Budget enforcement check (diagnostic, not a ranking)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, out_dir, "peak_memory_vs_budget.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="+", default=["artifacts/benchmark/events.jsonl"])
    parser.add_argument("--out-dir", default="artifacts/benchmark/report")
    args = parser.parse_args()

    df = load_events(args.events)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} events, {df['policy'].nunique()} policies, {sorted(df['cache_budget_gb'].unique())} GB budgets, {df['prompt_id'].nunique()} distinct prompts")

    agg = aggregate(df)
    print_tables(agg)

    print(f"\nWriting plots to {out_dir}/")
    plot_grouped_bar(agg, out_dir, "tokens_per_second", "tokens/second", "tokens_per_second_by_policy.png")
    plot_grouped_bar(agg, out_dir, "hit_rate", "cache hit rate (%)", "hit_rate_by_policy.png", as_pct=True)
    plot_grouped_bar(agg, out_dir, "fraction_of_oracle_gain", "fraction of oracle's achievable gain realized (%)", "fraction_of_oracle_gain.png", as_pct=True)
    plot_grouped_bar(agg, out_dir, "tpot_ms_p95", "TPOT p95 (ms/token)", "tpot_p95_by_policy.png")
    plot_io_vs_compute(agg, out_dir)
    plot_prefetch_waste(df, out_dir)
    plot_lookahead_sweep(df, out_dir)
    plot_peak_memory(agg, out_dir)

    agg.to_csv(out_dir / "aggregated_metrics.csv", index=False)
    print(f"  wrote {out_dir / 'aggregated_metrics.csv'}")


if __name__ == "__main__":
    main()
