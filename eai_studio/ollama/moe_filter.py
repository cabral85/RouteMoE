"""Identifies which model names (Ollama's own, or an HF Hub id) are
Mixture-of-Experts architectures - eai-studio's whole value proposition
(chunked expert loading, a fixed-budget expert cache) is specific to MoE;
a dense model gets no benefit from it at all (see eai-poc's
dense_layer_streaming_experiment.py finding: no sparsity to exploit, so no
EAI-style win is even possible - don't pretend otherwise in the UI).

Two layers, since neither alone is reliable:
  1. KNOWN_MOE_MODELS - an explicit allowlist of model families already
     validated (by name/family substring) against eai-poc's own traced
     models or well-documented public MoE releases.
  2. A permissive substring heuristic (is_likely_moe) for anything NOT on
     the allowlist, so a new MoE release doesn't just silently fail to show
     up - flagged as "likely", not "known", in the UI (see
     docs/ADDING_A_MODEL.md for promoting a model from likely to known).
"""

from __future__ import annotations

# (family substring, display name) - matched case-insensitively against an
# Ollama model name or HF Hub id. Families actually traced/benchmarked in
# eai-poc get a plain comment saying so; the rest are well-documented public
# MoE releases not yet run through this project's own pipeline.
KNOWN_MOE_MODELS = [
    ("olmoe", "OLMoE"),  # traced + benchmarked extensively in eai-poc (Rounds 1-7)
    ("qwen1.5-moe", "Qwen1.5-MoE"),  # traced + benchmarked in eai-poc (Round 3+)
    ("qwen2moe", "Qwen1.5/2-MoE"),
    ("qwen3-30b-a3b", "Qwen3-30B-A3B"),  # traced + benchmarked at scale in eai-poc (Round 6+)
    ("qwen3moe", "Qwen3-MoE"),
    ("mixtral", "Mixtral"),
    ("deepseek-moe", "DeepSeek-MoE"),
    ("deepseek-v2", "DeepSeek-V2 (MoE)"),
    ("deepseek-v3", "DeepSeek-V3 (MoE)"),
    ("granitemoe", "Granite MoE"),
    ("granite-moe", "Granite MoE"),
    ("jetmoe", "JetMoE"),
    ("phi-3.5-moe", "Phi-3.5-MoE"),
    ("gpt-oss", "GPT-OSS (MoE)"),
    ("dbrx", "DBRX"),
    ("grok", "Grok (MoE)"),
    ("arctic", "Snowflake Arctic (MoE)"),
]

# Generic substrings that suggest "probably MoE" even for a family not on
# the explicit list above - deliberately broad, meant to surface candidates
# for a human to confirm, not to be authoritative on its own.
_LIKELY_SUBSTRINGS = ["moe", "mixture-of-experts", "a3b", "a2.7b", "8x7b", "8x22b"]


def is_known_moe(name: str) -> bool:
    lowered = name.lower()
    return any(family in lowered for family, _ in KNOWN_MOE_MODELS)


def is_likely_moe(name: str) -> bool:
    lowered = name.lower()
    return is_known_moe(name) or any(s in lowered for s in _LIKELY_SUBSTRINGS)


def display_name(name: str) -> str | None:
    lowered = name.lower()
    for family, label in KNOWN_MOE_MODELS:
        if family in lowered:
            return label
    return None
