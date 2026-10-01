#!/usr/bin/env python
"""Unit tests for eai_studio/ollama/moe_filter.py - pure string matching,
no model/network needed.

    python tests/test_moe_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai_studio.ollama.moe_filter import display_name, is_known_moe, is_likely_moe


def test_known_moe_models_detected():
    assert is_known_moe("olmoe:latest")
    assert is_known_moe("OLMoE-1B-7B-0924")  # case-insensitive
    assert is_known_moe("mixtral:8x7b")
    assert is_known_moe("qwen3-30b-a3b-instruct")
    print("test_known_moe_models_detected: OK")


def test_dense_models_not_flagged_known():
    assert not is_known_moe("llama3:8b")
    assert not is_known_moe("qwen2.5:7b")
    assert not is_known_moe("mistral:7b")
    print("test_dense_models_not_flagged_known: OK")


def test_likely_moe_catches_unlisted_naming_patterns():
    # not on the explicit allowlist, but the "a3b"/"8x22b"-style naming
    # convention strongly suggests MoE even for an unrecognized family
    assert is_likely_moe("some-new-model-a3b")
    assert is_likely_moe("brand-new-8x22b-release")
    assert not is_likely_moe("llama3:70b")
    print("test_likely_moe_catches_unlisted_naming_patterns: OK")


def test_display_name_returns_human_label_or_none():
    assert display_name("olmoe:latest") == "OLMoE"
    assert display_name("llama3:8b") is None
    print("test_display_name_returns_human_label_or_none: OK")


if __name__ == "__main__":
    test_known_moe_models_detected()
    test_dense_models_not_flagged_known()
    test_likely_moe_catches_unlisted_naming_patterns()
    test_display_name_returns_human_label_or_none()
    print("\nOK - all moe_filter unit tests passed")
