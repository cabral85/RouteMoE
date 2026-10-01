#!/usr/bin/env python
"""Unit tests for eai_studio/engine/model_manager.py - the parts that don't
need an actual multi-GB model downloaded: snapshot resolution, the
chat-template fallback, and the manager's bookkeeping (load/unload/is_loaded
guard rails). Loading a real model is covered by manual end-to-end testing
against the running server (see docs/EXPOSING_VIA_API.md), not here.

    python tests/test_model_manager.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai_studio.engine.model_manager import (
    LoadedModel,
    ModelConfig,
    ModelManager,
    _render_prompt,
    find_snapshot_dir,
)


def test_find_snapshot_dir_raises_for_unknown_model():
    try:
        find_snapshot_dir("some-org/definitely-not-cached-anywhere")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "docs/ADDING_A_MODEL.md" in str(e)  # points the user somewhere useful, not a bare traceback
    print("test_find_snapshot_dir_raises_for_unknown_model: OK")


def test_render_prompt_uses_chat_template_when_present():
    tokenizer = SimpleNamespace(
        chat_template="irrelevant-but-present",
        apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEMPLATED",
    )
    result = _render_prompt(tokenizer, [{"role": "user", "content": "hi"}])
    assert result == "TEMPLATED"
    print("test_render_prompt_uses_chat_template_when_present: OK")


def test_render_prompt_falls_back_for_base_models():
    # e.g. OLMoE-1B-7B-0924's base checkpoint: no chat_template attribute at all
    tokenizer = SimpleNamespace()
    result = _render_prompt(tokenizer, [{"role": "user", "content": "hi"}])
    assert result == "user: hi\nassistant:"
    print("test_render_prompt_falls_back_for_base_models: OK")


def test_manager_rejects_unknown_model_operations():
    manager = ModelManager()
    assert not manager.is_loaded("nothing/loaded")
    assert manager.list_loaded() == []
    try:
        manager.unload("nothing/loaded")
        assert False, "expected KeyError"
    except KeyError:
        pass
    try:
        manager.generate("nothing/loaded", [{"role": "user", "content": "hi"}])
        assert False, "expected KeyError"
    except KeyError:
        pass
    print("test_manager_rejects_unknown_model_operations: OK")


def test_manager_rejects_double_load():
    # Bypass the real load() (needs a cached model on disk) by inserting a
    # fake LoadedModel directly - this test is only about the registry guard.
    manager = ModelManager()
    config = ModelConfig(model_id="fake/already-loaded")
    fake = LoadedModel(config=config, model=None, blocks=[], shard_index=None, cache=SimpleNamespace(resident_bytes=0, resident_keys=[]), tokenizer=None)
    manager._models[config.model_id] = fake
    try:
        manager.load(config)
        assert False, "expected ValueError for an already-loaded model_id"
    except ValueError as e:
        assert "already loaded" in str(e)
    print("test_manager_rejects_double_load: OK")


if __name__ == "__main__":
    test_find_snapshot_dir_raises_for_unknown_model()
    test_render_prompt_uses_chat_template_when_present()
    test_render_prompt_falls_back_for_base_models()
    test_manager_rejects_unknown_model_operations()
    test_manager_rejects_double_load()
    print("\nOK - all model_manager unit tests passed")
