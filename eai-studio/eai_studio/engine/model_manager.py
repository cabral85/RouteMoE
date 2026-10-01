"""Load, serve, and unload MoE models on demand - the piece that turns the
research PoC's validated engine (chunked_loader.py + expert_cache.py, both
carried over byte-for-byte from eai-poc, Rounds 4-7) into something an API
server can actually drive: "load this model, generate from it, unload it"
instead of "run this one benchmark script and exit."

Why a fixed-budget GlobalExpertCache by default, not full residency: the
whole point of this project is making a model that wouldn't otherwise fit
(a 30B-class MoE needs ~58GB of expert weights alone) fit anyway, in a
bounded amount of RAM/VRAM, by streaming experts from disk on demand. See
eai-poc's README for the full validated story - this module is the
production-facing wrapper around exactly that mechanism, not a new one.
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from .chunked_loader import load_chunked_model
from .expert_cache import BenchmarkStats, GlobalExpertCache

logger = logging.getLogger("eai_studio.model_manager")

EvictionPolicyName = Literal["reactive", "lru", "lfu", "hybrid"]


def find_snapshot_dir(model_id: str) -> str:
    """Locate an already-downloaded HF Hub snapshot for `model_id` (e.g.
    "allenai/OLMoE-1B-7B-0924"). Does NOT download - see
    docs/ADDING_A_MODEL.md for how a model gets here in the first place."""
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = model_id.split("/")
    pattern = os.path.join(cache_home, "hub", f"models--{org}--{name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No local snapshot found for {model_id!r} under {pattern}. "
            f"Download it first (see docs/ADDING_A_MODEL.md) - this manager "
            f"never downloads on your behalf, to keep load latency predictable."
        )
    return matches[0]


def _tensor_name_fn(layer_idx: int, expert_idx: int, proj: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}_proj.weight"


@dataclass
class ModelConfig:
    """Everything needed to load one model - the request body for POST
    /api/models/load, and what docs/ADDING_A_MODEL.md's checklist produces."""

    model_id: str  # HF Hub id, e.g. "allenai/OLMoE-1B-7B-0924"
    cache_gb: float = 4.0  # expert cache budget - this, not the model's full size, is what has to fit in RAM/VRAM
    policy: EvictionPolicyName = "hybrid"  # see eai-poc/docs/cache_policy_matrix.md for how each policy compares
    device: str = "cpu"  # "cuda" makes cache_gb a VRAM budget instead
    dtype: str = "bfloat16"
    max_resident_per_layer: int = 24  # bounds stage-of-load memory spikes on large models - see eai-poc Round 7
    # hybrid policy weights (ignored by every other policy) - defaults match
    # eai-poc's own small weight sweep's best-performing combination so far
    # (docs/cache_policy_matrix.md), not an untested guess.
    hybrid_alpha: float = 0.2
    hybrid_beta: float = 0.2
    hybrid_gamma: float = 0.5
    hybrid_delta: float = 3.0
    hybrid_lambda: float = 0.5


@dataclass
class LoadedModel:
    config: ModelConfig
    model: object
    blocks: list
    shard_index: object
    cache: GlobalExpertCache
    tokenizer: object
    loaded_at: float = field(default_factory=time.time)
    stats: BenchmarkStats = field(default_factory=BenchmarkStats)

    def status(self) -> dict:
        return {
            "model_id": self.config.model_id,
            "device": self.config.device,
            "policy": self.config.policy,
            "cache_gb": self.config.cache_gb,
            "loaded_at": self.loaded_at,
            "uptime_seconds": time.time() - self.loaded_at,
            "resident_bytes": self.cache.resident_bytes,
            "resident_experts": len(self.cache.resident_keys),
            **self.stats.as_dict(),
        }


class ModelManager:
    """One process, any number of models loaded/unloaded over its lifetime -
    but deliberately ONE active generation at a time per model (a lock, not
    a request queue - see generate()'s docstring for why that's the right
    tradeoff for a local, single-user Studio rather than a multi-tenant
    server).
    """

    def __init__(self):
        self._models: dict[str, LoadedModel] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def list_loaded(self) -> list[dict]:
        with self._registry_lock:
            return [m.status() for m in self._models.values()]

    def is_loaded(self, model_id: str) -> bool:
        with self._registry_lock:
            return model_id in self._models

    def load(self, config: ModelConfig) -> dict:
        with self._registry_lock:
            if config.model_id in self._models:
                raise ValueError(f"{config.model_id!r} is already loaded - unload it first to reload with different settings")

        from transformers import AutoTokenizer

        model_dir = find_snapshot_dir(config.model_id)
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA device is available on this machine")

        logger.info("loading %s on device=%s cache_gb=%.1f policy=%s", config.model_id, config.device, config.cache_gb, config.policy)
        t0 = time.perf_counter()
        model, blocks, shard_index, _load_stats = load_chunked_model(config.model_id, model_dir, dtype=dtype, device=config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)

        stats = BenchmarkStats()
        cache_policy = "lru" if config.policy == "reactive" else config.policy
        # "reactive" (no prediction, no smart eviction) is FIFO underneath -
        # GlobalExpertCache's own policy name is "fifo" for that; everything
        # else maps 1:1. See eai-poc/eai/expert_cache.py for why reactive
        # and lru are mechanically distinct despite neither prefetching.
        if config.policy == "reactive":
            cache_policy = "fifo"
        cache = GlobalExpertCache(
            shard_index=shard_index,
            budget_bytes=int(config.cache_gb * 1e9),
            policy=cache_policy,
            dtype=dtype,
            stats=stats,
            tensor_name_fn=_tensor_name_fn,
            device=config.device,
            hybrid_alpha=config.hybrid_alpha,
            hybrid_beta=config.hybrid_beta,
            hybrid_gamma=config.hybrid_gamma,
            hybrid_delta=config.hybrid_delta,
            hybrid_lambda=config.hybrid_lambda,
        )
        for b in blocks:
            b.attach_global_cache(cache)

        loaded = LoadedModel(config=config, model=model, blocks=blocks, shard_index=shard_index, cache=cache, tokenizer=tokenizer, stats=stats)
        with self._registry_lock:
            self._models[config.model_id] = loaded
            self._locks[config.model_id] = threading.Lock()

        elapsed = time.perf_counter() - t0
        logger.info("loaded %s in %.1fs", config.model_id, elapsed)
        return {"model_id": config.model_id, "load_seconds": elapsed, **loaded.status()}

    def unload(self, model_id: str) -> dict:
        with self._registry_lock:
            loaded = self._models.pop(model_id, None)
            self._locks.pop(model_id, None)
        if loaded is None:
            raise KeyError(f"{model_id!r} is not currently loaded")

        resident_gb = loaded.cache.resident_bytes / 1e9
        loaded.cache.evict_all()
        loaded.shard_index.close()
        device = loaded.config.device
        del loaded.model, loaded.blocks, loaded.cache, loaded.tokenizer, loaded
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info("unloaded %s (freed ~%.2fGB resident)", model_id, resident_gb)
        return {"model_id": model_id, "unloaded": True, "freed_resident_gb": resident_gb}

    def generate(self, model_id: str, messages: list[dict], max_new_tokens: int = 256, temperature: float = 0.0) -> dict:
        """Greedy (temperature=0) or simple sampled generation. One request
        at a time per model (the lock below) - this is a local Studio for
        one person's own use, not a multi-tenant inference server; batching
        concurrent requests through one shared KV cache is real, separate
        engineering (continuous batching, vLLM/TGI-style) that's explicitly
        out of scope until this foundation is validated in daily use.
        """
        with self._registry_lock:
            loaded = self._models.get(model_id)
            lock = self._locks.get(model_id)
        if loaded is None:
            raise KeyError(f"{model_id!r} is not loaded - call load() first")

        with lock:
            from transformers import DynamicCache

            device = next(loaded.model.parameters()).device
            prompt_text = _render_prompt(loaded.tokenizer, messages)
            prompt_ids = loaded.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

            kv_cache = DynamicCache(config=loaded.model.config)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = loaded.model(input_ids=prompt_ids, past_key_values=kv_cache, use_cache=True)
            ttft = time.perf_counter() - t0

            generated_ids: list[int] = []
            next_id = out.logits[0, -1].argmax().item() if temperature <= 0 else _sample(out.logits[0, -1], temperature)
            t0 = time.perf_counter()
            for _ in range(max_new_tokens):
                generated_ids.append(next_id)
                if next_id == loaded.tokenizer.eos_token_id:
                    break
                next_input = torch.tensor([[next_id]], device=device)
                with torch.no_grad():
                    out = loaded.model(input_ids=next_input, past_key_values=kv_cache, use_cache=True)
                next_id = out.logits[0, -1].argmax().item() if temperature <= 0 else _sample(out.logits[0, -1], temperature)
            gen_seconds = time.perf_counter() - t0

            text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return {
                "text": text,
                "output_tokens": len(generated_ids),
                "ttft_ms": ttft * 1000,
                "tokens_per_second": len(generated_ids) / max(1e-9, gen_seconds),
                "cache_stats": loaded.stats.as_dict(),
            }


def _sample(logits: torch.Tensor, temperature: float) -> int:
    probs = torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    """Use the tokenizer's own chat template when the model ships one
    (every modern Instruct/Chat checkpoint does); fall back to a plain
    role-prefixed transcript for base models that don't (e.g. OLMoE's base
    checkpoint) instead of hard-failing - a base model still completes text
    reasonably from a simple transcript, just without the finetuned
    chat behavior an Instruct model would give."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)
