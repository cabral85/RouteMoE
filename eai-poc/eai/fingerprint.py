"""Cheap prompt/token fingerprints, tried in the order the spec asks for:

  A) hidden_state - mean-pooled early hidden states from the model itself (cheapest
     that still carries real semantic signal, no extra model to load).
  B) a separate small embedding model - not implemented in this PoC. Slot left open
     (see `FINGERPRINT_METHODS`) - only worth adding once A's metrics say it's needed.
  C) hashing - text feature-hashing, zero model computation. Useful as a sanity-check
     lower bound: if the cluster predictor barely beats this, fingerprints aren't
     carrying much semantic signal.

Two granularities of (A) are supported:
  - "prompt": one fingerprint per prompt (mean-pooled over all its tokens). The
    predictor then makes one prediction per prompt per layer.
  - "token": one fingerprint per token (that token's own, unpooled hidden state -
    already contextualized by everything before it, since the model is causal).
    The predictor then makes one prediction per *token* per layer. This is the
    follow-up experiment the PoC's results pointed at: if routing is driven by
    local/lexical token features rather than whole-prompt topic, clustering
    tokens directly should beat clustering prompts.
"""

from __future__ import annotations

import numpy as np

from .tracing import TraceSet

HASHING_DIM_DEFAULT = 256
GRANULARITIES = ("prompt", "token")


def _normalize(fp: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(fp, axis=1, keepdims=True)
    return fp / np.clip(norms, 1e-8, None)


def hidden_state_fingerprint(traces: TraceSet, normalize: bool = True, granularity: str = "prompt") -> np.ndarray:
    """Option A: hidden state at a shallow layer, computed during tracing.

    granularity="prompt" -> (num_prompts, hidden_size), mean-pooled per prompt.
    granularity="token"  -> (total_tokens, hidden_size), one row per token.
    """
    if granularity == "prompt":
        fp = traces.fingerprints_hidden.astype(np.float32)
    elif granularity == "token":
        fp = traces.fingerprints_hidden_tokens.astype(np.float32)
    else:
        raise ValueError(f"unknown granularity: {granularity} (known: {GRANULARITIES})")
    return _normalize(fp) if normalize else fp


def hashing_fingerprint(texts: list[str], dim: int = HASHING_DIM_DEFAULT) -> np.ndarray:
    """Option C: cheap feature-hashing baseline, no model needed. Prompt-level only."""
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(
        n_features=dim, alternate_sign=True, norm="l2", analyzer="char_wb", ngram_range=(3, 5)
    )
    return vec.transform(texts).toarray().astype(np.float32)


def fingerprint_dim(method: str, traces: TraceSet | None = None, dim: int = HASHING_DIM_DEFAULT) -> int:
    if method == "hidden_state":
        assert traces is not None
        return traces.fingerprints_hidden.shape[1]
    if method == "hashing":
        return dim
    raise ValueError(f"unknown fingerprint method: {method}")


def compute_fingerprints(
    method: str,
    traces: TraceSet,
    prompts_by_id: dict[str, str] | None = None,
    dim: int = HASHING_DIM_DEFAULT,
    granularity: str = "prompt",
) -> np.ndarray:
    """Fingerprint matrix for every prompt (or token, if granularity="token") in `traces`.

    Row order matches `traces.prompt_ids` (granularity="prompt") or the
    concatenated token order used by `traces.token_prompt_idx` (granularity="token").
    """
    if method == "hidden_state":
        return hidden_state_fingerprint(traces, granularity=granularity)
    if method == "hashing":
        if granularity != "prompt":
            raise ValueError("hashing fingerprint is text-based and only defined at prompt granularity")
        if prompts_by_id is None:
            raise ValueError("hashing fingerprint needs the raw prompt text (prompts_by_id)")
        texts = [prompts_by_id[str(pid)] for pid in traces.prompt_ids]
        return hashing_fingerprint(texts, dim=dim)
    raise ValueError(f"unknown fingerprint method: {method} (known: hidden_state, hashing)")


FINGERPRINT_METHODS = ("hidden_state", "hashing")
