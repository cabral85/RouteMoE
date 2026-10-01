#!/usr/bin/env python
"""Minimal smoke test for the eai pipeline using synthetic data (no model download).

Builds a tiny fake TraceSet with two clearly-separable "prompt types" that route
to disjoint experts, runs it through clustering -> profile -> storage -> predictor
-> metrics, and asserts the cluster-based predictor recovers the pattern (and
beats the frequency baselines, which can't see per-prompt differences).

    python tests/test_pipeline_synthetic.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eai import metrics as M
from eai.clustering import fit_clusters
from eai.fingerprint import hidden_state_fingerprint
from eai.predictor import Predictor
from eai.profile import build_profiles
from eai.storage import EAI_VERSION, EaiMetadata, load_index, save_index
from eai.tracing import TraceSet

NUM_LAYERS = 2
NUM_EXPERTS = 8
TOP_K_MODEL = 2
STORED_TOP_K = 4
TOKENS_PER_PROMPT = 6


def make_synthetic_traceset(num_prompts_per_type: int, seed: int) -> TraceSet:
    rng = np.random.default_rng(seed)
    prompt_ids, categories, num_tokens, fps = [], [], [], []
    token_prompt_idx, token_position = [], []
    selected_chunks, weight_chunks = [], []

    prompt_idx = 0
    for type_id, expert_pair in enumerate([(0, 1), (4, 5)]):
        for _ in range(num_prompts_per_type):
            prompt_ids.append(f"type{type_id}-{prompt_idx}")
            categories.append(f"type{type_id}")
            num_tokens.append(TOKENS_PER_PROMPT)
            base = np.zeros(4, dtype=np.float32)
            base[type_id] = 1.0
            fps.append(base + rng.normal(0, 0.02, size=4).astype(np.float32))

            token_prompt_idx.append(np.full(TOKENS_PER_PROMPT, prompt_idx, dtype=np.int32))
            token_position.append(np.arange(TOKENS_PER_PROMPT, dtype=np.int32))

            sel = np.tile(np.array(expert_pair, dtype=np.int32), (NUM_LAYERS, TOKENS_PER_PROMPT, 1))
            w = np.full((NUM_LAYERS, TOKENS_PER_PROMPT, TOP_K_MODEL), 0.5, dtype=np.float32)
            selected_chunks.append(sel)
            weight_chunks.append(w)
            prompt_idx += 1

    fps_arr = np.stack(fps, axis=0)
    return TraceSet(
        prompt_ids=np.array(prompt_ids, dtype=str),
        categories=np.array(categories, dtype=str),
        num_tokens=np.array(num_tokens, dtype=np.int32),
        fingerprint_layer=2,
        fingerprints_hidden=fps_arr,
        fingerprints_hidden_tokens=np.repeat(fps_arr, TOKENS_PER_PROMPT, axis=0),
        token_prompt_idx=np.concatenate(token_prompt_idx),
        token_position=np.concatenate(token_position),
        selected_experts=np.concatenate(selected_chunks, axis=1),
        router_weights=np.concatenate(weight_chunks, axis=1),
        forward_pass_seconds=np.full(prompt_idx, 0.01, dtype=np.float32),
        model_id="synthetic-test-model",
        architecture="SyntheticMoE",
        num_experts=NUM_EXPERTS,
        top_k=TOP_K_MODEL,
        quantization="none",
    )


def main():
    train = make_synthetic_traceset(num_prompts_per_type=10, seed=0)
    test = make_synthetic_traceset(num_prompts_per_type=4, seed=1)

    fingerprints = hidden_state_fingerprint(train, normalize=False)
    centroids, labels = fit_clusters(fingerprints, k=2, seed=42)
    assert centroids.shape == (2, 4)
    assert set(np.unique(labels)) == {0, 1}

    profiles = build_profiles(train, labels, num_clusters=2, num_experts=NUM_EXPERTS, stored_top_k=STORED_TOP_K)
    assert profiles.top_experts.shape == (2, NUM_LAYERS, STORED_TOP_K)
    # every training cluster should be observed at every layer
    assert (profiles.observation_counts > 0).all()

    with tempfile.TemporaryDirectory() as tmp:
        eai_path = str(Path(tmp) / "synthetic.eai")
        metadata = EaiMetadata(
            version=EAI_VERSION,
            model_id=train.model_id,
            architecture=train.architecture,
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K_MODEL,
            quantization="none",
            stored_top_k=STORED_TOP_K,
            fingerprint_method="hidden_state",
            fingerprint_dimension=4,
            granularity="prompt",
            number_of_clusters=2,
            created_at="test",
        )
        save_index(eai_path, metadata, profiles, centroids)
        assert Path(eai_path).exists()

        index = load_index(eai_path)
        assert index.metadata.model_id == "synthetic-test-model"
        assert index.top_experts.shape == (2, NUM_LAYERS, STORED_TOP_K)

        predictor = Predictor(index)
        test_fps = hidden_state_fingerprint(test, normalize=False)
        n = test.num_prompts
        predicted = np.full((n, NUM_LAYERS, STORED_TOP_K), -1, dtype=np.int32)
        for i in range(n):
            result = predictor.predict(test_fps[i])
            predicted[i] = result.top_experts_per_layer
            assert result.lookup_seconds >= 0

        report = M.evaluate_predictions(predicted, test, NUM_EXPERTS, ks=(1, 2, 4))

        top1 = report["aggregate"]["top1_accuracy"]
        recall2 = report["aggregate"]["recall_at_k"][2]
        print(f"synthetic top1={top1:.3f} recall@2={recall2:.3f}")
        assert top1 == 1.0, f"expected perfect top-1 on a trivially separable synthetic set, got {top1}"
        assert recall2 == 1.0, f"expected perfect recall@2, got {recall2}"

        # the cluster predictor must beat the global-frequency baseline: the two
        # synthetic prompt types use *disjoint* experts, so a single global list
        # can cover at most one type's pair.
        b1 = M.baseline_global_topk(profiles.global_expert_freq, STORED_TOP_K)
        b1_layer = np.tile(b1[None, :], (NUM_LAYERS, 1))
        b1_pred = M.broadcast_baseline(b1_layer, n)
        b1_report = M.evaluate_predictions(b1_pred, test, NUM_EXPERTS, ks=(1, 2, 4))
        b1_recall2 = b1_report["aggregate"]["recall_at_k"][2]
        print(f"baseline1 recall@2={b1_recall2:.3f}")
        assert recall2 > b1_recall2, "cluster predictor should beat the global-frequency baseline here"

    print("OK - synthetic pipeline smoke test passed")


if __name__ == "__main__":
    main()
