"""The .eai file format: a safetensors container.

Why safetensors over .npz or msgpack: it's a flat map of named tensors with a
small string-only metadata header, backed by mmap on load (zero-copy, no pickle,
no arbitrary code execution) - which is exactly "fast read, small file, easy
partial/mmap loading, simple implementation" from the spec, and the dependency
is already required by `transformers`.

Required arrays (per spec): centroids, top_experts, probabilities,
observation_counts. This implementation adds a few more named tensors
(confidence, coactivation, global_expert_freq) that section 4/7 also ask for -
safetensors' flat tensor map makes that a pure addition, not a schema break.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .profile import ActivationProfiles

EAI_VERSION = "0.1.0"


@dataclass
class EaiMetadata:
    version: str
    model_id: str
    architecture: str
    num_layers: int
    num_experts: int
    top_k: int  # the model's own router top-k (experts actually activated per token)
    quantization: str  # "none", "4bit", or "8bit" - ground-truth fidelity caveat, see README
    stored_top_k: int  # candidates kept per (cluster, layer) in this index
    fingerprint_method: str
    fingerprint_dimension: int
    granularity: str  # "prompt" or "token" - what a cluster/prediction corresponds to
    number_of_clusters: int
    created_at: str


@dataclass
class EaiIndex:
    metadata: EaiMetadata
    centroids: np.ndarray  # (K, fp_dim) float32
    top_experts: np.ndarray  # (K, L, stored_top_k) int32
    probabilities: np.ndarray  # (K, L, stored_top_k) float32
    observation_counts: np.ndarray  # (K, L) int32
    confidence: np.ndarray  # (K, L) float32
    coactivation: np.ndarray  # (K, L, stored_top_k, stored_top_k) float32
    global_expert_freq: np.ndarray  # (L, num_experts) float32


def save_index(path: str, metadata: EaiMetadata, profiles: ActivationProfiles, centroids: np.ndarray) -> None:
    tensors = {
        "centroids": np.ascontiguousarray(centroids, dtype=np.float32),
        "top_experts": np.ascontiguousarray(profiles.top_experts, dtype=np.int32),
        "probabilities": np.ascontiguousarray(profiles.probabilities, dtype=np.float32),
        "observation_counts": np.ascontiguousarray(profiles.observation_counts, dtype=np.int32),
        "confidence": np.ascontiguousarray(profiles.confidence, dtype=np.float32),
        "coactivation": np.ascontiguousarray(profiles.coactivation, dtype=np.float32),
        "global_expert_freq": np.ascontiguousarray(profiles.global_expert_freq, dtype=np.float32),
    }
    meta_dict = asdict(metadata)
    str_meta = {k: str(v) for k, v in meta_dict.items()}
    str_meta["meta_json"] = json.dumps(meta_dict)
    save_file(tensors, path, metadata=str_meta)


def load_index(path: str) -> EaiIndex:
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        meta_raw = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    if "meta_json" not in meta_raw:
        raise ValueError(f"{path} has no meta_json header - not a valid .eai file")
    meta_dict = json.loads(meta_raw["meta_json"])
    metadata = EaiMetadata(**meta_dict)

    return EaiIndex(
        metadata=metadata,
        centroids=tensors["centroids"],
        top_experts=tensors["top_experts"],
        probabilities=tensors["probabilities"],
        observation_counts=tensors["observation_counts"],
        confidence=tensors["confidence"],
        coactivation=tensors["coactivation"],
        global_expert_freq=tensors["global_expert_freq"],
    )
