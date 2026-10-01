"""Router instrumentation: run the model and record which experts fire per layer/token.

We don't hook into internal router modules. Every MoE model in Transformers that
follows the Mixtral convention (Mixtral, Qwen2/3-MoE, OLMoE, GraniteMoE, DBRX, ...)
already computes and can return per-layer router logits via
`model(..., output_router_logits=True)`. Reusing that supported API is more robust
than patching internal gate modules, and gives the exact tensor the model's own
top-k selection is computed from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch


def load_prompts_jsonl(path: str) -> list[dict]:
    """Read a {id, category, text} per line JSONL prompt dataset."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


@dataclass
class ModelHandle:
    """A loaded model + tokenizer plus the MoE shape facts tracing/profile code needs."""

    model: object
    tokenizer: object
    model_id: str
    architecture: str
    num_layers: int
    num_experts: int
    top_k: int
    hidden_size: int
    quantization: str  # "none", "4bit", or "8bit"

    @property
    def input_device(self):
        return next(self.model.parameters()).device


def load_model(
    model_id: str,
    device: str | None = None,
    quantize: str | None = None,
    offload_folder: str | None = None,
    max_memory: dict | None = None,
) -> ModelHandle:
    """Load a MoE causal LM, by default in full bf16.

    device=None lets `device_map="auto"` (accelerate) split the model across
    GPU/CPU automatically. By default we never quantize: this project's ground
    truth is "which experts the router actually picked", and quantizing router
    weights perturbs that ground truth. Full precision (bf16, as released) is
    slower on constrained hardware but keeps evaluation honest.

    `quantize` ("4bit" or "8bit") is an explicit, documented exception for
    models too large to fit in RAM+VRAM at full precision. Traces collected
    this way are labeled accordingly (see TraceSet.quantization) - treat
    their absolute numbers as directional, not as precise as the unquantized
    runs. In practice, on this project's dev machine, on-the-fly bitsandbytes
    4-bit CPU quantization of a 61GB bf16 checkpoint reliably triggered a
    runaway memory spike during loading (reproduced twice, killed by an
    external RAM watchdog both times, well before reaching the ~15GB the
    quantized model should occupy once loaded) - a loading-time problem, not
    a quantized-inference one. `offload_folder`/`max_memory` below sidesteps
    it entirely by not quantizing at all.

    `offload_folder` + `max_memory` (e.g. `max_memory={"cpu": "24GiB"}`) let
    accelerate keep only `max_memory`'s worth of weights resident in RAM at
    once, spilling the rest to disk-backed files under `offload_folder` and
    streaming them back in per layer during the forward pass. Slower (disk
    I/O per layer) but keeps peak RAM bounded and exact - the model still
    runs in full bf16, so this has none of the fidelity caveats quantization
    does. The right tool for "big model, not enough RAM, one-off tracing run"
    rather than "big model, need it fast and resident."
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16)
    if quantize is not None:
        from transformers import BitsAndBytesConfig

        if quantize == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantize == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"unknown quantize option: {quantize} (known: 4bit, 8bit)")
    if quantize is not None:
        # bitsandbytes rejects mixed CPU/GPU dispatch for standard 4bit/8bit
        # (see quantizer_bnb_4bit.py's validate_environment) unless
        # llm_int8_enable_fp32_cpu_offload=True, which keeps the CPU-offloaded
        # part in fp32 - defeating the point of quantizing to fit in RAM at
        # all. So a quantized load always pins every module to one device.
        kwargs["device_map"] = {"": device or "cpu"}
    elif offload_folder is not None or max_memory is not None:
        kwargs["device_map"] = "auto"
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            kwargs["offload_folder"] = offload_folder
    elif device is None:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device is not None and quantize is None and offload_folder is None:
        model = model.to(device)
    model.eval()

    cfg = model.config
    num_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    top_k = getattr(cfg, "num_experts_per_tok", None)
    if num_experts is None or top_k is None:
        raise ValueError(
            f"Model config for {model_id} does not expose num_experts/num_experts_per_tok "
            "- this model's MoE layer doesn't follow the standard Transformers convention."
        )

    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        architecture=type(model).__name__,
        num_layers=cfg.num_hidden_layers,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=cfg.hidden_size,
        quantization=quantize or "none",
    )


@dataclass
class PromptTrace:
    """Ground-truth routing trace for a single prompt, one forward pass."""

    prompt_id: str
    category: str
    num_tokens: int
    fingerprint_hidden: np.ndarray  # (hidden_size,) float32 - mean-pooled hidden state at a fixed early layer
    fingerprint_hidden_tokens: np.ndarray  # (num_tokens, hidden_size) float32 - same layer, per-token (unpooled)
    selected_experts: np.ndarray  # (num_layers, num_tokens, top_k) int32
    router_weights: np.ndarray  # (num_layers, num_tokens, top_k) float32
    forward_pass_seconds: float


def trace_prompt(handle: ModelHandle, prompt_id: str, category: str, text: str,
                  fingerprint_layer: int = 2) -> PromptTrace:
    """Run one forward pass over `text` and extract the routing ground truth.

    `fingerprint_layer` picks which entry of `hidden_states` (0 = embeddings output,
    i = output of transformer block i) is mean-pooled into the prompt fingerprint.
    A shallow layer keeps the fingerprint cheap relative to running the full model.
    """
    tokenizer = handle.tokenizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(handle.input_device) for k, v in inputs.items()}
    num_tokens = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = handle.model(**inputs, output_router_logits=True, output_hidden_states=True)
    elapsed = time.perf_counter() - t0

    if out.router_logits is None or out.router_logits[0] is None:
        raise RuntimeError(
            f"{handle.model_id} did not return router_logits. "
            "output_router_logits=True is likely unsupported for this architecture."
        )

    selected = np.empty((handle.num_layers, num_tokens, handle.top_k), dtype=np.int32)
    weights = np.empty((handle.num_layers, num_tokens, handle.top_k), dtype=np.float32)
    for layer_idx, layer_logits in enumerate(out.router_logits):
        # layer_logits: [num_tokens, num_experts] (batch folded in, batch=1 here)
        probs = torch.softmax(layer_logits.float(), dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=handle.top_k, dim=-1)
        selected[layer_idx] = topk_idx.cpu().numpy()
        weights[layer_idx] = topk_vals.cpu().numpy()

    fp_hidden_tokens = out.hidden_states[fingerprint_layer][0].float().cpu().numpy()  # (num_tokens, hidden)
    fp_hidden = fp_hidden_tokens.mean(axis=0)

    return PromptTrace(
        prompt_id=prompt_id,
        category=category,
        num_tokens=num_tokens,
        fingerprint_hidden=fp_hidden,
        fingerprint_hidden_tokens=fp_hidden_tokens,
        selected_experts=selected,
        router_weights=weights,
        forward_pass_seconds=elapsed,
    )


def trace_prompt_chunked(model, blocks, tokenizer, input_device, prompt_id: str, category: str, text: str,
                          top_k: int, fingerprint_layer: int = 2) -> PromptTrace:
    """Same contract as `trace_prompt`, but for a model built by
    `eai.chunked_expert_loader.load_chunked_model` instead of a stock
    `AutoModelForCausalLM`. Needed because the chunked loader replaces each
    layer's router-adjacent module with a custom one that the
    `output_router_logits=True` capture mechanism (`OutputRecorder`, keyed to
    the *original* router class) doesn't know how to hook - so ground truth
    is read directly off each block's own `last_selected`/
    `last_selected_weights` (already computed for the correctness checks in
    scripts/chunked_inference_experiment.py) instead of `out.router_logits`.

    This is how trace collection becomes possible for models too large to
    load via the normal `eai.tracing.load_model` path at all (see README,
    "Attempting a 30B-class model") - the chunked loader never needs the
    full model resident, so it can gather ground truth where the accelerate/
    bitsandbytes path couldn't even finish loading.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    num_tokens = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    elapsed = time.perf_counter() - t0

    num_layers = len(blocks)
    selected = np.empty((num_layers, num_tokens, top_k), dtype=np.int32)
    weights = np.empty((num_layers, num_tokens, top_k), dtype=np.float32)
    for layer_idx, block in enumerate(blocks):
        if block.last_selected is None or block.last_selected_weights is None:
            raise RuntimeError(
                f"{prompt_id}: layer {layer_idx} never recorded a router selection - "
                "the forward pass didn't reach this block, or the chunked loader is misconfigured."
            )
        selected[layer_idx] = block.last_selected.cpu().numpy()
        weights[layer_idx] = block.last_selected_weights.float().cpu().numpy()

    fp_hidden_tokens = out.hidden_states[fingerprint_layer][0].float().cpu().numpy()
    fp_hidden = fp_hidden_tokens.mean(axis=0)

    return PromptTrace(
        prompt_id=prompt_id,
        category=category,
        num_tokens=num_tokens,
        fingerprint_hidden=fp_hidden,
        fingerprint_hidden_tokens=fp_hidden_tokens,
        selected_experts=selected,
        router_weights=weights,
        forward_pass_seconds=elapsed,
    )


@dataclass
class TraceSet:
    """Concatenated traces for a whole prompt split (train or test).

    Token rows from every prompt are concatenated along one axis; `token_prompt_idx`
    and `token_position` reconstruct which (prompt, local token) each row came from,
    so the full per-record schema from the spec -
    {prompt_id, layer_id, token_position, selected_expert_ids, router_scores} -
    is recoverable without keeping a separate Python object per token.
    """

    prompt_ids: np.ndarray  # (N,) str
    categories: np.ndarray  # (N,) str
    num_tokens: np.ndarray  # (N,) int32
    fingerprint_layer: int
    fingerprints_hidden: np.ndarray  # (N, hidden_size) float32 - mean-pooled hidden state per prompt
    fingerprints_hidden_tokens: np.ndarray  # (total_tokens, hidden_size) float32 - same layer, per-token
    token_prompt_idx: np.ndarray  # (total_tokens,) int32, index into prompt_ids
    token_position: np.ndarray  # (total_tokens,) int32, local position within its prompt
    selected_experts: np.ndarray  # (num_layers, total_tokens, top_k) int32
    router_weights: np.ndarray  # (num_layers, total_tokens, top_k) float32
    forward_pass_seconds: np.ndarray  # (N,) float32
    model_id: str
    architecture: str
    num_experts: int
    top_k: int
    quantization: str

    @staticmethod
    def from_traces(traces: list[PromptTrace], fingerprint_layer: int, handle: "ModelHandle") -> "TraceSet":
        if not traces:
            raise ValueError("no traces to assemble")

        prompt_ids, categories, num_tokens_list, fps, fwd_secs = [], [], [], [], []
        token_prompt_idx_parts, token_position_parts = [], []
        selected_parts, weights_parts, fp_tokens_parts = [], [], []

        for i, tr in enumerate(traces):
            prompt_ids.append(tr.prompt_id)
            categories.append(tr.category)
            num_tokens_list.append(tr.num_tokens)
            fps.append(tr.fingerprint_hidden)
            fwd_secs.append(tr.forward_pass_seconds)
            token_prompt_idx_parts.append(np.full(tr.num_tokens, i, dtype=np.int32))
            token_position_parts.append(np.arange(tr.num_tokens, dtype=np.int32))
            selected_parts.append(tr.selected_experts)
            weights_parts.append(tr.router_weights)
            fp_tokens_parts.append(tr.fingerprint_hidden_tokens)

        return TraceSet(
            prompt_ids=np.array(prompt_ids, dtype=str),
            categories=np.array(categories, dtype=str),
            num_tokens=np.array(num_tokens_list, dtype=np.int32),
            fingerprint_layer=fingerprint_layer,
            fingerprints_hidden=np.stack(fps, axis=0).astype(np.float32),
            fingerprints_hidden_tokens=np.concatenate(fp_tokens_parts, axis=0).astype(np.float32),
            token_prompt_idx=np.concatenate(token_prompt_idx_parts),
            token_position=np.concatenate(token_position_parts),
            selected_experts=np.concatenate(selected_parts, axis=1),
            router_weights=np.concatenate(weights_parts, axis=1),
            forward_pass_seconds=np.array(fwd_secs, dtype=np.float32),
            model_id=handle.model_id,
            architecture=handle.architecture,
            num_experts=handle.num_experts,
            top_k=handle.top_k,
            quantization=handle.quantization,
        )

    @staticmethod
    def concat(sets: list["TraceSet"]) -> "TraceSet":
        """Concatenate already-built TraceSets end to end (e.g. previously
        checkpointed progress + newly collected prompts) - used by
        scripts/collect_chunked.py to resume a collection run that got
        interrupted partway through, without redoing already-collected
        prompts. All sets must share fingerprint_layer/model_id/architecture/
        num_experts/top_k/quantization - they're the same collection run,
        just gathered in pieces.
        """
        if not sets:
            raise ValueError("no TraceSets to concatenate")
        if len(sets) == 1:
            return sets[0]
        first = sets[0]
        for s in sets[1:]:
            if s.fingerprint_layer != first.fingerprint_layer or s.model_id != first.model_id:
                raise ValueError("cannot concat TraceSets from different fingerprint_layer/model_id")

        token_prompt_idx_parts = []
        prompt_offset = 0
        for s in sets:
            token_prompt_idx_parts.append(s.token_prompt_idx + prompt_offset)
            prompt_offset += s.num_prompts

        return TraceSet(
            prompt_ids=np.concatenate([s.prompt_ids for s in sets]),
            categories=np.concatenate([s.categories for s in sets]),
            num_tokens=np.concatenate([s.num_tokens for s in sets]),
            fingerprint_layer=first.fingerprint_layer,
            fingerprints_hidden=np.concatenate([s.fingerprints_hidden for s in sets], axis=0),
            fingerprints_hidden_tokens=np.concatenate([s.fingerprints_hidden_tokens for s in sets], axis=0),
            token_prompt_idx=np.concatenate(token_prompt_idx_parts),
            token_position=np.concatenate([s.token_position for s in sets]),
            selected_experts=np.concatenate([s.selected_experts for s in sets], axis=1),
            router_weights=np.concatenate([s.router_weights for s in sets], axis=1),
            forward_pass_seconds=np.concatenate([s.forward_pass_seconds for s in sets]),
            model_id=first.model_id,
            architecture=first.architecture,
            num_experts=first.num_experts,
            top_k=first.top_k,
            quantization=first.quantization,
        )

    @property
    def num_prompts(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_layers(self) -> int:
        return self.selected_experts.shape[0]

    def mask_for_prompt(self, prompt_idx: int) -> np.ndarray:
        return self.token_prompt_idx == prompt_idx

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            prompt_ids=self.prompt_ids.astype(str),
            categories=self.categories.astype(str),
            num_tokens=self.num_tokens,
            fingerprint_layer=np.array(self.fingerprint_layer, dtype=np.int32),
            fingerprints_hidden=self.fingerprints_hidden,
            fingerprints_hidden_tokens=self.fingerprints_hidden_tokens,
            token_prompt_idx=self.token_prompt_idx,
            token_position=self.token_position,
            selected_experts=self.selected_experts,
            router_weights=self.router_weights,
            forward_pass_seconds=self.forward_pass_seconds,
            model_id=np.array(str(self.model_id)),
            architecture=np.array(str(self.architecture)),
            num_experts=np.array(self.num_experts, dtype=np.int32),
            top_k=np.array(self.top_k, dtype=np.int32),
            quantization=np.array(str(self.quantization)),
        )

    @staticmethod
    def load(path: str) -> "TraceSet":
        data = np.load(path, allow_pickle=False)
        quantization = str(data["quantization"]) if "quantization" in data else "none"
        return TraceSet(
            prompt_ids=data["prompt_ids"],
            categories=data["categories"],
            num_tokens=data["num_tokens"],
            fingerprint_layer=int(data["fingerprint_layer"]),
            fingerprints_hidden=data["fingerprints_hidden"],
            fingerprints_hidden_tokens=data["fingerprints_hidden_tokens"],
            token_prompt_idx=data["token_prompt_idx"],
            token_position=data["token_position"],
            selected_experts=data["selected_experts"],
            router_weights=data["router_weights"],
            forward_pass_seconds=data["forward_pass_seconds"],
            model_id=str(data["model_id"]),
            architecture=str(data["architecture"]),
            num_experts=int(data["num_experts"]),
            top_k=int(data["top_k"]),
            quantization=quantization,
        )
