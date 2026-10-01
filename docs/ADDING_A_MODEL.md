# Adding a model

eai-studio never downloads a model on your behalf - it reads whatever's
already in your local Hugging Face cache (`~/.cache/huggingface/hub`), the
same cache `transformers`/`huggingface_hub` always use. This keeps load
latency predictable (no surprise multi-GB download mid-request) and makes
"which models are available" just "what's on disk."

## 1. Confirm the architecture is actually MoE (and supported)

eai-studio's whole value proposition - a fixed-budget expert cache,
chunked on-demand expert loading - only applies to Mixture-of-Experts
models. A dense model gets zero benefit from any of this (see
`eai-poc/scripts/dense_layer_streaming_experiment.py` - no sparsity to
exploit, so there's nothing to cache selectively). Loading a dense model
through eai-studio's engine would just mean "load the whole thing," no
different from (and slower than) using `transformers` directly.

Currently supported MoE architectures (`eai_studio/engine/chunked_loader.py`):

| `config.model_type` | Loader | Notes |
|---|---|---|
| `olmoe` | `load_chunked_olmoe` | No shared expert |
| `qwen3_moe` | `load_chunked_qwen3moe` | No shared expert; same math as OLMoE |
| `qwen2_moe` | `load_chunked_qwen2moe` | Has an always-active shared expert, handled separately |

Check `eai_studio/ollama/moe_filter.py`'s `KNOWN_MOE_MODELS` list for every
model family already confirmed to work. If your model's family isn't
there, check its `config.json`'s `model_type` against the table above
first - if it matches one of the three, it'll likely work even if not
explicitly listed (add it to `KNOWN_MOE_MODELS` once confirmed, so the UI
stops calling it "likely" and starts calling it "known").

If it's a genuinely new MoE architecture (not one of the three above), see
"Adding a new architecture" below before anything else - loading it through
the chunked loader as-is will fail at the `AutoConfig`/`AutoModelForCausalLM`
step.

## 2. Download it

Nothing eai-studio-specific here - just populate the HF cache normally:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")  # downloads, then discard - we only needed it cached
```

or the `huggingface-cli download` CLI. Either way, once it's under
`~/.cache/huggingface/hub/models--<org>--<name>/`, eai-studio can see it.

## 3. Load it

```bash
curl -X POST http://127.0.0.1:8877/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "allenai/OLMoE-1B-7B-0924", "cache_gb": 4.0, "policy": "hybrid", "device": "cpu"}'
```

- **`cache_gb`**: the expert cache budget, NOT the model's total size - this
  is the number that actually has to fit in your RAM (or VRAM, if
  `device="cuda"`). See `eai-poc/docs/cache_policy_matrix.md` for how to
  pick a sane starting point relative to a model's `(top_k * num_layers *
  avg_expert_bytes)` - going below that floor makes every policy perform
  badly, not just a slow one.
- **`policy`**: `reactive` (no prediction, FIFO), `lru`, `lfu`, or `hybrid`
  (weighted combination of predicted probability, coactivation, frequency,
  recency, and load cost - see `eai-poc/docs/cache_policy_matrix.md` for
  what's actually been validated so far). `hybrid` is the default and
  currently the most configurable; it is NOT unconditionally the best
  policy in every regime - the research PoC found real cases where plain
  `lfu` wins (a warm, long-running session) and real cases where
  `eai_coactivation`-style prediction wins (an under-provisioned, cold
  cache). Default weights are the best combination from eai-poc's own
  small sweep so far, not an exhaustively-tuned optimum.
- **`device`**: `"cpu"` or `"cuda"`. See eai-poc's GPU findings
  (`docs/benchmark_findings_current.md` §4/§8, §6 in this doc's README)
  before assuming `cuda` is faster for your model/batch size - it measured
  worse for a small model at batch=1 on this project's own test machine.

## 4. Adding a new architecture (not OLMoE/Qwen2-MoE/Qwen3-MoE)

This is real engineering, not a config change - see
`eai_studio/engine/chunked_loader.py`'s `_load_chunked_whole_block` and
`load_chunked_qwen2moe` for the two existing patterns:

1. **Confirm the checkpoint stores experts as separate named tensors**
   (`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` or
   similar) - inspect the `.safetensors` file's own header/index, not just
   the loaded model's structure (a loaded model may present experts as one
   stacked tensor even when the checkpoint itself stores them separately,
   which is exactly what makes chunked loading possible).
2. **Read the reference implementation's router + expert FFN math** (the
   model's own `modeling_<arch>.py` in `transformers`) and confirm whether
   there's an always-active shared expert alongside the routed ones
   (Qwen2-MoE pattern - needs the surgical `ChunkedQwen2MoeExperts`
   approach, not the whole-block replacement) or not (OLMoE/Qwen3-MoE
   pattern - the whole `.mlp` block can be replaced).
3. **Write `load_chunked_<name>`**, following whichever existing pattern
   matches, and register it in `chunked_loader.py`'s
   `_LOADERS_BY_MODEL_TYPE` dict keyed on the config's own `model_type`.
4. **Verify correctness before trusting anything else** - compare the
   chunked loader's router selections against a reference
   `output_router_logits=True` forward pass on the same input, same device,
   same dtype, for every layer. This project's whole track record of
   catching real bugs (see `eai-poc/README.md`'s Round 4-7 history) comes
   from never skipping this step.

## 5. Tuning the cache for your specific model

Once loaded, `GET /api/models/loaded` reports `hit_rate`,
`eviction_churn`, `reload_amplification`, and `useful_io_ratio` for the
currently-loaded instance - if `hit_rate` is near 0% and `eviction_churn`
is near 1.0 (an eviction on almost every access), your `cache_gb` is likely
below that model's minimum per-step working set
(`top_k * num_layers * avg_expert_bytes`) - raise the budget before
concluding the policy itself is bad. See
`eai-poc/docs/benchmark_findings_2026-09-19.md`'s Qwen3-30B incident for a
real example of exactly this failure mode.
