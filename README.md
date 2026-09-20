# Expert Activation Index (EAI) - PoC

An external, portable, local index that learns which experts a Mixture-of-Experts
(MoE) model's router tends to activate for a given kind of prompt, and predicts
that activation *before* the model actually runs - the same spirit as
Profile-Guided Optimization, applied to expert routing.

**This PoC does not touch the router.** It never blocks, reorders, or influences
*which* experts the model actually uses - that's still decided by the real
router, unchanged, every time. [Round 4](#round-4-acting-on-the-prediction-not-just-scoring-it)
goes further than "just measure": it changes *when* expert weights get
loaded into memory, seeded by a prediction - but never what gets computed,
verified bit-exact against a fresh reference model.

The hypothesis under test:

> There is enough correlation between prompt characteristics and the experts a
> MoE model activates that a compact external index can predict a useful part
> of the execution path.

See [Does the hypothesis hold?](#does-the-hypothesis-hold) at the bottom for
the full verdict once you've run the pipeline. Short version: yes, on three
different models spanning 7B to 30B total parameters, once the index is
keyed on individual tokens rather than whole prompts (see
[Results](#results)) - and the underlying problem is real enough that a
dozen+ recent research papers attack it, though nobody ships this as an
installable tool yet (see [Related
work](#related-work-is-this-valuable-to-the-market)). Acting on the
prediction (not just scoring it) went through two more rounds: a whole-prompt
prefetch that worked correctly but barely saved memory
([Round 4](#round-4-acting-on-the-prediction-not-just-scoring-it)), then real
token-by-token streaming eviction that did -
[Round 5](#round-5-real-streaming-eviction---load-process-unload-per-token)
and [Round 6](#round-6-a-30b-model-actually-fitting-and-running-fast) - a
30B-class model that six different standard loading strategies couldn't even
get into memory now loads in ~2s and streams with ~9-10GB peak resident
memory, about 16% of its ~58GB expert weights.

## How it works

```
profiling (offline)                    inference (measured, router untouched)
--------------------                   -------------------------------------
prompt -> forward pass -----+          prompt -> fingerprint -> nearest
          (output_router_        |                centroid -> cluster_id
           logits=True)          |                -> activation profile
          |                      |                -> predicted top-K experts/layer
          v                      v
     ground-truth            model.eai                    |
     expert trace            (centroids +                  v
          |                   per-cluster,            run the model normally,
          v                   per-layer profiles)      compare prediction vs
   cluster prompts by                                  what the router actually
   fingerprint (KMeans)                                picked
          |
          v
   aggregate per (cluster, layer)
   activation profile -> model.eai
```

1. **Trace collector** ([eai/tracing.py](eai/tracing.py)) - runs the model with
   `output_router_logits=True` and records, per layer and token, which experts
   fired and their router (softmax) weight. We reuse the model's own supported
   output instead of patching internal gate modules - every Transformers MoE
   that follows the Mixtral convention (Mixtral, Qwen2/3-MoE, OLMoE, GraniteMoE,
   DBRX...) exposes this the same way, so it's robust across architectures.
2. **Prompt fingerprint** ([eai/fingerprint.py](eai/fingerprint.py)) - a cheap
   per-prompt vector. Two of the three options from the spec are implemented:
   - **A) hidden_state** (default): mean-pooled hidden state from a shallow
     layer (index 2) of the model itself. No extra model to load.
   - **C) hashing**: text feature-hashing (character n-grams), zero model
     computation - a sanity-check lower bound.
   - **B) separate embedding model**: not implemented. The spec asks to try A
     first and only add complexity if the metrics justify it - see
     [Limitations](#limitations--recommendations).
3. **Clustering** ([eai/clustering.py](eai/clustering.py)) - KMeans (or
   MiniBatchKMeans) over fingerprints, K configurable. Two granularities:
   cluster whole **prompts** (one fingerprint/prediction per prompt, mean-pooled
   over its tokens - the spec's original framing), or cluster individual
   **tokens** (one fingerprint/prediction per token, using that token's own,
   unpooled hidden state). `--granularity token` is the follow-up experiment
   this PoC's own first-round results pointed at, and it wins by a wide margin
   - see [Results](#results).
4. **Activation profile** ([eai/profile.py](eai/profile.py)) - per
   (cluster, layer): top experts by activation frequency, their frequency,
   observation count, a confidence score, and a small co-activation matrix.
5. **model.eai** ([eai/storage.py](eai/storage.py)) - see
   [File format](#file-format-modeleai).
6. **Predictor** ([eai/predictor.py](eai/predictor.py)) - fingerprint -> nearest
   centroid -> cluster's stored profile -> predicted top-K experts per layer.
7. **Metrics** ([eai/metrics.py](eai/metrics.py)) - top-1 accuracy, recall@K,
   precision@K, coverage, confidence, per-layer and aggregate results, against
   two frequency baselines.

## Model choice

Three models, from three different families and spanning 7B to 30B total
parameters, all traced and evaluated the same way, to check whether the
[token-granularity result](#results) generalizes or was an OLMoE fluke:

| model | license | layers | experts/layer | top-k | total / active params | size (bf16) |
|---|---|---|---|---|---|---|
| [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924) | Apache 2.0 | 16 | 64 | 8 | ~7B / ~1.3B | ~13.8 GB |
| [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat) | Apache 2.0 (Tongyi Qianwen) | 24 | 60 | 4 | ~14.3B / ~2.7B | ~28.6 GB |
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Apache 2.0 | 48 | 128 | 8 | ~30B / ~3B | ~58 GB (experts) |

Why the first two: both have **real, trained routing** (not a
randomly-initialized test model - the router has actually learned something,
which is the entire premise being tested), both are natively supported by
Transformers' `output_router_logits=True` with no hooks into internal
modules, and between them they cover two different routing shapes (64
experts/top-8/12.5% density vs. 60 experts/top-4/6.7% density) and two
unrelated model families/training runs. Neither fits this machine's 8GB GPU
in full precision; both load in bf16 via `device_map="auto"`
(`eai/tracing.py::load_model`), letting `accelerate` split layers across GPU
and CPU automatically. We deliberately never quantize for the *primary*
generalization test: this PoC's ground truth *is* "what the router actually
picked," and quantizing the router's own weights would perturb the very
thing being measured.

Why the third: it's the model the whole second half of this project exists
for. `eai/tracing.py::load_model` (the path above) cannot load it at all on
this machine - see
[Attempting a 30B-class model, round 1](#attempting-a-30b-class-model-round-1-a-well-documented-failure).
[eai/chunked_expert_loader.py](eai/chunked_expert_loader.py) can, and does -
see [Round 6](#round-6-a-30b-model-actually-fitting-and-running-fast).

### A real gotcha: `output_router_logits` doesn't work for every architecture

Getting to these two working models meant ruling out several others first -
worth documenting, because the failures are as informative as the successes:

- **`ibm-granite/granite-3.0-1b-a400m-instruct`** - in the installed
  Transformers version (5.9.0), `GraniteMoeForCausalLM` accepts
  `output_router_logits=True` and even computes an aux loss from it, but
  `GraniteMoeModel.forward` never actually collects router logits from its
  decoder layers into the output - so `output.router_logits` comes back
  `None` and instrumentation fails outright, not subtly.
- **`JetMoeForCausalLM`** - identical bug, same code pattern, likely dropped
  during the same "modular" code-sharing refactor as Granite.
- **`deepseek-ai/DeepSeek-V2-Lite`** (the built-in `deepseek_v2` architecture,
  not the gated custom-code `deepseek-moe-16b-chat` repo) - `output_router_logits`
  isn't even a parameter on `DeepseekV2Model.forward` in this version; the
  capability was never wired up at all for this architecture.
- **`microsoft/Phi-3.5-MoE-instruct`** and **`mistralai/Mixtral-8x7B`** - the
  mechanism (`OutputRecorder`-based, same pattern as Qwen3) checks out fine on
  a zero-download random-config smoke test, but the only real checkpoints
  ship at 84 GB / 93 GB bf16 - too large for this machine's 62GB RAM without
  quantization, and quantizing means giving up exactly the router-fidelity
  guarantee this PoC is built around. Left as future work with better
  hardware, not attempted here.
- **`Qwen/Qwen3-30B-A3B-Instruct-2507`** - mechanism verified working, and
  unlike the others this one got a real, sustained attempt (see
  [Attempting a 30B-class model](#attempting-a-30b-class-model-round-1-a-well-documented-failure)
  below) rather than being ruled out from a size number alone. Six different
  loading strategies across two operating systems all hit the same wall.
  Confirms the same conclusion as Phi-3.5-MoE/Mixtral, but with much more
  evidence behind it: this class of model needs a machine with real
  headroom, not a clever workaround.

Every one of these was ruled out or in with a **zero-download check** before
committing to any real download - construct a tiny randomly-initialized
config for the model's class and confirm `output.router_logits` actually comes
back non-`None`:

```python
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM  # swap for your model's classes

cfg = Qwen2MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=32,
                      shared_expert_intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
                      mlp_only_layers=[])
model = Qwen2MoeForCausalLM(cfg).eval()
out = model(torch.randint(0, 32, (1, 5)), output_router_logits=True)
assert out.router_logits is not None and out.router_logits[0] is not None
```

**Run that check before adding any new model** - it takes 30 seconds and
would have saved two wasted downloads (Granite, DeepSeek-V2-Lite) here. Per
the spec's own instruction ("prefer switching models over fragile hacks"),
none of these were patched around - broken instrumentation means a different
model, not a hook into private internals.

### Attempting a 30B-class model, round 1: a well-documented failure

**Update: a later attempt succeeded, using this project's own chunked loader
instead of standard Transformers loading - see
[Round 6](#round-6-a-30b-model-actually-fitting-and-running-fast) in Results.
This section is kept in full because *why* the standard path fails is real,
useful evidence in its own right, and directly explains why Round 6's
different approach was necessary rather than optional.**

This is worth writing up in detail rather than compressing to one line,
because the *how it failed* is itself evidence relevant to
[the market question](#related-work-is-this-valuable-to-the-market): the
papers proposing expert prefetch/caching all target this exact model size
class, precisely because it's where the memory bottleneck actually bites -
and it turns out that class of model is hard to even *load* for a one-off
research script on ordinary hardware, before any of the interesting routing
questions come into play.

Six strategies were tried, on a machine with an 8GB GPU and 62GB system RAM,
against `Qwen/Qwen3-30B-A3B-Instruct-2507` (~61GB in bf16, 531 safetensors
tensors across shards):

1. **4-bit quantization, `device_map="auto"`** - rejected immediately: standard
   bitsandbytes 4-bit refuses mixed CPU/GPU dispatch outright
   (`quantizer_bnb_4bit.py::validate_environment`).
2. **4-bit quantization, `device_map={"":"cpu"}`** - accepted, but loading
   crashed the process twice, RAM climbing to ~41GB free→0GB free before an
   external watchdog (see below) killed it at 23% and 54% of shards loaded.
3. **Full bf16, `accelerate` disk offload** (`max_memory={"cpu":"20GiB"}` +
   `offload_folder`) - crashed before a single shard reached disk; the RAM
   spike happens during shard *dispatch*, before `max_memory` accounting
   takes effect the way its documentation implies.
4. **Same 4-bit CPU approach, inside WSL2/Ubuntu instead of native Windows**
   (new distro installed, full Python/PyTorch/Transformers/accelerate/
   bitsandbytes stack rebuilt from scratch, GPU passthrough verified working,
   HF cache reused via `/mnt/c/...` to avoid re-downloading) - crashed again,
   but this time the *Linux kernel's own OOM killer* logged the exact moment
   and cause: `Killed process ... python ... anon-rss:43565696kB` - **41.5GB
   resident memory**, almost identical to the Windows-side peak. This was the
   key finding: **the same ~41-44GB peak shows up independent of the
   operating system**, which rules out "Windows-specific accelerate/bnb bug"
   as the explanation. Something in how a 531-shard, 61GB checkpoint gets
   quantized on CPU load holds roughly 2.7x the eventual ~15GB quantized
   footprint in memory at once, most likely several shards' raw bf16 tensors
   staying resident simultaneously while a multi-threaded loader outpaces the
   per-shard quantize-then-free cycle.
5. **Same, with WSL2's memory ceiling raised from 44GB to 49GB (20GB swap)** -
   got further (100% of shards loaded, further than any previous attempt),
   but produced no confirming output afterward and no new OOM log entry -
   an inconclusive result, not a clean success.
6. **Re-run of (5) with unbuffered stdout**, to settle the ambiguity - this
   attempt was killed by the *harness itself* ("system is running low on
   memory"), external to any of our own scripts, and host free RAM measured
   3.7GB immediately after - i.e. this one nearly took down the whole
   machine, not just the Python process.

**Safety net that made this a well-evidenced failure rather than a real
outage:** [scripts/ram_watchdog.py](scripts/ram_watchdog.py) polls free RAM
every 1.5s and force-kills any Python process over 3GB the instant free RAM
drops below a configurable floor (6GB in all of the above). It fired
correctly every time it was needed. Attempt 6 is the one case it didn't catch
in time - the harness's own out-of-band safety mechanism did instead. Every
attempt left the machine recoverable; none required a hard reboot.

**What was cleaned up afterward:** the 57GB download (deleted twice - once
after the initial Windows-only attempts, again after the WSL attempts),
`.wslconfig`'s memory override (removed, restoring WSL's safe default ~50%-of-host
cap so it doesn't affect other WSL/Docker Desktop usage going forward), and
the disk-offload scratch folder (never actually received any files - every
crash happened before offloading began). The WSL Ubuntu distro itself was
left installed (a few GB, reusable if this is retried later - a working
GPU-passthrough Linux Python environment is genuinely useful infrastructure
for a next attempt, not a mess to undo).

**The honest conclusion:** this is not "we didn't try hard enough." Six
independent strategies across two operating systems, informed by root-cause
diagnosis at each step (an actual kernel OOM log, not a guess), converged on
the same ~41-44GB peak-memory wall. That number matters for anyone trying to
reproduce this: **a 61GB-class MoE checkpoint needs meaningfully more than
its own quantized footprint in free RAM just to *load* via
Transformers+accelerate+bitsandbytes** - budget for roughly 3x the target
quantized size as a rule of thumb, not the target size itself. On a machine
with that headroom (or a GPU large enough to hold the model without CPU
offloading at all), this would very likely just work - nothing about the
*code* in this repo is the blocker.

## Project structure

```
eai-poc/
  eai/                  # library
    tracing.py           # load model, run forward pass, extract router ground truth
    fingerprint.py        # prompt -> cheap vector
    clustering.py         # fingerprints -> K clusters
    profile.py             # traces + clusters -> per-(cluster,layer) activation profile
    storage.py              # model.eai read/write (safetensors)
    predictor.py             # fingerprint -> predicted experts (does not touch the model)
    metrics.py                # accuracy/recall/precision/coverage + baselines
    chunked_expert_loader.py   # round 4/6: loads a model with 0 experts resident; OLMoE, Qwen2-MoE, Qwen3-MoE
  scripts/
    collect.py       # phase 2: trace train+test prompts -> artifacts/traces_{split}.npz (models that fit normally)
    collect_chunked.py # round 6: same, via the chunked loader - for models too large to load any other way
    build_index.py    # phase 3+4: cluster + profile + save -> artifacts/model.eai (architecture-agnostic)
    evaluate.py         # phase 5+6: predict on test set, report metrics vs baselines (architecture-agnostic)
    chunked_inference_experiment.py  # round 4: correctness + real memory, whole-prompt prefetch
    streaming_eviction_experiment.py  # round 5/6: token-by-token streaming, real eviction, 3 policies compared
    ram_watchdog.py                     # safety net for large-model load attempts - see Model choice
  data/
    prompts_train.jsonl  # 120 prompts, 10 categories (profiling/training)
    prompts_test.jsonl    # 40 prompts, same categories, disjoint from train
  artifacts/
    traces_train.npz   # transient - raw per-token routing traces, regenerable (OLMoE)
    traces_test.npz      # transient (OLMoE)
    model.eai              # the persisted index, OLMoE (granularity=prompt, the spec's default)
    model_token128.eai      # recommended: OLMoE, granularity=token - see Results
    model_hashing.eai        # OLMoE, baseline C comparison - see Results
    qwen15moe/                # second model's traces + indexes - see Results, round 3
    qwen3_30b/                  # third model (30B-class) - see Results, round 6
  tests/
    test_pipeline_synthetic.py  # end-to-end pipeline check on synthetic data, no model download
```

## Setup

Requires Python 3.12+, and enough disk space for the model (~14GB download,
cached under `~/.cache/huggingface`).

```bash
pip install -e .
```

## Usage

```bash
python scripts/collect.py        # ~15-25 min on CPU/mixed GPU+CPU: traces both splits
python scripts/build_index.py    # seconds: cluster + build profiles + save model.eai
python scripts/evaluate.py       # seconds: predict on test set, print the report
```

Useful flags (all optional, sensible defaults):

```bash
python scripts/collect.py --device cpu                         # force CPU
python scripts/build_index.py --clusters 24 --top-k-store 16   # tune K / candidates kept per cell
python scripts/build_index.py --fingerprint-method hashing     # try baseline C instead of A
python scripts/evaluate.py --ks 1,4,8,16                        # which K values to report

# recommended: token granularity clearly beats the spec's original per-prompt
# design (see Results) - cluster individual tokens instead of whole prompts
python scripts/build_index.py --granularity token --clusters 128 --out artifacts/model_token128.eai
python scripts/evaluate.py --index artifacts/model_token128.eai
```

`evaluate.py` reads `--granularity` back out of the index's own metadata, so
you never need to pass it again yourself - point it at whichever `.eai` file
you built.

## File format: model.eai

A [safetensors](https://github.com/huggingface/safetensors) container. Chosen
over `.npz` / msgpack because it's a flat named-tensor map with a small
string-only metadata header, mmap-backed on load (zero-copy, no pickle, no
arbitrary code execution on load) - "fast read, small file, easy partial/mmap
loading, simple implementation" in one already-required dependency
(`transformers` depends on it).

**Metadata** (string-valued header, plus a `meta_json` key with the same fields
properly typed):

| field | meaning |
|---|---|
| `version` | .eai schema version |
| `model_id` | source model (e.g. `allenai/OLMoE-1B-7B-0924`) |
| `architecture` | Transformers class name (e.g. `OlmoeForCausalLM`) |
| `num_layers` | MoE layers in the model |
| `num_experts` | experts per layer |
| `top_k` | the model's own router top-k (experts actually activated per token) |
| `stored_top_k` | candidates kept per (cluster, layer) *in this index* |
| `fingerprint_method` | `hidden_state` or `hashing` |
| `fingerprint_dimension` | fingerprint vector length |
| `granularity` | `prompt` (one fingerprint/prediction per prompt) or `token` (per token) |
| `number_of_clusters` | K |
| `created_at` | build timestamp |

**Tensors** (required by the spec: `centroids`, `top_experts`, `probabilities`,
`observation_counts`; this implementation adds three more the spec also asks
for in prose - a flat tensor map makes that a pure addition, not a schema
break):

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `centroids` | `(K, fingerprint_dim)` | f32 | cluster centroids |
| `top_experts` | `(K, L, stored_top_k)` | i32 | expert ids, ranked by frequency desc |
| `probabilities` | `(K, L, stored_top_k)` | f32 | activation frequency `[0,1]` per stored expert |
| `observation_counts` | `(K, L)` | i32 | token observations backing the estimate |
| `confidence` | `(K, L)` | f32 | mean probability over the stored top-K |
| `coactivation` | `(K, L, stored_top_k, stored_top_k)` | f32 | joint activation frequency among stored experts |
| `global_expert_freq` | `(L, num_experts)` | f32 | all-clusters frequency, used to compute the frequency baselines |

## Metrics

The fingerprint/cluster pipeline makes **one prediction per prompt per layer**,
not one per token - but ground truth is still scored **per token**, against
that token's own `selected_experts` (always exactly the model's `top_k` ids).
A prompt's single prediction is broadcast to every token in that prompt before
comparing.

This matters: an earlier version of this evaluation defined "actual" as the
*union* of every expert used anywhere in the prompt. That union saturates fast
- with 64 experts and 8 picked per token, a ~20-token prompt already touches
most of the expert vocabulary at least once, which makes recall@K tiny and
nearly identical for every predictor regardless of quality. Per-token ground
truth keeps the denominator fixed at the model's real `top_k` and directly
answers the question that matters for prefetch/caching: *"if we warmed the
predicted experts before running this prompt, what fraction of the router's
real per-token decisions would have been covered?"*

Reported by `scripts/evaluate.py`:
- **Top-1 accuracy**: is our #1 guess among the token's actually-selected
  experts.
- **Recall@K / Precision@K**: overlap between our top-K guesses and each
  token's actual `top_k` selection, averaged over all test tokens, for K in
  `--ks` (default 1, 4, 8).
- **Coverage**: fraction of test predictions backed by >= `--min-observations`
  (default 3) training-token observations in that cluster/layer cell, vs.
  falling back on a cold/sparse cell.
- **Confidence**: the index's own stored confidence for the predicted cell.
- **Per-layer and aggregate** breakdowns.
- **Index lookup time** (mean/p95/max, nearest-centroid search only) and the
  `.eai` file size.
- Three-way baseline comparison:
  1. globally most-frequent experts (one list, reused for every layer),
  2. most-frequent experts per layer,
  3. this project's cluster-based predictor.

## Results

Same model, same train/test traces, same evaluation code throughout:
`allenai/OLMoE-1B-7B-0924`, 120 train / 40 test prompts (10 categories),
906 test tokens.

### Round 1: cluster whole prompts (the spec's original design)

```
Model: allenai/OLMoE-1B-7B-0924
Layers: 16
Experts/layer: 64  (router top_k=8)

Index:
  clusters: 16
  fingerprint: hidden_state (2048d)
  stored candidates/cell: 16
  size: 0.41 MB

Test set: 40 prompts, 906 tokens

Prediction - cluster-based (ours, granularity=prompt):
  Top-1 accuracy: 81.6%
  Recall@1: 10.2%    Precision@1: 81.6%
  Recall@4: 28.6%    Precision@4: 57.1%
  Recall@8: 43.6%    Precision@8: 43.6%
  Coverage (>= 3 train obs): 100.0%
  Mean prediction confidence: 0.325

Baseline 1 - globally most-frequent experts (same list, every layer):
  Top-1 accuracy: 21.7%
  Recall@8: 18.9%    Precision@8: 18.9%

Baseline 2 - most-frequent experts per layer:
  Top-1 accuracy: 80.7%
  Recall@8: 42.6%    Precision@8: 42.6%

Index lookup (nearest-centroid search only):
  mean: 0.019 ms   p95: 0.031 ms   max: 0.045 ms
```

Two prompt-level follow-ups (same traces, before moving to round 2):

| variant | Top-1 | Recall@8 |
|---|---|---|
| fingerprint = hidden_state, layer 2 (default above) | 81.6% | 43.6% |
| fingerprint = hidden_state, layer 8 (deeper) | 82.3% | 45.2% |
| fingerprint = hashing, 256d (option C, no model at all) | 81.4% | 43.9% |
| baseline 2 - per-layer frequency (no prompt info at all) | 80.7% | 42.6% |
| baseline 1 - global frequency (no layer info either) | 21.7% | 18.9% |

Verdict at this point: barely better than baseline 2. Whatever a whole-prompt
fingerprint captures, a real embedding and a free text hash capture it about
equally well - see the original analysis this prompted in
[Limitations](#limitations--recommendations).

### Round 2: cluster individual tokens instead (the follow-up experiment)

Same fingerprint (hidden state, layer 2), same K sweep, but now every *token*
gets its own fingerprint (its own, unpooled hidden state - already
contextualized by everything before it, since the model is causal) and its
own prediction, instead of one fingerprint/prediction shared by the whole
prompt:

| K (clusters) | file size | Top-1 | Recall@4 | Recall@8 | Coverage |
|---|---|---|---|---|---|
| 64 | 1.64 MB | 94.1% | 40.1% | 65.6% | 100.0% |
| 128 | 3.27 MB | 94.1% | 40.6% | 67.1% | 100.0% |
| 256 | 6.54 MB | 93.4% | 40.9% | 68.1% | 99.0% |

vs. baseline 2 (per-layer frequency, no token info at all): **80.7% top-1 /
42.6% recall@8** - the same baseline round 1 could barely beat.

Full report at K=128:

```
Prediction - cluster-based (ours, granularity=token):
  Top-1 accuracy: 94.1%
  Recall@1: 11.8%    Precision@1: 94.1%
  Recall@4: 40.6%    Precision@4: 81.2%
  Recall@8: 67.1%    Precision@8: 67.1%
  Coverage (>= 3 train obs): 100.0%
  Mean prediction confidence: 0.434

Per-layer Recall@8 (ours vs baseline-2), selected layers:
  layer  0: ours=  56.0%   baseline2=  47.6%
  layer  8: ours=  66.5%   baseline2=  44.1%
  layer 15: ours=  80.6%   baseline2=  58.2%

Index lookup (nearest-centroid search only, K=64 measured):
  mean: 0.076 ms   p95: 0.096 ms   max: 0.282 ms
```

Every single layer improves, by a wide and consistent margin - not just the
average. Returns diminish past K~128 (256 adds file size and a slightly
sparser tail of clusters - coverage drops to 99% - for +1 point of recall@8),
so K=128 is the sweet spot on this dataset, not 256.

`.eai` file size stays small either way (0.3-6.5 MB across every variant
tried, well under the illustrative 8.4 MB in the spec's example - our
dataset/K are smaller than that example implies). Index lookup overhead is
still effectively free relative to a forward pass (~0.003% at its highest,
K=64).

### Round 3: does this generalize to a different model? Yes.

Same pipeline, same code, same 120/40 train/test split, run against
[Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)
instead - a different model family, a different routing shape (60 experts,
top-4, 24 layers vs. OLMoE's 64/top-8/16), never tuned for or seen during
round 1/2's development:

| | Top-1 | Recall@4 | Recall@8 |
|---|---|---|---|
| ours, granularity=**prompt** (K=16) | 25.4% | 18.2%\* | 30.6% |
| ours, granularity=**token** (K=64) | **70.7%** | 50.6% | **67.9%** |
| baseline 2 - per-layer frequency (no prompt/token info) | 24.1% | 17.4%\* | 29.0% |
| baseline 1 - global frequency (no layer info either) | 9.2% | 8.7%\* | 16.6% |

\* precision@4 equals recall@4 here because Qwen1.5-MoE's router top_k is 4 -
recall@k and precision@k coincide exactly when k matches the model's own top_k.

The pattern from OLMoE reproduces almost exactly: prompt-granularity barely
beats the zero-information baseline (25.4% vs 24.1% top-1), token-granularity
blows past it by a wide margin (70.7% vs 24.1% top-1, +38.9 points recall@8).
Every one of Qwen1.5-MoE's 24 layers improved under token granularity, same
as OLMoE's 16. Two unrelated model families, two different expert-count/top-k
shapes, the same qualitative result - this is not an OLMoE-specific artifact.

One real difference: Qwen1.5-MoE's absolute numbers are lower across the
board than OLMoE's (67.9% vs 67.1%+ recall@8 is close, but top-1 94.1% vs
70.7% is a real gap). Two candidate explanations, not distinguished by this
PoC: (a) top-4 of 60 experts (6.7% density) is a harder target to guess
completely right than top-8 of 64 (12.5% density) - fewer correct slots to
land in per token; (b) Qwen1.5-MoE's shared-expert design (`shared_expert`
alongside the routed experts, present in its architecture but not modeled by
our profile/predictor) means part of what a token "uses" isn't captured by
`selected_experts` at all in the traces. Worth resolving before quoting a
single cross-model accuracy number as if it were architecture-independent.

### Round 4: acting on the prediction, not just scoring it

Rounds 1-3 all answer the same question - *could* we have guessed the right
experts? - without ever changing what actually happens when the model runs.
This round builds the piece that acts on the guess:
[eai/chunked_expert_loader.py](eai/chunked_expert_loader.py) loads OLMoE with
**zero experts resident** and a `prefetch(layer, expert_ids)` call the EAI
predictor feeds into before each forward pass, plus a reactive on-demand
fetch for anything the real router needed that wasn't prefetched. Run with
[scripts/chunked_inference_experiment.py](scripts/chunked_inference_experiment.py).

**Why this bypasses `AutoModelForCausalLM`.** The installed Transformers
version stores each layer's 64 experts as one dense `(64, ...)` tensor
(`OlmoeExperts.gate_up_proj`/`down_proj`), built from the checkpoint's
original per-expert tensors via an internal conversion step
(`@use_experts_implementation`) that isn't a safe public hook to intercept -
and a single dense tensor can't have "some experts resident, some not"
anyway, since PyTorch tensors are contiguous. Confirmed by direct
inspection that the *checkpoint itself* stores every expert as a separately
named tensor
(`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`) - exactly
the shape selective loading needs - `ChunkedExpertBlock` reads those
directly and replaces each layer's MoE block wholesale, while attention,
norms, embeddings, rotary and the router gate reuse the real, unmodified
Transformers modules (they're small and not the part in question -
reimplementing them would add risk without addressing anything).

**Correctness, verified rigorously, not assumed.** A first attempt compared
against the pre-collected `traces_test.npz` and got a confusing ~35% layer
match rate. That trace was collected under `device_map="auto"` (mixed
GPU+CPU); comparing a CPU-only reimplementation against it conflates two
different questions. The fix: load a fresh, full reference model in the
*same process, same device, same dtype* and compare directly. At **float32**,
every one of 16 layers matched the reference **exactly**, zero mismatches -
proof the router and expert FFN math are bit-correct, not merely "close." At
**bf16** (the dtype the rest of this PoC uses), 94.4% of (token, layer) pairs
matched exactly, 5.6% were a *near-tie* (exactly one expert swapped, always
between two experts whose router probabilities were nearly equal), and 0.0%
were a real mismatch (2+ experts differing). The near-ties are an expected,
well-understood artifact - the reference computes gate/up via one *fused*
matmul, this module computes them via two *separate* matmuls read from the
checkpoint's original tensors; mathematically identical, but bf16's ~3
decimal digits of precision can round a fused vs. split computation
differently enough to flip a genuinely-tied top-8-of-64 boundary. That this
shows up as *exactly* one swapped expert, only when the swap partners'
probabilities are nearly equal, and disappears entirely at float32, is the
signature of expected floating-point non-associativity - not a logic bug.

**Real memory measured, and a real limitation found.** Running the actual
predicted-and-prefetched model against 20 test prompts:

```
Per-prompt resident expert memory (evicted between prompts, 20 prompts):
  mean: 11450.4 MB   min: 10909.4 MB   max: 12129.9 MB
  vs. full model's expert weights: 12882 MB (88.9% of full, on average, per single prompt)
```

Only an ~11% reduction per prompt - far short of the ~5.4x the theoretical
math in [Model choice](#model-choice) suggested. Before blaming the
predictor, this was checked against a **perfect oracle**: using the real
ground truth (not a prediction at all), a single ~20-token test prompt
*already* touches a mean of 43.8 of 64 experts per layer (68%) just from
token-to-token diversity **within one prompt** - different tokens in the same
prompt legitimately need different experts. That ceiling, not prediction
accuracy, is most of what limits this round's memory number; our predictor's
88.9% sits a bit above the 68% oracle ceiling (`stored_top_k=16` is generous,
and the union is taken per-prompt rather than tightened per-token), but even
a flawless predictor would have hit noticeably less than the ~12.5%-of-full
figure a naive "each token needs 8 of 64" framing implies.

**Why, and what the fix looks like.** This round prefetches the *union* of
every token's predicted experts *before running the whole prompt at once* -
whatever any token might need, all loaded up front. That's exactly the wrong
granularity for the memory question: it recreates the same "prompt vs. token"
lesson from [Round 1 vs. Round 2](#round-2-cluster-individual-tokens-instead-the-follow-up-experiment)
one level up the stack. The two-stage designs in the actual research
literature ([ADEPT](https://www.researchgate.net/publication/400888358_Two-Stage_Expert_Offloading_for_Domain-Aware_MoE_Inference),
named explicitly in
[Related work](#related-work-is-this-valuable-to-the-market)) split this into
domain-aware prefetch *for the prefill phase* plus **locality-aware,
token-by-token eviction during decode** - keeping only a small sliding window
of experts resident as generation moves token to token, not the whole
prompt's union at once. This round built the first half only. The second
half - incremental, per-token forward passes with real eviction between
steps - is real additional engineering (a token-at-a-time generation loop
with KV caching, not the single batched forward pass this PoC's tracing
methodology uses throughout) and is exactly where the *actual* memory
reduction would come from. Correctly scoped out of this round, not because
it isn't valuable, but because proving the mechanism *works correctly* first
was the higher-priority, more foundational claim to get right - which this
round did.

**What this round did prove, concretely:** the loading mechanism is real,
correct (bit-exact at float32, explainably close at bf16), and functional -
zero experts resident at load time, predictions from the same `.eai` index
used throughout this PoC actually determine what gets read off disk and
when, with a 99.7%+ hit rate against the real router's needs in the
no-eviction/cumulative scenario. What it did *not* prove is that whole-prompt
prefetch alone delivers large memory savings - it doesn't, and now there's a
measured, oracle-checked number (68% ceiling) explaining exactly why, and a
specific, scoped next step (token-level streaming eviction) instead of a
vague "future work" gesture.

### Round 5: real streaming eviction - load, process, unload, per token

Round 4 identified the fix but didn't build it: prefetch a whole prompt's
union of predicted experts up front, and you've already lost most of the
possible memory savings, because different tokens in the same prompt
legitimately need different experts. This round builds the actual fix -
[scripts/streaming_eviction_experiment.py](scripts/streaming_eviction_experiment.py)
replays a prompt **one token at a time**, with a real KV cache (exactly like
generation), evicting each layer's cache down to (ideally) just what the
*current* token needs before every step - literal load -> process -> unload,
the pattern
[github.com/josesilva05/kimi-k3-in-c](https://github.com/josesilva05/kimi-k3-in-c)
calls "expert streaming" for a similar reason (see that project's own
techniques, summarized in [Model choice](#model-choice), for independent
confirmation this is the right shape of solution).

Three eviction policies compared, same prompts, same code:

| policy | what it does | peak resident memory (OLMoE) | peak resident memory (Qwen1.5-MoE) |
|---|---|---|---|
| `no_eviction` | accumulate forever (Round 4's behavior) | 66.9% of full model | 78.8% of full model |
| `lru` | plain least-recently-used, fixed budget, **zero prediction** | 33.2% | 44.0% |
| `eai_predict` | evict to exactly what the EAI index predicts for the current token, prefetch that | **29.9%** | **40.5%** |

Correctness, checked the same rigorous way as Round 4 (fresh same-process
reference, at scale - 20 prompts, 960-1700+ token/layer pairs each run): 0
real mismatches on either model, every time. The KV-cached, one-token-at-
a-time code path produces bit-for-bit the same router decisions as a single
batched forward pass - streaming doesn't trade correctness for memory.

**`eai_predict` beats plain LRU on peak memory on both models** - real
evidence the fingerprint-based prediction is adding value over a
system with no learned index at all, not just matching what recency alone
would give you. But it comes with a real cost the peak-memory number hides:
total I/O volume. At zero lookahead, `eai_predict` moved **765GB** of expert
weights off disk across 20 OLMoE prompts, versus LRU's 294GB and
no-eviction's 172GB - aggressive per-token eviction thrashes, repeatedly
reloading experts that a slightly larger window would have kept resident. A
small lookahead (keep the next 2 tokens' predictions resident too, not just
the current one) cut that to 468GB (-39%) at the cost of nearly doubling
peak memory (53.6% of full model, up from 29.9%). This is a genuine,
measured dial, not a free lunch: less memory means more I/O, and where you
sit on that curve should depend on whether the deployment is I/O-bound (fast
NVMe, prioritize low memory, small/no lookahead) or memory-bound (plenty of
RAM, prioritize low I/O, larger lookahead) - exactly the kind of tuning knob
a real "frenetic I/O" streaming system needs, now backed by real numbers
instead of a guess.

### Round 6: a 30B model, actually fitting and running fast

Everything above was validated on models that fit comfortably in RAM even
fully loaded (13.8GB, 28.6GB) - useful for proving the mechanism cheaply and
safely, but not the regime where an external index actually matters. This
round targets that regime directly:
`Qwen/Qwen3-30B-A3B-Instruct-2507` - 48 layers, 128 experts/layer, top-8,
~58GB of expert weights alone in bf16 - the exact model that failed six
different ways in
[round 1's attempt](#attempting-a-30b-class-model-round-1-a-well-documented-failure),
never getting past the loading step at all.

**What made it work this time: never asking for the whole model.**
`eai/chunked_expert_loader.py` - the same module built for
[Round 4](#round-4-acting-on-the-prediction-not-just-scoring-it), extended
in this round to a third architecture (`load_chunked_qwen3moe`, reusing
`ChunkedExpertBlock` unchanged - Qwen3-MoE's router/FFN math is
bit-for-bit identical to OLMoE's, confirmed by reading both modeling files
side by side) - starts with **zero experts resident** and reads individual
experts off disk by name, on demand. Loading no longer means "materialize
~58GB of expert tensors plus overhead" (the operation that reliably needed
~41-44GB of peak RAM just to *attempt*, per round 1) - it means "materialize
a ~3GB backbone and nothing else yet":

```
LOADED_OK in 2.1s
RSS after load: 0.78 GB
input tokens: 15
FORWARD_OK in 22.5s
RSS after forward: 22.29 GB
next predicted token: ' A'
```

One forward pass, reactive-only (no prefetch at all - experts loaded purely
on demand as the real router asked for them), peaked at 22GB - comfortably
inside a 32GB consumer machine, nowhere near the ~41-44GB wall every
accelerate/bitsandbytes-based strategy hit without even finishing loading.
Watched with the same [RAM watchdog](scripts/ram_watchdog.py) as round 1,
this time it never came close to firing.

**A second, real failure mode surfaced and got fixed here too.** Collecting
router ground truth for this model (120 train + 40 test prompts, needed to
build a `.eai` index the same way as the other two models) crashed twice
more, *despite* correctly evicting our own cache between prompts - free RAM
dropped steadily across a growing number of prompts anyway. Root cause,
confirmed by direct measurement: `safetensors`' `get_tensor()` reads via
memory-mapped files, and while our own dict-based cache correctly drops its
*reference* to a tensor on eviction, the underlying mmap'd pages stay
resident in the process's working set for as long as the file handle itself
stays open - regardless of whether anything still references the tensor.
For OLMoE (13.8GB) and Qwen1.5-MoE (28GB) - both comfortably smaller than
free RAM - enough of the file could be mapped in without ever mattering. For
a ~58GB checkpoint, touching a large, *different* subset of experts across
many prompts eventually mapped in most of the file regardless of our own
cache being correctly bounded. The fix
(`ExpertShardIndex.recycle()`, called once per prompt in
[scripts/collect_chunked.py](scripts/collect_chunked.py)) closes and
reopens the shard file handles, forcing those pages to actually be
released - collection then ran cleanly to completion. This is also,
independently, *why*
[kimi-k3-in-c](https://github.com/josesilva05/kimi-k3-in-c) reads experts
with `O_DIRECT` instead of mmap: it sidesteps this exact page-cache
accumulation problem by design, for the same underlying reason we
discovered the hard way.

**Correctness: trusted by architecture-level proof, not re-verified
bit-exact on this specific model.** Doing the same fresh-reference-model
comparison used for OLMoE and Qwen1.5-MoE would mean loading a full 30B
reference model in the same process - exactly the memory risk this whole
approach exists to avoid, and self-defeating to attempt just to prove a
point already established. Instead: `ChunkedExpertBlock`'s router and FFN
math is unchanged from the version already verified bit-exact (float32,
zero mismatches) against OLMoE, and confirmed identical to Qwen3-MoE's own
`Qwen3MoeSparseMoeBlock`/`Qwen3MoeExperts`/`Qwen3MoeTopKRouter` by reading
both files side by side (see `_load_chunked_whole_block`'s docstring). The
sanity check available without that risk - a coherent, plausible next-token
prediction (`' A'`, a real, sensible continuation, not garbage) - passed.

**Prediction accuracy, same pipeline, same code, at this scale:**

| | Top-1 | Recall@8 | vs. baseline-2 (per-layer frequency, no prompt info) |
|---|---|---|---|
| granularity=prompt | 77.6% | 38.5% | (baseline: 72.7% / 33.6%) |
| granularity=token | **88.1%** | **59.3%** | (baseline: 72.7% / 33.6%) |

The token-vs-prompt pattern from [Round 2](#round-2-cluster-individual-tokens-instead-the-follow-up-experiment)
and [Round 3](#round-3-does-this-generalize-to-a-different-model-yes) holds a
third time, at nearly 3x the parameter count of the largest model tested
before - and by the widest margin yet: every one of 48 layers improved under
token granularity, not just the average (see the full per-layer table
`scripts/evaluate.py` prints).

**Real streaming memory, at this scale (20 test prompts; checked first at 8
prompts, then re-run at 20 for a more robust sample - both agree closely,
shown here):**

| policy | peak resident (measured) | % of full ~58GB expert weights |
|---|---|---|
| `no_eviction` | 20.6 GB mean / 22.9 GB max | 35.5% |
| `lru` (no prediction) | 9.5 GB mean / 9.9 GB max | 16.4% |
| `eai_predict` | 9.4 GB mean / 10.4 GB max | **16.1%** |

A ~58GB model, running with **~9-10GB peak resident expert memory** -
roughly a 6x reduction versus loading everything, and comfortably inside a
16GB consumer machine, let alone this project's 62GB one. Unlike the smaller
models, `eai_predict` and `lru` land nearly tied here (16.1% vs. 16.4%, a
real but small gap, holding steady between the 8- and 20-prompt runs) rather
than `eai_predict` clearly winning - plausibly because the `.eai` index for
this model was built from only 120 training prompts spread across 128
experts x 48 layers (many more cells than OLMoE's 64x16 or Qwen1.5-MoE's
60x24), leaving each cluster/layer cell with fewer training observations to
estimate from. A larger profiling set is the natural next thing to try
before concluding prediction stops helping at scale.

**What this round settles, concretely:** the "make a model that wouldn't fit,
fit, and run fast" goal this whole extension was aimed at is no longer
hypothetical. A model that failed to even *load* six different ways now
loads in ~2 seconds, runs a forward pass in ~22 seconds reactively, and
streams with real eviction at ~9-10GB peak - all correct (architecture-level
proof, sane outputs), all measured (not estimated), on the exact model class
the market-relevant papers in
[Related work](#related-work-is-this-valuable-to-the-market) target.

### Round 7: does EAI actually beat simple caching, under a fixed memory budget?

Every round above measured prediction accuracy or peak memory. None of them
asked the sharper question a real deployment cares about: under the SAME
fixed memory budget, does spending cycles on prediction beat just caching
well? `scripts/benchmark_streaming.py` (a new two-stage trace-then-replay
harness) answers this directly - `GlobalExpertCache`, one shared byte budget
across every layer, 8 pluggable eviction policies (`reactive`/`lru`/`lfu`/
`eai`/`eai_lookahead_N`/`eai_coactivation`/`oracle`), a per-step
router-correctness gate (0 mismatches across every run below), 4 workload
orderings, warm/cold cache scenarios. Full data and reasoning in
[docs/benchmark_findings_current.md](docs/benchmark_findings_current.md);
short version:

- **Plain `eai` is often net-negative.** At low-to-mid budgets on OLMoE,
  prefetching the predictor's raw `stored_top_k=16`-candidate window (2x the
  router's real `top_k=8`) evicted things that would have been reused
  naturally, for as little as ~9% prefetch precision - worse than `reactive`
  (no prediction at all) at 2GB (8.2% vs 23.3% hit_rate) and 4GB (32.3% vs
  39.0%). This is the first result in the whole project where "predict
  something" measurably lost to "predict nothing."
- **`eai_coactivation` mostly fixes it.** Instead of the raw candidate list,
  it greedily builds a `top_k`-sized clique from the persisted (but
  previously unused, since Round 2) coactivation matrix - experts that fire
  *together*, not just individually-frequent ones. Beat plain `eai` at every
  budget tested, competitive with `lru`/`lfu` in cold-cache conditions.
  Addresses [Round 6](#round-6-a-30b-model-actually-fitting-and-running-fast)'s
  open item #3 directly.
- **The whole "does prediction help" question is scoped to under-provisioned
  budgets.** Oracle's hit_rate lead over plain `reactive` caching shrinks
  from 30.7pp at 2GB to 4.8pp at 10GB (OLMoE, ~13.8GB total) as the budget
  approaches the full model - once the budget comfortably fits the model,
  every policy converges and prediction adds nothing but predictor-lookup
  overhead.
- **The single most counter-intuitive finding: in a warm, multi-prompt
  cache, plain `lfu` can beat a perfect one-step Oracle.** At 8GB warm,
  `lfu` hit 77.4% vs. Oracle's 76.3% - confirmed with real counters, not
  just hit_rate: Oracle evicted 2.4x more than `lfu` on an identical prompt
  sequence, because its prefetching only ever knows the very next step's
  need and keeps evicting things to make room for it, while `lfu` never
  evicts proactively and passively protects whatever stayed popular across
  the whole session. **Practical implication: in a long-running serving
  session (the realistic case, not a cold reset per request), a well-tuned
  LFU cache may already capture most of the achievable benefit**, and EAI's
  marginal value shrinks further than cold-cache numbers alone suggest.
- **Validated at Qwen3-30B scale too** (48 layers, 128 experts, the same
  model Round 6 got running): same qualitative shape, smaller absolute
  numbers (a 10GB budget covers a much smaller fraction of ~57GB of expert
  weight at this size) - reactive 19.1%/lru 20.8%/oracle 27.9% hit_rate at
  10GB.
- **Two real infrastructure bugs found and fixed by actually running this at
  scale, not by reasoning about it** - the kind of thing this project keeps
  finding every time it pushes to a bigger model. `ExpertShardIndex` read
  via mmap, which kept evicted experts' pages resident in the process's own
  working set regardless of the cache dropping its reference (fixed:
  plain `seek()`+`readinto()`, no mmap at all - see
  [docs/benchmark_findings_2026-09-19.md](docs/benchmark_findings_2026-09-19.md)
  §4). And the trace-collection cache from
  [Round 5](#round-5-real-streaming-eviction---load-process-unload-per-token)
  being unbounded *within* one prompt, fine at OLMoE's 16x64 expert grid,
  pulled 19GB from a single Qwen3-30B prefill alone before generating a
  token - fixed with bounded eviction plus chunked prefill (numerically
  verified equivalent to single-shot: float32 max logit diff ~1e-5, pure
  floating-point non-associativity, same class of noise already documented
  in this README, not a new bug).
- **First GPU (CUDA) measurement**: `device="cuda"` now works end to end
  (`eai/chunked_expert_loader.py`, `eai/expert_cache.py`). On this machine's
  8.5GB laptop GPU, disk-read cost was identical whether the destination was
  RAM or VRAM (2.237ms vs 2.225ms per ~4.2MB tensor) - PCIe bandwidth was
  never the bottleneck, disk always was - and GPU tok/s was actually
  *slightly worse* than CPU (88%) for this small a model at batch=1, not
  enough parallel work to amortize CUDA kernel overhead. Not yet wired into
  `benchmark_streaming.py`'s own sweep.
- **Does the MoE chunked-loading idea generalize to a dense model, database-
  partitioning style?** Tested directly (`scripts/dense_layer_streaming_experiment.py`):
  no. A dense model has no sparsity to exploit - every layer runs on every
  token unconditionally, so there's nothing to *predict*, only load/evict
  mechanics with no EAI-style upside. Measured net negative on both memory
  and speed on GPT-2 and Qwen2.5-1.5B (streaming used *more* memory than a
  fully-resident baseline, 14-38% of baseline tok/s) - though this predates
  the mmap fix above and hasn't been re-measured since.

## Related work: is this valuable to the market?

Checked before investing further, since "prove it's worth continuing" was an
explicit goal of this round. Two things are true at once:

**The problem is real and actively researched, not niche.** For large MoE
models, expert loading dominates inference latency once experts don't fit in
VRAM - memory transfer accounts for 84-88% of time per output token on models
like Qwen3-30B-A3B. At least a dozen papers from the last ~18 months attack
exactly this: [ADEPT](https://www.researchgate.net/publication/400888358_Two-Stage_Expert_Offloading_for_Domain-Aware_MoE_Inference)
(domain-aware prefetching - close to this PoC's own approach), [MoE-Infinity](https://www.semanticscholar.org/paper/MoE-Infinity%3A-Activation-Aware-Expert-Offloading-Xue-Fu/5e612ce3f2fdde6906aba85f1e7ab5113add1212),
[FineMoE](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
(EuroSys 2026), [MoE-SpeQ](https://arxiv.org/html/2511.14102),
[ST-MoE](https://arxiv.org/abs/2606.15453),
[DuoServe-MoE](https://arxiv.org/html/2509.07379v2),
[MoE-Beyond](https://arxiv.org/abs/2508.17137).

**Nobody ships this as an installable tool yet.** llama.cpp (what Ollama runs
on) only offers `--n-cpu-moe`: a static "offload the first N layers' experts
to CPU" heuristic, tuned by trial and error, blind to the prompt. vLLM has an
[open, unimplemented RFC](https://github.com/vllm-project/vllm/issues/38256)
for something similar. That gap between "research proves it works" and "a
tool a person can install" is where product value would actually live.

**Where this PoC sits technically, honestly:** the strongest published
result, [MoE-Beyond](https://arxiv.org/abs/2508.17137), trains a small
transformer on 66 million expert-activation traces to get 97.5%
accuracy/86.6% F1, lifting cache hit rate from 17% to 72%.
[ST-MoE](https://arxiv.org/abs/2606.15453) reports 85% prediction accuracy.
This PoC's zero-training KMeans approach reached 94.1% top-1/67.1% recall@8
on OLMoE and 70.7%/67.9% on Qwen1.5-MoE, off of ~2,500 training tokens -
genuinely competitive with the heuristic-caching baselines those papers beat,
but there's real headroom above us, and the way papers close that gap is
exactly what this PoC's own spec said not to build yet: a trained neural
predictor.

**Realistic differentiation, not a state-of-the-art claim:** near-zero
training cost (KMeans on a laptop vs. 66M traces + GPU training), a portable
model-agnostic file format, and a focus on the local/personal-use niche these
datacenter-oriented papers don't target. "Most of the benefit at a fraction
of the cost, for people running models locally" is a real, narrower claim -
worth building toward, not worth overselling.

## Limitations & recommendations

- **Prompt-level topic barely matters; token-level local context matters a
  lot.** Round 1 (cluster whole prompts) only added ~1-2.5 points over a
  baseline with zero prompt information (80.7%/42.6%) - the natural reading
  at that point was "layer identity explains almost everything, prompt
  content explains almost nothing." Round 2 (cluster individual tokens by
  their own hidden state) overturned that: 94.1% top-1 / 65-68% recall@8,
  every layer improved, not just the average. The correct conclusion wasn't
  "routing is unpredictable beyond layer popularity" - it was "prompt-level
  topic is the wrong granularity to look for the signal at." That's worth
  internalizing beyond this PoC: when an aggregate fingerprint barely beats a
  dumb baseline, the fix isn't necessarily a better embedding of the same
  unit - it might be the unit itself.
- **Semantic embedding and a free text hash scored the same - at prompt
  granularity.** `hidden_state` (2048d, requires running part of the model)
  and `hashing` (256d, zero model computation) landed within 0.5-1 point of
  each other in round 1. That comparison was not repeated at token
  granularity (hashing is defined on prompt text, not on a single token in
  isolation) - doing so honestly would need a token-level lexical baseline
  (e.g. hash the token string itself, or a small local window of tokens),
  not the whole-prompt hasher already implemented. Left for the next
  iteration.
- **Dataset size and cluster balance.** 120 train / 40 test prompts is enough
  to sanity-check the hypothesis, not to publish a claim. Token granularity
  helps here too, mechanically: clustering 2583 training tokens instead of
  120 prompts gives much larger, more balanced clusters at the same K (e.g.
  K=64: min 6/max 140/mean 40 tokens per cluster, vs. K=16 on prompts:
  min 1/max 22/mean 7.5) - part of why round 2 generalizes better, not only
  the change in what's being clustered.
- **Coactivation is computed but not yet used** by the predictor or metrics -
  it's persisted for a future prefetch/cache policy to consume (e.g. "if
  expert 12 is predicted, also warm expert 47 - they fire together 80% of the
  time"). It's currently only computed at whatever granularity you build the
  index with; not re-validated at token granularity.
- **A token-level hidden state still costs a partial forward pass to obtain**
  - cheaper than the full model, not free. This PoC does not measure that
  cost separately (see `forward_pass_seconds` caveat below); a real
  prefetch/cache system would need to know how many layers deep the
  fingerprint layer sits relative to how many layers of lookahead it buys.
- **CPU/GPU split adds latency noise.** `device_map="auto"` splits both models
  across GPU and CPU on this 8GB card, so `forward_pass_seconds` in the traces
  includes cross-device transfer overhead unrelated to the model's real
  compute cost - don't read those numbers as "how fast this model runs on a
  proper GPU."
- **Cross-model generalization holds qualitatively, not quantitatively.**
  [Round 3](#results) reproduced the exact pattern (token >> prompt >>
  baseline) on Qwen1.5-MoE, a different family and routing shape - real
  evidence this isn't an OLMoE-specific artifact. But the absolute numbers
  differ (94.1% vs 70.7% top-1), and this PoC didn't isolate why: lower
  expert density (top-4/60 vs top-8/64) and Qwen1.5-MoE's always-on shared
  expert (not modeled by `profile.py`/`predictor.py` at all) are both
  plausible, undistinguished causes. Only two models were tested - see
  [Model choice](#model-choice) for which other architectures were considered
  and ruled out (broken instrumentation, unverified custom code, or too
  large for this machine), not by design choice.
- **Only two market-relevant scales tested (7B, 14B total params).** The
  papers in [Related work](#related-work-is-this-valuable-to-the-market)
  target the regime where offloading actually hurts (30B+ total params, where
  memory transfer dominates latency). A serious, six-strategy attempt to test
  there (see
  [Attempting a 30B-class model](#attempting-a-30b-class-model-round-1-a-well-documented-failure))
  hit a hard, well-diagnosed ~41-44GB peak-RAM wall during model *loading
  alone*, independent of OS. The generalization evidence here is real but
  doesn't yet reach the scale where the market problem is most painful - not
  for lack of trying, but because loading a checkpoint that size needs
  roughly 3x its quantized footprint in free RAM just to get it resident,
  which this machine doesn't have. That number is itself a useful data point
  for anyone else attempting this on similarly modest hardware.
- Next steps this PoC deliberately did not attempt (per the spec's scope):
  predictive prefetch, an expert cache, SSD/RAM/VRAM offloading, a trained
  neural predictor, HNSW, distributed inference. All of those should follow
  *after* the hypothesis section below, not before.

## Does the hypothesis hold?

**Yes - once the index is keyed on the right unit.** The first attempt (cluster
whole prompts) said "barely." The follow-up (cluster individual tokens) said
"clearly." Both used the same model, traces, and evaluation code - only the
granularity changed.

The hypothesis was "characteristics correlate with expert activation enough
for an external index to predict a useful part of the execution path."
Tracing how the answer changed between the two rounds is itself the finding:

- **Round 1 (prompt granularity) looked like a weak result.** Recall@8 went
  from ~12.5% (random) to 43.6% - not nothing, but only ~1 point above a
  baseline that uses zero information about the prompt (42.6%, just "what's
  popular at this layer"). Swapping a real embedding for a free text hash
  changed almost nothing either. Taken alone, that pattern would say
  "external prediction barely works here, don't build on it."
- **Round 2 (token granularity) says that reading was about the wrong unit,
  not a dead end.** Clustering individual tokens by their own hidden state -
  same fingerprint layer, same model, same evaluation - takes recall@8 to
  65-68% and top-1 to ~94%, beating the zero-information baseline by 20-25
  points instead of 1. Every layer improved, not just a lucky average.
- **Why that makes sense in hindsight:** the router decides per token, and a
  causal model's token-level hidden state is already a contextualized summary
  of everything up to that point - closer to what the router actually
  conditions on than a whole-prompt average that blends a code snippet, a
  question, and a closing sentence into one vector. Round 1's fingerprint was
  diluting a real signal by averaging it away; round 2 stopped doing that.
- **Round 3 (a second, unrelated model) says this isn't an OLMoE fluke.**
  Qwen1.5-MoE-A2.7B - different family, different routing shape (60
  experts/top-4 vs. 64/top-8) - reproduced the exact same qualitative
  pattern: prompt granularity barely beat its baseline (25.4% vs 24.1%
  top-1), token granularity blew past it (70.7% vs 24.1% top-1, +38.9 points
  recall@8). The mechanism generalizes across architectures, even though the
  absolute numbers don't transfer 1:1 (see
  [Limitations](#limitations--recommendations) for why that gap is still
  unexplained).
- **The market check says the problem is real but under-served, and this PoC
  is in reasonable company, not ahead of it.** A dozen+ recent papers attack
  the same problem for datacenter-scale MoE; nothing ships as an installable
  tool yet. This PoC's zero-training approach is competitive with published
  heuristic-caching baselines, not with the best trained predictors - see
  [Related work](#related-work-is-this-valuable-to-the-market).
- **Round 4 (acting on the prediction, not just scoring it) proved the
  mechanism works, and found the next real bottleneck.** Loading OLMoE with
  zero experts resident and prefetching from the EAI index produced a model
  that's bit-exact with a fresh reference at float32, and correctly explains
  its residual bf16 differences as expected rounding noise, not bugs. But
  per-prompt resident memory only dropped ~11% versus loading everything -
  because even a *perfect oracle* needs 68% of a layer's experts just for one
  ~20-token prompt, token-to-token diversity *within* the prompt, not a
  prediction-quality problem. The fix is the same lesson as
  [Round 1 vs. 2](#round-2-cluster-individual-tokens-instead-the-follow-up-experiment)
  one level up: prefetch per-token with eviction (matching how ADEPT and
  similar papers actually do it), not once per whole prompt.
- **Round 5 (real streaming eviction) built that fix and it worked.**
  Token-by-token replay with a real KV cache, evicting down to the current
  token's prediction between steps, took peak resident memory to 29.9% of
  the full model on OLMoE and 40.5% on Qwen1.5-MoE - beating even a plain
  LRU cache with no prediction at all (33.2% / 44.0%), at 0 real
  correctness mismatches on either model. The cost this uncovered: I/O
  volume, not memory - aggressive per-token eviction thrashes (765GB moved
  for 20 OLMoE prompts at zero lookahead), and a small lookahead window
  trades some of that memory win back for less thrashing. A real dial, not
  a free lunch.
- **Round 6 (a genuine 30B-class model) is where this stopped being a
  toy-scale result.** The exact model round 1 failed to even load, six
  different ways, now loads in ~2s, runs a forward pass in ~22s, and streams
  with real eviction at ~9-10GB peak resident memory - **16.1% of its ~58GB
  expert weights**, roughly a 6x reduction, comfortably inside a 16GB
  consumer machine. Prediction accuracy held too: 88.1% top-1 / 59.3%
  recall@8 (token granularity) vs. 72.7% / 33.6% baseline - the widest
  margin of any model tested, on all 48 layers. A second real infrastructure
  bug surfaced and got fixed along the way (mmap page accumulation during
  trace collection, independent confirmation of why
  [kimi-k3-in-c](https://github.com/josesilva05/kimi-k3-in-c) uses
  `O_DIRECT`) - the kind of finding that only shows up by actually running
  at this scale, not by reasoning about it.
- **Round 7 (fixed-budget benchmark against simple caching) is where the
  hypothesis stopped being "does prediction correlate with routing" and
  became "does spending cycles on prediction actually pay for itself."**
  The honest answer: sometimes, and less than hoped. Plain `eai` lost to
  `reactive` (zero prediction) at low budgets; `eai_coactivation` (using the
  coactivation tensor Round 6 flagged as unused) mostly closed that gap; the
  oracle-vs-reactive advantage itself shrinks toward zero as budget
  approaches model size, by construction, not a bug; and in a warm,
  multi-prompt session, plain `lfu` beat a perfect Oracle outright, because
  Oracle's own prefetching disturbs cross-prompt residency a frequency
  count naturally protects. This is the project's first result where "add
  prediction" was measurably the wrong call in part of the space it was
  tested on - reported as-is, not around. See
  [Round 7](#round-7-does-eai-actually-beat-simple-caching-under-a-fixed-memory-budget)
  and [docs/benchmark_findings_current.md](docs/benchmark_findings_current.md)
  for the full data.

**Recommendation: the three things this PoC most needed to prove - "does
streaming eviction actually save memory," "does this work on a model large
enough for the market pain to be real," and "does prediction actually beat
simple caching under a fair, fixed budget" - are now all proven or honestly
disproven, not just planned.** What's left is refinement and scoping, not a
missing foundational piece:
1. ~~Tune the memory/I-O dial using the persisted `coactivation` tensor~~ -
   **done in Round 7** (`eai_coactivation`); it beats plain `eai` but the
   real result was narrower than hoped - see Round 7's bullets above for
   where it does and doesn't help, especially the warm-cache/`lfu` finding.
2. **Build an `eai_coactivation` + long-run-frequency hybrid.** Round 7's
   warm-scenario result (plain `lfu` beating a perfect Oracle) suggests the
   next real gain isn't a smarter *prediction*, it's giving the eviction
   policy the same kind of cross-prompt memory `lfu` gets almost for free -
   worth testing before concluding prediction has hit its ceiling.
3. **Explain why `eai_predict` and `lru` nearly tied at 30B scale** (16.1%
   vs. 16.3%, Round 6) when `eai_predict` clearly won at smaller scale
   (29.9% vs. 33.2% on OLMoE). The likely cause - 120 training prompts
   spread across 128 experts x 48 layers leaves each cluster/layer cell with
   less data than OLMoE's 64x16 grid got - is stated but not yet tested; a
   larger profiling set is the natural next experiment.
4. **Explain the OLMoE-vs-Qwen1.5-MoE accuracy gap** (density? shared
   expert?) before treating any single model's numbers as
   architecture-independent.
5. **More Qwen3-30B sweep coverage** - Round 7's target-scale numbers used
   only 2 prompts and cold-cache only; not enough to trust
   `eai_coactivation`'s exact ranking there, or to know if the warm-cache
   `lfu` finding holds at this scale too.
6. **Push past 30B** - the chunked-loader approach never needs the full
   model resident even at load time, so the ~41-44GB wall that stopped
   round 1's accelerate/bitsandbytes attempts may not reappear at 70B+
   either; worth finding out directly rather than assuming.
7. **Only then prototype a real cache**: predict → check hit-rate in a
   shadow run → promote to actually prefetching in a real serving loop, in
   that order, still without touching the router's real decisions until the
   shadow numbers justify it. Everything before this point has deliberately
   stayed a measurement harness, not a serving system - crossing that line
   is a genuinely new scope, not a natural continuation of what's already
   built.

## Studio / Ollama / API (out of scope for this PoC)

The broader idea of a visual, lightweight desktop "Studio" - listing/pulling
models via a local Ollama instance and exposing a local OpenAI-compatible API,
the way LM Studio does - is intentionally **not** part of this deliverable.
It's a UI/server project layered on top of this research core, not a research
question; building it now would have diluted the one thing this PoC needs to
prove first. Recommended shape when that work starts: a Python FastAPI backend
(imports `eai` as a library, proxies Ollama's local REST API, exposes an
OpenAI-compatible endpoint) behind a `pywebview` native window - one process,
one language, no Node/Rust toolchain, packageable to a single `.exe` with
PyInstaller.
