# eai-studio

Load, serve, and unload large Mixture-of-Experts (MoE) models on modest
hardware — the productized follow-on to [`eai-poc`](../eai-poc), which spent
seven rounds of benchmarking validating the underlying mechanism: a model
whose expert weights don't fit in RAM/VRAM can still run, by keeping only a
fixed-byte-budget cache of experts resident and streaming the rest from disk
on demand, through a chunked loader that never materializes the full expert
set at once.

This project takes exactly the parts of that research that held up under
correctness testing — the chunked loader, the fixed-budget cache with its
`hybrid`/`lru`/`lfu`/`reactive` eviction policies — and wraps them in
something you actually run day to day: an OpenAI-compatible API server, a
small web GUI, and Ollama-aware model discovery, with explicit load/unload
control over what's resident at any moment (the same mental model as LM
Studio's model picker, applied to models that wouldn't fit any other way).

## Why this exists, not just eai-poc

`eai-poc` answers "does this caching idea work, and under which policy?" with
benchmark scripts you run from a terminal and read a JSON report from. It
answers that question rigorously, but it isn't something you *use*.
`eai-studio` is the other half: given the policy that eai-poc validated,
expose it as a long-running service you can point a chat UI or an
OpenAI-client script at, load/unload models into, and extend with a new
model family without touching the cache internals at all.

Nothing about the caching/eviction logic changes here — `chunked_loader.py`
and `expert_cache.py` under `eai_studio/engine/` are carried over from
eai-poc byte-for-byte. All the new code is orchestration: turning "run one
benchmark and exit" into "load, generate, unload, repeat, across any number
of models, driven by HTTP requests."

## Quickstart

```bash
python -m pip install -e .
python -m eai_studio.api.server
```

Then open `http://127.0.0.1:8877/ui/` for the web GUI, or
`http://127.0.0.1:8877/docs` for interactive API docs. Loading a model
requires it to already be in your local Hugging Face cache — see
[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) for why (predictable load
latency: no surprise multi-GB download mid-request) and how to populate it.

```bash
curl -X POST http://127.0.0.1:8877/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "allenai/OLMoE-1B-7B-0924", "cache_gb": 4.0, "policy": "hybrid", "device": "cpu"}'

curl -X POST http://127.0.0.1:8877/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "allenai/OLMoE-1B-7B-0924", "messages": [{"role": "user", "content": "Hello!"}]}'

curl -X POST http://127.0.0.1:8877/api/models/allenai%2FOLMoE-1B-7B-0924/unload
```

## Layout

```
eai_studio/
  engine/
    chunked_loader.py   # copied from eai-poc - zero-experts-resident model construction
    expert_cache.py     # copied from eai-poc - fixed-budget cache, pluggable eviction policies
    model_manager.py    # NEW - load/generate/unload orchestration, one model at a time per lock
  api/
    server.py           # NEW - FastAPI: OpenAI-compatible + eai-studio management endpoints
  ollama/
    client.py           # NEW - talks to a local Ollama daemon, degrades gracefully if absent
    moe_filter.py        # NEW - known/likely-MoE model-name detection, used by the GUI's filter
  gui/static/index.html # NEW - single-page web GUI (load/unload/chat), no build step
docs/
  ADDING_A_MODEL.md     # how to get a new model (or architecture) working
  EXPOSING_VIA_API.md   # full endpoint reference + what's not implemented yet
tests/
  test_model_manager.py
  test_moe_filter.py
```

## Supported architectures today

`olmoe`, `qwen2_moe` (has a shared expert), `qwen3_moe` — see
[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) for the detection table and
the checklist for adding a new one. A dense model gains nothing from this
project (there's no sparsity to exploit) — don't load one here.

## What's deliberately not done yet

- **Ollama-hosted inference** — the Ollama integration today is model
  *discovery* (`GET /api/ollama/models`, surfaced in the GUI's filterable
  list) so you can see what Ollama already has pulled and MoE-filter it;
  actually routing a chat request *through* Ollama's own runtime, instead of
  through this project's chunked-loading engine, is a reasonable follow-up
  but a different code path, not yet built.
- **A native desktop window** — the GUI is a plain page served by the same
  FastAPI process (`/ui/`); a `pywebview` wrapper for a standalone window
  (and eventually a packaged `.exe`) is a thin follow-up, not a redesign —
  see the `gui` extra in `pyproject.toml`.
- **Streaming responses, concurrent requests to one model, auth** — see
  [docs/EXPOSING_VIA_API.md](docs/EXPOSING_VIA_API.md)'s "What's NOT
  implemented yet" section for the full list and why each is out of scope
  for now rather than forgotten.

## Relationship to eai-poc

`../eai-poc` keeps every benchmark script, finding, and negative result from
the research phase — nothing there is deleted or superseded. Consult
`eai-poc/docs/cache_policy_matrix.md` before changing this project's default
cache policy weights; they're not arbitrary, they're eai-poc's own
best-performing sweep combination so far.
