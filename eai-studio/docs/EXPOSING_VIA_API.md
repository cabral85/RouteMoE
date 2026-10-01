# Exposing a model via API

eai-studio runs one local FastAPI server (`eai_studio/api/server.py`) with
two kinds of endpoints: its own model-management surface (load/unload/list -
nothing like this exists in the OpenAI API, since OpenAI's own API never
needs to "load" anything) and an OpenAI-compatible inference surface (so any
existing tool that already speaks the OpenAI client protocol - an IDE
plugin, a chat UI, a script using the `openai` Python package pointed at a
custom `base_url` - works against eai-studio with no changes beyond the URL
and model name).

## Start the server

```bash
python -m eai_studio.api.server          # http://127.0.0.1:8877, see --port below
EAI_STUDIO_PORT=9000 python -m eai_studio.api.server   # custom port
```

Interactive API docs (Swagger UI) at `http://127.0.0.1:8877/docs` once
running - every endpoint below, with a "try it" form, comes from there for
free (FastAPI generates it from the route definitions).

## Management endpoints (eai-studio-specific)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/models/load` | Load a model with a given cache budget/policy/device - see `docs/ADDING_A_MODEL.md` |
| `POST` | `/api/models/{model_id}/unload` | Free a loaded model's RAM/VRAM |
| `GET` | `/api/models/loaded` | Every currently-loaded model + its live cache stats |
| `GET` | `/api/ollama/models?moe_only=true` | Models Ollama already has pulled, optionally MoE-filtered |
| `GET` | `/api/moe-registry` | The known-MoE allowlist this build ships with |

## OpenAI-compatible endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | Lists currently-*loaded* models (not everything on disk - matches LM Studio's own `/v1/models` semantics, not OpenAI's "every model your account can use") |
| `POST` | `/v1/chat/completions` | Standard chat completion request/response shape, plus `usage.eai_studio_tokens_per_second`/`eai_studio_ttft_ms` extras |

**A model must be loaded (`POST /api/models/load`) before
`/v1/chat/completions` will serve it** - this is the one real difference
from a hosted OpenAI-API-compatible service, and unavoidable: eai-studio's
whole purpose is controlling exactly when a model's weights are resident,
not having everything loaded all the time. A request for an unloaded model
returns a `404` with that explanation, not a confusing generic error.

### Example: point an existing OpenAI-client tool at eai-studio

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8877/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="allenai/OLMoE-1B-7B-0924",  # must already be loaded - see above
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## What's NOT implemented yet

- **Streaming (`stream=true`)** - `/v1/chat/completions` returns the
  complete response in one payload today; a request with `stream=true`
  gets a clear `400`, not a silent fallback to non-streaming. Real
  token-by-token SSE streaming is a natural next addition (the engine
  already generates one token at a time internally -
  `eai_studio/engine/model_manager.py::generate` - this is a thin
  response-shape change, not new generation logic).
- **Concurrent requests to the same model** - `generate()` holds a
  per-model lock; a second request while one is in flight blocks (queues)
  rather than running in parallel. Correct for a single local user, wrong
  for multi-tenant serving - see `model_manager.py`'s docstring for why
  that's a deliberate scope boundary, not an oversight.
- **Authentication** - this is a local, loopback-bound (`127.0.0.1`)
  server by default. Binding it to `0.0.0.0` or a public interface without
  adding auth in front of it would expose both inference and the
  load/unload controls to anyone who can reach the port - don't do that
  without adding a real auth layer first.
- **`/v1/completions` (legacy, non-chat)** and **`/v1/embeddings`** - only
  chat completions are wired up; add analogous routes in
  `eai_studio/api/server.py` if a tool needs the older completion shape.
