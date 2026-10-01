"""FastAPI app: an OpenAI-compatible inference endpoint plus eai-studio's own
model-management endpoints (load/unload/status), both backed by the same
ModelManager instance - so a model loaded via the management API is
immediately servable at the standard /v1/chat/completions path any
OpenAI-client-compatible tool (an IDE plugin, a CLI, a chat UI) already
knows how to call.

    python -m eai_studio.api.server              # serves on :8000
    uvicorn eai_studio.api.server:app --reload    # dev mode
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..engine.model_manager import ModelConfig, ModelManager
from ..ollama.client import OllamaClient
from ..ollama.moe_filter import KNOWN_MOE_MODELS, is_known_moe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eai_studio.api")

app = FastAPI(title="eai-studio", description="Load, serve, and unload large MoE models on modest hardware")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

manager = ModelManager()
ollama = OllamaClient()

_STATIC_DIR = Path(__file__).resolve().parent.parent / "gui" / "static"


# ---------------------------------------------------------------------------
# Management API - eai-studio's own control surface (not OpenAI-shaped)
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    model_id: str
    cache_gb: float = 4.0
    policy: Literal["reactive", "lru", "lfu", "hybrid"] = "hybrid"
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"


@app.post("/api/models/load")
def load_model(req: LoadRequest):
    """Load a model into memory with a FIXED expert-cache budget - this is
    the "fit a model that wouldn't otherwise fit" step. See
    docs/ADDING_A_MODEL.md if the model isn't downloaded locally yet."""
    try:
        return manager.load(ModelConfig(
            model_id=req.model_id, cache_gb=req.cache_gb, policy=req.policy,
            device=req.device, dtype=req.dtype,
        ))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("load failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{model_id:path}/unload")
def unload_model(model_id: str):
    """Free a loaded model's memory - RAM/VRAM back to the OS/driver, not
    just dropped from eai-studio's own bookkeeping."""
    try:
        return manager.unload(model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/models/loaded")
def list_loaded():
    """Everything currently resident - cache hit rate, resident bytes,
    uptime, per the metrics schema eai-poc's benchmark work standardized."""
    return {"models": manager.list_loaded()}


@app.get("/api/ollama/models")
def list_ollama_models(moe_only: bool = False):
    """List models Ollama already has pulled locally, optionally filtered to
    known MoE architectures only. Returns an empty list with an
    `ollama_available: false` flag (not an error) if Ollama isn't running -
    this feature is additive, not a hard dependency."""
    if not ollama.is_available():
        return {"ollama_available": False, "models": []}
    models = ollama.list_models()
    if moe_only:
        models = [m for m in models if is_known_moe(m["name"])]
    return {"ollama_available": True, "models": models}


@app.get("/api/moe-registry")
def moe_registry():
    """The known-MoE allowlist eai-studio ships with - see
    eai_studio/ollama/moe_filter.py and docs/ADDING_A_MODEL.md for how to
    extend it with a new architecture."""
    return {"known_moe_models": KNOWN_MOE_MODELS}


# ---------------------------------------------------------------------------
# OpenAI-compatible API - what any existing OpenAI-client tool already speaks
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = False


@app.get("/v1/models")
def openai_list_models():
    """OpenAI-shaped /v1/models - only models eai-studio currently has
    loaded are "available" for completion, same semantics as LM Studio's
    own /v1/models (what's loaded, not everything on disk)."""
    return {
        "object": "list",
        "data": [
            {"id": m["model_id"], "object": "model", "owned_by": "eai-studio"}
            for m in manager.list_loaded()
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not yet implemented - see docs/EXPOSING_VIA_API.md's roadmap")
    if not manager.is_loaded(req.model):
        raise HTTPException(
            status_code=404,
            detail=f"model {req.model!r} is not loaded - POST /api/models/load first (OpenAI's API has no load/unload concept, so this extra step is eai-studio-specific; see docs/EXPOSING_VIA_API.md).",
        )
    try:
        result = manager.generate(
            req.model, [m.model_dump() for m in req.messages],
            max_new_tokens=req.max_tokens, temperature=req.temperature,
        )
    except Exception as e:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "completion_tokens": result["output_tokens"],
            "eai_studio_tokens_per_second": result["tokens_per_second"],
            "eai_studio_ttft_ms": result["ttft_ms"],
        },
    }


@app.get("/")
def root():
    return {"name": "eai-studio", "docs": "/docs", "ui": "/ui", "status": "ok"}


# Mounted last: StaticFiles(html=True) would otherwise shadow the API routes
# above at the same prefix if mounted at "/" - keeping the GUI under "/ui"
# avoids that entirely while still letting "/" stay a plain JSON health check.
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


def main():
    import os

    import uvicorn

    # 8877, not 8000/8080/etc - this machine already has other local
    # services bound to the more common ports; --port overrides either way.
    port = int(os.environ.get("EAI_STUDIO_PORT", "8877"))
    uvicorn.run("eai_studio.api.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
