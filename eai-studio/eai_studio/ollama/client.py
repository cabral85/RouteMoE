"""Talks to a LOCAL Ollama instance's own REST API (default
http://localhost:11434) - lists what's already pulled, checks whether
Ollama is even running. Deliberately thin and read-mostly: eai-studio
doesn't reimplement Ollama's model management, it reads from it so its own
GUI can show "models Ollama already has" without a separate download step
for anything the user already pulled.

Not a hard dependency: every method degrades to "unavailable"/empty rather
than raising, when Ollama isn't installed or isn't running - see
is_available().
"""

from __future__ import annotations

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[dict]:
        """Raw `ollama list`-equivalent - name, size, modified date, and
        Ollama's own reported parameter/quantization details where present.
        Returns [] (not an error) if Ollama isn't reachable."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        data = r.json()
        models = []
        for m in data.get("models", []):
            details = m.get("details", {})
            models.append({
                "name": m.get("name", ""),
                "size_bytes": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
                "parameter_size": details.get("parameter_size", ""),
                "quantization_level": details.get("quantization_level", ""),
                "family": details.get("family", ""),
            })
        return models

    def show(self, model_name: str) -> dict | None:
        """Ollama's /api/show - full modelfile/template/params for one
        model, useful for confirming architecture details (e.g. expert
        count) beyond what list_models()'s summary exposes."""
        try:
            r = httpx.post(f"{self.base_url}/api/show", json={"name": model_name}, timeout=self.timeout_seconds)
            r.raise_for_status()
        except httpx.HTTPError:
            return None
        return r.json()
