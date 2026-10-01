# RouteMoE

Two projects, one history each, combined here as siblings:

- [`eai-poc/`](eai-poc) — the Expert Activation Index research: does a compact
  external index predict which experts a MoE model's router will activate,
  and does acting on that prediction let a model run in far less memory than
  its full expert set would need? Seven rounds of benchmarking, all findings
  (positive and negative) preserved.
- [`eai-studio/`](eai-studio) — the product built on top of what that research
  validated: load/serve/unload MoE models through an OpenAI-compatible API
  and a small web GUI, Ollama-aware, LM-Studio-style.

See each directory's own README for details. `eai-studio`'s engine
(`eai_studio/engine/`) is copied byte-for-byte from `eai-poc/eai/` — the
research code and the product code are kept in sync manually, not shared via
a package dependency, since `eai-poc` stays a standalone research artifact.
