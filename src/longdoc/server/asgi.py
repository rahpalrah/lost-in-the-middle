"""ASGI entry point configured from environment variables.

Run with any ASGI server, for example::

    uvicorn longdoc.server.asgi:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables:

``ANTHROPIC_API_KEY``
    Credential for Claude (read by the Anthropic SDK).
``LONGDOC_API_KEY``
    Bearer token that callers of this API must send. Strongly recommended.
``LONGDOC_STORE_DIR``
    Directory for persisted documents. Required for more than one worker, so that
    workers share documents. Defaults to in-memory storage.
``LONGDOC_MODEL``, ``LONGDOC_EFFORT``, ``LONGDOC_STRATEGY``, ``LONGDOC_CACHE_TTL``
    Override :class:`longdoc.QAConfig` defaults.
``LONGDOC_FALLBACKS``
    ``default`` (the default) or ``none``.
``LONGDOC_MAX_UPLOAD_MB``
    Upload size limit in megabytes (default 25).
``LONGDOC_LOCAL_MODEL``
    Path to a GGUF model. When set, answers come from this local model instead of
    Claude, and no Anthropic API key is needed (requires the ``local`` extra).
``LONGDOC_LOCAL_CTX``
    Local model context window in tokens (default 16384).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from fastapi import FastAPI

from ..config import QAConfig, Strategy
from ..service import DocumentService
from ..store import DocumentStore, InMemoryStore, JSONFileStore
from .app import create_app


def config_from_env(env: Mapping[str, str]) -> QAConfig:
    """Build a :class:`QAConfig` from ``LONGDOC_*`` variables."""
    kwargs: dict[str, Any] = {}
    if model := env.get("LONGDOC_MODEL"):
        kwargs["model"] = model
    if effort := env.get("LONGDOC_EFFORT"):
        kwargs["effort"] = effort
    if strategy := env.get("LONGDOC_STRATEGY"):
        kwargs["strategy"] = Strategy(strategy)
    if ttl := env.get("LONGDOC_CACHE_TTL"):
        kwargs["cache_ttl"] = ttl
    if (fallbacks := env.get("LONGDOC_FALLBACKS")) is not None:
        kwargs["fallbacks"] = None if fallbacks.lower() in ("none", "off", "") else fallbacks
    return QAConfig(**kwargs)


def app_from_env(env: Mapping[str, str] | None = None) -> FastAPI:
    """Build the application from environment variables."""
    env = os.environ if env is None else env
    store_dir = env.get("LONGDOC_STORE_DIR")
    store: DocumentStore = JSONFileStore(store_dir) if store_dir else InMemoryStore()
    config = config_from_env(env)
    client = None
    if local_model := env.get("LONGDOC_LOCAL_MODEL"):
        from ..local import LlamaCppClient

        client = LlamaCppClient(local_model, n_ctx=int(env.get("LONGDOC_LOCAL_CTX", "16384")))
        config = replace(config, model=client.model_name, fallbacks=None)
    service = DocumentService(store=store, config=config, client=client)
    max_mb = float(env.get("LONGDOC_MAX_UPLOAD_MB", "25"))
    return create_app(
        service,
        api_key=env.get("LONGDOC_API_KEY") or None,
        max_upload_bytes=int(max_mb * 1024 * 1024),
    )


app = app_from_env()
