"""Choose which engine answers questions.

``make_client("auto")`` (the default everywhere) picks the first that is available:

1. **Claude**, if ``ANTHROPIC_API_KEY`` is set.
2. The built-in **extractive** engine, which needs nothing: no key, no model, no network.

OpenAI and local models are never chosen silently, because they need a model name or a model
file: select them explicitly with ``provider="openai"`` / ``provider="local"``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

PROVIDERS = ("auto", "anthropic", "openai", "extractive", "local")


def resolve_provider(provider: str = "auto", env: Mapping[str, str] | None = None) -> str:
    """Return the concrete provider name for ``provider`` (resolving ``"auto"``)."""
    env = os.environ if env is None else env
    provider = (provider or "auto").lower()
    if provider not in PROVIDERS:
        raise ConfigurationError(
            f"Unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}."
        )
    if provider == "auto":
        return "anthropic" if env.get("ANTHROPIC_API_KEY") else "extractive"
    return provider


def make_client(
    provider: str = "auto", *, env: Mapping[str, str] | None = None, **kwargs: Any
) -> Any:
    """Build a client for ``provider``. Extra keyword arguments go to its constructor."""
    name = resolve_provider(provider, env)
    if name == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(**kwargs)
    if name == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(**kwargs)
    if name == "local":
        from .local import LlamaCppClient

        if "model_path" not in kwargs:
            raise ConfigurationError("provider='local' needs model_path=... (a .gguf file).")
        return LlamaCppClient(kwargs.pop("model_path"), **kwargs)
    from .extractive import ExtractiveClient

    if provider == "auto":
        logger.info(
            "No ANTHROPIC_API_KEY found: using the built-in extractive engine, which quotes "
            "passages instead of writing answers. Pass a client or set a key to use a model."
        )
    return ExtractiveClient(**kwargs)
