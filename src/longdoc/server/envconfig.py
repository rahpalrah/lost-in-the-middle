"""Build a configured service from environment variables (no web framework involved).

Environment variables:

``LONGDOC_PROVIDER``
    ``auto`` (default: Claude if ``ANTHROPIC_API_KEY`` is set, otherwise the built-in extractive
    engine, which needs no key or model), ``anthropic``, ``openai``, ``extractive`` or ``local``.
``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
    Provider credentials. Never put them in code; see ``docs/OPENAI.md``.
``LONGDOC_MODEL``
    Model name. Required for ``openai``. Defaults to a Claude model for ``anthropic``.
``LONGDOC_OPENAI_BASE_URL``
    An OpenAI-compatible server instead of api.openai.com.
``LONGDOC_LOCAL_MODEL`` / ``LONGDOC_LOCAL_CTX``
    Path to a GGUF model and its context window (selects the ``local`` provider; needs the
    ``local`` extra).
``LONGDOC_API_KEY``
    Bearer token that callers of this API must send. Strongly recommended.
``LONGDOC_STORE_DIR``
    Directory for persisted documents. Required for more than one worker.
``LONGDOC_EFFORT``, ``LONGDOC_STRATEGY``, ``LONGDOC_CACHE_TTL``, ``LONGDOC_FALLBACKS``
    Override :class:`longdoc.QAConfig` defaults (``LONGDOC_FALLBACKS=none`` disables them).
``LONGDOC_MAX_UPLOAD_MB``
    Upload size limit in megabytes (default 25).
``LONGDOC_RATE_LIMIT_PER_MIN``
    Maximum uploads plus questions per client address per minute (default: unlimited).
``LONGDOC_MAX_CONCURRENCY``
    Requests handled at once by the built-in server before it answers 503 (default 32).
``LONGDOC_TRUST_PROXY``
    ``1`` to use ``X-Forwarded-For`` as the client address (only behind a trusted proxy).
``LONGDOC_ACCOUNTS`` / ``LONGDOC_ACCOUNTS_FILE``
    Turn on self-service keys (``/register``); the file persists accounts. Related:
    ``LONGDOC_PUBLIC_URL`` (used in emailed links), ``LONGDOC_ADMIN_TOKEN`` (for ``/admin``),
    ``LONGDOC_SMTP_HOST/PORT/USER/PASSWORD/FROM`` (verification and owner email),
    ``LONGDOC_OWNER_EMAIL``, ``LONGDOC_EVENTS_FILE``, ``LONGDOC_GITHUB_REPO`` +
    ``LONGDOC_GITHUB_TOKEN`` (owner alerts as issues in a **private** repo),
    ``LONGDOC_ACCOUNTS_DEV`` (return the verify link in the response; local testing only).
``LONGDOC_ENV_FILE``
    Path to a ``.env`` file. Real environment variables take precedence over the file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..accounts import AccountService
from ..api import DEFAULT_MAX_UPLOAD_BYTES
from ..config import QAConfig, Strategy
from ..env import read_env_file
from ..exceptions import ConfigurationError
from ..providers import make_client, resolve_provider
from ..service import DocumentService
from ..store import DocumentStore, InMemoryStore, JSONFileStore


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything needed to run the API."""

    service: DocumentService
    api_key: str | None
    max_upload_bytes: int
    rate_limit_per_minute: int | None
    max_concurrency: int
    trust_proxy: bool
    provider: str
    accounts: AccountService | None = None


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


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    """Read the environment (and ``LONGDOC_ENV_FILE``) and build the service and its settings."""
    env = os.environ if env is None else env
    if env_file := env.get("LONGDOC_ENV_FILE"):
        env = {**read_env_file(env_file), **dict(env)}  # real environment variables win

    store_dir = env.get("LONGDOC_STORE_DIR")
    store: DocumentStore = JSONFileStore(store_dir) if store_dir else InMemoryStore()
    config = config_from_env(env)

    requested = env.get("LONGDOC_PROVIDER", "auto")
    if env.get("LONGDOC_LOCAL_MODEL") and requested.lower() == "auto":
        requested = "local"
    provider = resolve_provider(requested, env)

    if provider == "openai":
        if not env.get("LONGDOC_MODEL"):
            raise ConfigurationError("LONGDOC_PROVIDER=openai requires LONGDOC_MODEL.")
        client = make_client(
            "openai",
            api_key=env.get("OPENAI_API_KEY") or None,
            base_url=env.get("LONGDOC_OPENAI_BASE_URL") or None,
        )
        config = replace(config, fallbacks=None)
    elif provider == "local":
        path = env.get("LONGDOC_LOCAL_MODEL")
        if not path:
            raise ConfigurationError("LONGDOC_PROVIDER=local requires LONGDOC_LOCAL_MODEL.")
        client = make_client(
            "local", model_path=path, n_ctx=int(env.get("LONGDOC_LOCAL_CTX", "16384"))
        )
        config = replace(config, model=client.model_name, fallbacks=None)
    elif provider == "anthropic":
        client = make_client("anthropic", api_key=env.get("ANTHROPIC_API_KEY") or None)
    else:
        client = make_client("extractive")
        config = replace(config, fallbacks=None)

    rate = env.get("LONGDOC_RATE_LIMIT_PER_MIN")
    return Settings(
        service=DocumentService(store=store, config=config, client=client),
        api_key=env.get("LONGDOC_API_KEY") or None,
        max_upload_bytes=int(float(env.get("LONGDOC_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
        if env.get("LONGDOC_MAX_UPLOAD_MB")
        else DEFAULT_MAX_UPLOAD_BYTES,
        rate_limit_per_minute=int(rate) if rate else None,
        max_concurrency=int(env.get("LONGDOC_MAX_CONCURRENCY", "32")),
        trust_proxy=env.get("LONGDOC_TRUST_PROXY", "") in ("1", "true", "yes"),
        provider=provider,
        accounts=AccountService.from_env(env),
    )
