"""ASGI entry point configured from environment variables.

Run with any ASGI server, for example::

    uvicorn longdoc.server.asgi:app --host 0.0.0.0 --port 8000 --workers 4

Configuration comes from environment variables; see :mod:`longdoc.server.envconfig`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from fastapi import FastAPI

from .app import create_app
from .envconfig import config_from_env, settings_from_env

__all__ = ["app", "app_from_env", "config_from_env"]


def app_from_env(env: Mapping[str, str] | None = None) -> FastAPI:
    """Build the FastAPI application from environment variables."""
    settings = settings_from_env(os.environ if env is None else env)
    return create_app(
        settings.service,
        api_key=settings.api_key,
        max_upload_bytes=settings.max_upload_bytes,
        rate_limit_per_minute=settings.rate_limit_per_minute,
    )


app = app_from_env()
