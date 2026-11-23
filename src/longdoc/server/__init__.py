"""HTTP serving for longdoc.

* **Standalone service, no dependencies** (any language can call it)::

      longdoc serve --store ./data --port 8000

  or ``from longdoc.server import create_server``. It uses only the standard library.

* **Mount into your existing FastAPI app** (optional; needs ``pip install "longdoc-qa[fastapi]"``)::

      from longdoc.server import create_router
      app.include_router(create_router(service), prefix="/longdoc")
"""

from __future__ import annotations

from typing import Any

from ..api import LongDocApi
from .envconfig import Settings, config_from_env, settings_from_env
from .stdlib import LongDocServer, create_server, serve

__all__ = [
    "LongDocApi",
    "LongDocServer",
    "Settings",
    "config_from_env",
    "create_app",
    "create_router",
    "create_server",
    "serve",
    "settings_from_env",
]


def __getattr__(name: str) -> Any:
    """Load the optional FastAPI integration only when it is asked for."""
    if name in ("create_app", "create_router"):
        try:
            from . import app
        except ImportError as exc:
            raise ImportError(
                'The FastAPI integration needs FastAPI: pip install "longdoc-qa[fastapi]"'
            ) from exc
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
