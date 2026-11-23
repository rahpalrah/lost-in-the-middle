"""HTTP API for longdoc (requires the ``server`` extra: ``pip install "longdoc-qa[server]"``).

Two ways to use it:

* **Mount into your existing FastAPI app**::

      from longdoc.server import create_router
      app.include_router(create_router(service), prefix="/longdoc")

* **Run as a standalone service** (any language can call it; OpenAPI docs at ``/docs``)::

      longdoc serve --store ./data --port 8000
"""

from __future__ import annotations

try:
    import fastapi  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'The HTTP server needs extra dependencies: pip install "longdoc-qa[server]"'
    ) from exc

from .app import create_app, create_router

__all__ = ["create_app", "create_router"]
