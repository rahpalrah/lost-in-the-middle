"""Async support for any client, without a second HTTP stack.

:class:`AsyncAdapter` presents the async interface longdoc expects
(``await client.beta.messages.stream(...)``) on top of a *synchronous* client, by running the
blocking call in a worker thread. It works with every client in this package and also with
the official Anthropic SDK's sync client, so there is only one implementation of each provider.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from .local import _AsyncStream


def _final_message(sync_client: Any, kwargs: dict[str, Any]) -> Any:
    with sync_client.beta.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


class _AsyncMessages:
    def __init__(self, sync_client: Any) -> None:
        self._sync = sync_client

    async def create(self, **kwargs: Any) -> Any:
        """Run ``messages.create`` in a worker thread."""
        return await asyncio.to_thread(self._sync.beta.messages.create, **kwargs)

    def stream(self, **kwargs: Any) -> _AsyncStream:
        """Async-context-manager form: ``async with ... as s: await s.get_final_message()``."""
        return _AsyncStream(asyncio.to_thread(_final_message, self._sync, kwargs))


class AsyncAdapter:
    """Async view of a synchronous client (see the module docstring)."""

    def __init__(self, sync_client: Any) -> None:
        self.sync_client = sync_client
        self.beta = SimpleNamespace(messages=_AsyncMessages(sync_client))
        self.messages = self.beta.messages

    def __repr__(self) -> str:
        return f"AsyncAdapter({self.sync_client!r})"
