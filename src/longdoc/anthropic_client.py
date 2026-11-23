"""Talk to Claude directly over HTTPS, with no ``anthropic`` package.

.. code-block:: python

    from longdoc import DocumentQA
    from longdoc.anthropic_client import AnthropicClient

    qa = DocumentQA.from_file("contract.pdf", client=AnthropicClient())  # ANTHROPIC_API_KEY

The key comes from the ``ANTHROPIC_API_KEY`` environment variable (or ``api_key=``) and is never
logged or shown. If you already use the official ``anthropic`` SDK you can pass its client to
longdoc instead; both work.

Status: the request and response mapping is covered by tests with a fake transport. It has not
been run against the live Claude API (no Anthropic key was used in development).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from ._version import __version__  # noqa: F401  (kept for parity with other clients)
from .aio import AsyncAdapter
from .exceptions import ConfigurationError
from .http import Response, Secret, Transport, post_json
from .local import (
    LocalBlock,
    LocalMessage,
    LocalStopDetails,
    LocalTextBlock,
    LocalUsage,
    _Stream,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_VERSION = "2023-06-01"
_BODY_KEYS = ("model", "max_tokens", "messages", "system", "output_config", "fallbacks", "thinking")

_NO_KEY = (
    "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment variable, or pass "
    "api_key=... (see docs/INTEGRATION.md)."
)


def build_body(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the Messages API fields out of longdoc's request arguments."""
    body = {k: kwargs[k] for k in _BODY_KEYS if kwargs.get(k) is not None}
    if not body.get("model"):
        raise ConfigurationError("A model is required, for example QAConfig(model=...).")
    return body


def parse_response(response: Response) -> LocalMessage:
    """Convert a Messages API response into the message shape longdoc reads."""
    data = response.body
    blocks: list[Any] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            blocks.append(LocalTextBlock(str(block.get("text", ""))))
        else:
            blocks.append(LocalBlock(type=str(block.get("type", "unknown"))))

    usage = data.get("usage") or {}
    iterations = [SimpleNamespace(type=i.get("type")) for i in usage.get("iterations") or []]
    details = data.get("stop_details")
    return LocalMessage(
        content=blocks,
        model=str(data.get("model") or ""),
        stop_reason=str(data.get("stop_reason") or "end_turn"),
        usage=LocalUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            iterations=iterations or None,
        ),
        stop_details=(
            LocalStopDetails(details.get("category"), details.get("explanation"))
            if isinstance(details, dict)
            else None
        ),
        _request_id=response.request_id or data.get("request_id"),
    )


class _Messages:
    def __init__(self, owner: AnthropicClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> LocalMessage:
        """Send one request."""
        return self._owner.complete(**kwargs)

    def stream(self, **kwargs: Any) -> _Stream:
        """Same as :meth:`create`, wrapped like the SDK's streaming helper."""
        return _Stream(self._owner.complete(**kwargs))


class AnthropicClient:
    """A dependency-free client for the Claude Messages API.

    Args:
        api_key: Your key. Prefer leaving this unset and using ``ANTHROPIC_API_KEY``.
        base_url: Override the API address (default ``ANTHROPIC_BASE_URL`` or the public API).
        anthropic_version: The ``anthropic-version`` header value.
        timeout: Seconds to wait for a response.
        max_retries: Automatic retries for transient failures (rate limit, overload, network).
        allow_insecure: Permit plain ``http://`` to a non-local host (not recommended).
        transport: Replace the HTTP call, for tests: ``(url, headers, body) -> Response``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_version: str = DEFAULT_VERSION,
        timeout: float = 600.0,
        max_retries: int = 2,
        allow_insecure: bool = False,
        transport: Transport | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError(_NO_KEY)
        self._key = Secret(key)
        self.base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.anthropic_version = anthropic_version
        self._transport = transport or (
            lambda url, headers, body: post_json(
                url,
                headers,
                body,
                timeout=timeout,
                max_retries=max_retries,
                allow_insecure=allow_insecure,
            )
        )
        self.beta = SimpleNamespace(messages=_Messages(self))
        self.messages = self.beta.messages

    def __repr__(self) -> str:  # never include credentials
        return f"AnthropicClient(base_url={self.base_url!r})"

    def complete(self, *, betas: list[str] | None = None, **kwargs: Any) -> LocalMessage:
        """Call the Messages API. ``betas`` become the ``anthropic-beta`` header."""
        headers = {
            "x-api-key": self._key.reveal(),
            "anthropic-version": self.anthropic_version,
        }
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        response = self._transport(f"{self.base_url}/v1/messages", headers, build_body(kwargs))
        return parse_response(response)

    def to_async(self) -> AsyncAdapter:
        """An async view of this client."""
        return AsyncAdapter(self)
