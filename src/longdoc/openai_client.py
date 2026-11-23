"""Use OpenAI, or any OpenAI-compatible server, with no ``openai`` package.

.. code-block:: python

    from longdoc import DocumentQA, QAConfig
    from longdoc.openai_client import OpenAIClient

    client = OpenAIClient()                       # reads OPENAI_API_KEY from the environment
    qa = DocumentQA.from_file(
        "contract.pdf", client=client, config=QAConfig(model="<a chat model you can use>")
    )
    print(qa.ask("What is the notice period?").text)

Requests go straight to ``/chat/completions`` over HTTPS using the Python standard library
(see :mod:`longdoc.http`), with retries and clear errors.

**Your API key** is never stored in your code or logged. It is read from the ``OPENAI_API_KEY``
environment variable (or passed as ``api_key=``) and held in a wrapper that hides it from
``repr()``. See ``docs/OPENAI.md`` for safe ways to provide it.

Differences from the Claude path: Claude-only options (effort, refusal fallbacks, explicit
cache markers) are ignored. OpenAI caches long, repeated prompt prefixes automatically, so
``Strategy.CACHED_DOCUMENT`` can still reduce cost; cached tokens are reported in
``answer.usage.cache_read_input_tokens``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Literal

from .aio import AsyncAdapter
from .exceptions import ConfigurationError
from .http import ProviderQuotaError, Response, Secret, Transport, post_json
from .local import (
    LocalMessage,
    LocalStopDetails,
    LocalTextBlock,
    LocalUsage,
    _Stream,
    to_chat_messages,
)

TokenParam = Literal["max_completion_tokens", "max_tokens"]
DEFAULT_BASE_URL = "https://api.openai.com/v1"

_NO_KEY = (
    "No OpenAI API key found. Set the OPENAI_API_KEY environment variable (see docs/OPENAI.md), "
    "or pass api_key=... . A local OpenAI-compatible server that needs no key can be used "
    "by passing base_url=... instead."
)


def is_out_of_credit(exc: BaseException) -> bool:
    """True if ``exc`` means the account has no credit or quota left (retrying never helps)."""
    return isinstance(exc, ProviderQuotaError)


def build_request(
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: Any,
    model: str | None,
    token_param: TokenParam,
    temperature: float | None,
    reasoning_effort: str | None,
    custom_server: bool = False,
) -> dict[str, Any]:
    """Translate longdoc's Anthropic-style request into chat-completions arguments."""
    if not model:
        raise ConfigurationError(
            "A model is required for OpenAI: set QAConfig(model=...), `--model`, or LONGDOC_MODEL."
        )
    if not custom_server and model.lower().startswith("claude"):
        raise ConfigurationError(
            f"{model!r} is a Claude model, but this client talks to OpenAI. Set "
            "QAConfig(model=...), `--model`, or LONGDOC_MODEL to a chat model your OpenAI "
            "account can use."
        )
    body: dict[str, Any] = {
        "model": model,
        "messages": to_chat_messages(system, messages),
        token_param: max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return body


def parse_response(response: Response, requested_model: str) -> LocalMessage:
    """Convert a chat-completions response into the message shape longdoc reads."""
    data = response.body
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    refusal = message.get("refusal")
    finish = choice.get("finish_reason")

    if refusal or finish == "content_filter":
        stop_reason = "refusal"
        details: LocalStopDetails | None = LocalStopDetails(
            category="content_filter" if finish == "content_filter" else "refusal",
            explanation=refusal or None,
        )
    else:
        stop_reason = "max_tokens" if finish == "length" else "end_turn"
        details = None

    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    text = (message.get("content") or "").strip()
    return LocalMessage(
        content=[LocalTextBlock(text)] if text else [],
        model=str(data.get("model") or requested_model),
        stop_reason=stop_reason,
        usage=LocalUsage(
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read_input_tokens=cached,
        ),
        stop_details=details,
        _request_id=response.request_id,
    )


class _Messages:
    def __init__(self, owner: OpenAIClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> LocalMessage:
        """Run one completion. Claude-only options are accepted and ignored."""
        return self._owner.complete(**kwargs)

    def stream(self, **kwargs: Any) -> _Stream:
        """Same as :meth:`create`, wrapped like the Anthropic streaming helper."""
        return _Stream(self._owner.complete(**kwargs))


class OpenAIClient:
    """A dependency-free client for OpenAI (or OpenAI-compatible) chat models.

    Args:
        api_key: Your key. Prefer leaving this unset and using ``OPENAI_API_KEY``.
        base_url: An OpenAI-compatible server (for example a local one). A key is optional
            when this is set. Defaults to ``OPENAI_BASE_URL`` or the public API.
        token_param: ``"max_completion_tokens"`` (current OpenAI models) or ``"max_tokens"``
            (many compatible servers).
        temperature: Sent only if set. Some models accept only their default.
        reasoning_effort: Passed through for reasoning models (for example ``"low"``).
        timeout: Seconds to wait for a response.
        max_retries: Automatic retries for transient failures (rate limit, outage, network).
        allow_insecure: Permit plain ``http://`` to a non-local host (not recommended).
        transport: Replace the HTTP call, for tests: ``(url, headers, body) -> Response``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        token_param: TokenParam = "max_completion_tokens",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        allow_insecure: bool = False,
        transport: Transport | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key and not base_url:
            raise ConfigurationError(_NO_KEY)
        self._key = Secret(key) if key else None
        self.custom_server = base_url is not None
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.token_param = token_param
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
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
        return f"OpenAIClient(base_url={self.base_url!r}, token_param={self.token_param!r})"

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: Any = None,
        model: str | None = None,
        **_ignored: Any,
    ) -> LocalMessage:
        """Generate an answer. Effort, caching and fallback options are ignored."""
        if max_tokens == 0:  # cache warm-up request: nothing to generate
            return LocalMessage([], model or "", "max_tokens", LocalUsage(0, 0))
        body = build_request(
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            model=model,
            token_param=self.token_param,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            custom_server=self.custom_server,
        )
        headers: Mapping[str, str] = (
            {"authorization": f"Bearer {self._key.reveal()}"} if self._key else {}
        )
        response = self._transport(f"{self.base_url}/chat/completions", headers, body)
        return parse_response(response, body["model"])

    def to_async(self) -> AsyncAdapter:
        """An async view of this client."""
        return AsyncAdapter(self)
