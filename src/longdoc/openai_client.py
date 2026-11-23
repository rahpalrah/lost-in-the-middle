"""Use OpenAI, or any OpenAI-compatible server, instead of Claude.

:class:`OpenAIClient` and :class:`AsyncOpenAIClient` expose the small part of the Anthropic
client interface that longdoc uses, so they work anywhere longdoc accepts a client
(``DocumentQA``, ``DocumentService``, the CLI and the HTTP server):

.. code-block:: python

    from longdoc import DocumentQA, QAConfig
    from longdoc.openai_client import OpenAIClient

    client = OpenAIClient()                       # reads OPENAI_API_KEY from the environment
    qa = DocumentQA.from_file(
        "contract.pdf", client=client, config=QAConfig(model="<a chat model you can use>")
    )
    print(qa.ask("What is the notice period?").text)

Install the dependency with ``pip install "longdoc-qa[openai]"``.

**Your API key** is never stored in your code or logged. It is read from the ``OPENAI_API_KEY``
environment variable (or passed as ``api_key=``), and this library never prints or returns it.
See ``docs/OPENAI.md`` for safe ways to provide it.

Differences from the Claude path: Claude-only options (effort, refusal fallbacks, explicit
cache markers) are ignored. OpenAI caches long, repeated prompt prefixes automatically, so
``Strategy.CACHED_DOCUMENT`` can still reduce cost; cached tokens are reported in
``answer.usage.cache_read_input_tokens``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from .exceptions import ConfigurationError
from .local import (
    LocalMessage,
    LocalStopDetails,
    LocalTextBlock,
    LocalUsage,
    _AsyncStream,
    _Stream,
    to_chat_messages,
)

TokenParam = Literal["max_completion_tokens", "max_tokens"]

_NO_KEY = (
    "No OpenAI API key found. Set the OPENAI_API_KEY environment variable (see docs/OPENAI.md), "
    "or pass api_key=... . A local OpenAI-compatible server that needs no key can be used "
    "by passing base_url=... instead."
)


def _import_openai() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            'The OpenAI adapter needs the openai package: pip install "longdoc-qa[openai]"'
        ) from exc
    return openai


def build_request(
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: Any,
    model: str | None,
    token_param: TokenParam,
    temperature: float | None,
    reasoning_effort: str | None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Translate longdoc's Anthropic-style request into OpenAI chat-completions arguments."""
    if not model:
        raise ConfigurationError(
            "A model is required for OpenAI: set QAConfig(model=...), `--model`, or LONGDOC_MODEL."
        )
    if base_url is None and model.lower().startswith("claude"):
        raise ConfigurationError(
            f"{model!r} is a Claude model, but this client talks to OpenAI. Set "
            "QAConfig(model=...), `--model`, or LONGDOC_MODEL to a chat model your OpenAI "
            "account can use."
        )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": to_chat_messages(system, messages),
        token_param: max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def parse_response(response: Any, requested_model: str) -> LocalMessage:
    """Convert an OpenAI chat completion into the message shape longdoc reads."""
    choice = response.choices[0]
    message = choice.message
    refusal = getattr(message, "refusal", None)
    finish = getattr(choice, "finish_reason", None)

    if refusal or finish == "content_filter":
        stop_reason = "refusal"
        details: LocalStopDetails | None = LocalStopDetails(
            category="content_filter" if finish == "content_filter" else "refusal",
            explanation=refusal or None,
        )
    else:
        stop_reason = "max_tokens" if finish == "length" else "end_turn"
        details = None

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
    text = (message.content or "").strip()
    return LocalMessage(
        content=[LocalTextBlock(text)] if text else [],
        model=getattr(response, "model", None) or requested_model,
        stop_reason=stop_reason,
        usage=LocalUsage(
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=output_tokens,
            cache_read_input_tokens=cached,
        ),
        stop_details=details,
        _request_id=getattr(response, "_request_id", None),
    )


_OUT_OF_CREDIT_CODES = frozenset(
    {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}
)


def is_out_of_credit(exc: BaseException) -> bool:
    """True if an OpenAI error means the account has no credit or quota left.

    OpenAI reports this as HTTP 429, the same status as ordinary rate limiting, but retrying
    never helps. Seen in practice: ``type="insufficient_quota"`` with
    ``code="credit_balance_exhausted"``.
    """
    return (
        getattr(exc, "type", None) == "insufficient_quota"
        or getattr(exc, "code", None) in _OUT_OF_CREDIT_CODES
    )


def _warmup_message(model: str | None) -> LocalMessage:
    return LocalMessage([], model or "", "max_tokens", LocalUsage(0, 0))


class _Messages:
    def __init__(self, owner: OpenAIClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> LocalMessage:
        """Run one completion. Claude-only options are accepted and ignored."""
        return self._owner.complete(**kwargs)

    def stream(self, **kwargs: Any) -> _Stream:
        """Same as :meth:`create`, wrapped like the Anthropic streaming helper."""
        return _Stream(self._owner.complete(**kwargs))


class _Beta:
    def __init__(self, messages: Any) -> None:
        self.messages = messages


class OpenAIClient:
    """A drop-in client that answers with an OpenAI (or OpenAI-compatible) chat model.

    Args:
        api_key: Your key. Prefer leaving this unset and using ``OPENAI_API_KEY``.
        base_url: Point at an OpenAI-compatible server (for example a local one).
            A key is optional when this is set.
        token_param: ``"max_completion_tokens"`` (current OpenAI models) or ``"max_tokens"``
            (many compatible servers).
        temperature: Sent only if set. Some models accept only their default.
        reasoning_effort: Passed through for reasoning models (for example ``"low"``).
        client: An existing ``openai.OpenAI`` (or a test double), instead of creating one.
        async_client: The matching async client, for :meth:`to_async`.
        **openai_kwargs: Passed to ``openai.OpenAI`` (``timeout``, ``max_retries``, ...).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        token_param: TokenParam = "max_completion_tokens",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        client: Any = None,
        async_client: Any = None,
        **openai_kwargs: Any,
    ) -> None:
        self.token_param = token_param
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self._openai_kwargs = openai_kwargs
        self._async_sdk = async_client
        if client is None:
            openai = _import_openai()
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and not base_url:
                raise ConfigurationError(_NO_KEY)
            client = openai.OpenAI(api_key=key or "not-needed", base_url=base_url, **openai_kwargs)
        self._sdk = client
        self.beta = _Beta(_Messages(self))
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
            return _warmup_message(model)
        kwargs = build_request(
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            model=model,
            token_param=self.token_param,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            base_url=self.base_url,
        )
        return parse_response(self._sdk.chat.completions.create(**kwargs), kwargs["model"])

    def to_async(self) -> AsyncOpenAIClient:
        """An async client with the same credentials and settings."""
        if self._async_sdk is not None:
            return AsyncOpenAIClient(client=self._async_sdk, **self._shared())
        openai = _import_openai()
        sdk = openai.AsyncOpenAI(
            api_key=self._sdk.api_key,
            base_url=self._sdk.base_url,
            max_retries=self._sdk.max_retries,
            timeout=self._sdk.timeout,
        )
        return AsyncOpenAIClient(client=sdk, **self._shared())

    def _shared(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "token_param": self.token_param,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


class _AsyncMessages:
    def __init__(self, owner: AsyncOpenAIClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> LocalMessage:
        """Run one completion without blocking the event loop."""
        return await self._owner.complete(**kwargs)

    def stream(self, **kwargs: Any) -> _AsyncStream:
        """Async-context-manager form of :meth:`create`."""
        return _AsyncStream(self.create(**kwargs))


class AsyncOpenAIClient:
    """Async counterpart of :class:`OpenAIClient` (same options)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        token_param: TokenParam = "max_completion_tokens",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        client: Any = None,
        **openai_kwargs: Any,
    ) -> None:
        self.token_param = token_param
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        if client is None:
            openai = _import_openai()
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and not base_url:
                raise ConfigurationError(_NO_KEY)
            client = openai.AsyncOpenAI(
                api_key=key or "not-needed", base_url=base_url, **openai_kwargs
            )
        self._sdk = client
        self.beta = _Beta(_AsyncMessages(self))
        self.messages = self.beta.messages

    def __repr__(self) -> str:  # never include credentials
        return f"AsyncOpenAIClient(base_url={self.base_url!r}, token_param={self.token_param!r})"

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: Any = None,
        model: str | None = None,
        **_ignored: Any,
    ) -> LocalMessage:
        """Generate an answer. Effort, caching and fallback options are ignored."""
        if max_tokens == 0:
            return _warmup_message(model)
        kwargs = build_request(
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            model=model,
            token_param=self.token_param,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            base_url=self.base_url,
        )
        response = await self._sdk.chat.completions.create(**kwargs)
        return parse_response(response, kwargs["model"])
