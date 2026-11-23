"""Run longdoc on a local open-source model, with no API key and no network.

:class:`LlamaCppClient` and :class:`AsyncLlamaCppClient` run a GGUF model through
`llama.cpp <https://github.com/ggml-org/llama.cpp>`_ (``pip install "longdoc-qa[local]"``)
and expose the small part of the Anthropic client interface that longdoc uses
(``client.beta.messages.stream`` and ``create``). Pass one wherever longdoc accepts a
client:

.. code-block:: python

    from longdoc import DocumentQA
    from longdoc.local import LlamaCppClient

    llm = LlamaCppClient("models/Phi-3.5-mini-instruct-Q4_K_M.gguf", n_ctx=16384)
    qa = DocumentQA.from_file("contract.pdf", client=llm)
    print(qa.ask("What is the notice period?").text)

Use it for offline, air-gapped or no-cost deployments. Expect lower accuracy than a
frontier model and much slower answers on CPU. Claude-specific request options
(prompt caching, refusal fallbacks, effort) are ignored, so prefer
``Strategy.RETRIEVAL``: it keeps prompts small enough for local context windows.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .exceptions import LongDocError


class ContextWindowExceededError(LongDocError):
    """The prompt does not fit in the local model's context window."""

    def __init__(self, prompt_tokens: int, n_ctx: int, max_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.n_ctx = n_ctx
        super().__init__(
            f"Prompt needs {prompt_tokens:,} tokens plus {max_tokens:,} for the answer, but the "
            f"local model's context window is {n_ctx:,}. Use Strategy.RETRIEVAL, lower "
            "max_parents / max_context_chars, or raise n_ctx."
        )


# -- response objects shaped like the parts of anthropic's Message that longdoc reads -----


@dataclass(frozen=True, slots=True)
class LocalTextBlock:
    """A text content block."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class LocalUsage:
    """Token usage. Local models have no billed prompt cache, so cache fields are 0."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    iterations: None = None


@dataclass(frozen=True, slots=True)
class LocalStopDetails:
    """Why a response was refused: a category and the provider's explanation, if any."""

    category: str | None = None
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class LocalMessage:
    """A completed local-model response."""

    content: list[LocalTextBlock]
    model: str
    stop_reason: str
    usage: LocalUsage
    stop_details: LocalStopDetails | None = None
    _request_id: str | None = None


# -- helpers ----------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(
        block["text"] if isinstance(block, dict) else str(getattr(block, "text", ""))
        for block in content
        if (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
        == "text"
    )


def to_chat_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert Anthropic-style ``system`` + ``messages`` into OpenAI-style chat messages."""
    chat: list[dict[str, str]] = []
    if system:
        chat.append({"role": "system", "content": _text_of(system)})
    for m in messages:
        role = "system" if m["role"] == "system" else m["role"]
        chat.append({"role": role, "content": _text_of(m["content"])})
    return chat


# -- clients ----------------------------------------------------------------------------


class _Stream:
    """Context manager returned by ``messages.stream`` (the answer is computed up front)."""

    def __init__(self, message: LocalMessage) -> None:
        self._message = message

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> LocalMessage:
        """Return the completed message."""
        return self._message


@dataclass
class _Messages:
    owner: LlamaCppClient

    def create(self, **kwargs: Any) -> LocalMessage:
        """Run one completion. Accepts (and ignores) Claude-only options."""
        return self.owner.complete(**kwargs)

    def stream(self, **kwargs: Any) -> _Stream:
        """Same as :meth:`create`, wrapped like the Anthropic streaming helper."""
        return _Stream(self.owner.complete(**kwargs))


@dataclass
class _Beta:
    messages: _Messages


class LlamaCppClient:
    """A local GGUF model, used in place of ``anthropic.Anthropic``.

    Args:
        model_path: Path to a ``.gguf`` model file.
        n_ctx: Context window to allocate, in tokens. Prompts plus answers must fit.
        n_threads: CPU threads (defaults to all cores).
        temperature: Sampling temperature; 0 gives repeatable answers.
        llama: An existing ``llama_cpp.Llama`` instance (for tests or custom setups);
            ``model_path`` is then only used as the model name.
        **llama_kwargs: Passed to ``llama_cpp.Llama`` (for example ``n_gpu_layers``).

    The model is not thread-safe, so calls are serialised with a lock.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        n_ctx: int = 16_384,
        n_threads: int | None = None,
        temperature: float = 0.0,
        llama: Any = None,
        **llama_kwargs: Any,
    ) -> None:
        self.model_name = f"local:{Path(model_path).name}"
        self.n_ctx = n_ctx
        self.temperature = temperature
        if llama is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise ImportError(
                    'Local models need llama-cpp-python: pip install "longdoc-qa[local]"'
                ) from exc
            llama = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads or os.cpu_count(),
                verbose=False,
                **llama_kwargs,
            )
        self._llama = llama
        self._lock = threading.Lock()
        self.beta = _Beta(messages=_Messages(self))
        self.messages = self.beta.messages

    def count_tokens(self, chat: list[dict[str, str]]) -> int:
        """Approximate prompt tokens for ``chat`` (chat-template overhead excluded)."""
        text = "\n".join(m["content"] for m in chat)
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=True))

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: Any = None,
        model: str | None = None,
        **_ignored: Any,
    ) -> LocalMessage:
        """Generate an answer. ``model``, caching, effort and fallback options are ignored."""
        chat = to_chat_messages(system, messages)
        prompt_tokens = self.count_tokens(chat)
        if max_tokens == 0:  # cache warm-up request: nothing to generate
            return LocalMessage([], self.model_name, "max_tokens", LocalUsage(prompt_tokens, 0))
        budget = min(max_tokens, self.n_ctx - prompt_tokens - 64)
        if budget < 256:
            raise ContextWindowExceededError(prompt_tokens, self.n_ctx, min(max_tokens, 256))

        with self._lock:
            result = self._llama.create_chat_completion(
                messages=chat, max_tokens=budget, temperature=self.temperature
            )
        choice = result["choices"][0]
        usage = result.get("usage") or {}
        return LocalMessage(
            content=[LocalTextBlock((choice["message"].get("content") or "").strip())],
            model=self.model_name,
            stop_reason="max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
            usage=LocalUsage(
                input_tokens=int(usage.get("prompt_tokens", prompt_tokens)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
        )


# -- async ------------------------------------------------------------------------------


class _AsyncStream:
    def __init__(self, coro: Any) -> None:
        self._coro = coro
        self._message: LocalMessage | None = None

    async def __aenter__(self) -> _AsyncStream:
        self._message = await self._coro
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_final_message(self) -> LocalMessage:
        """Return the completed message."""
        if self._message is None:
            raise RuntimeError("get_final_message() called outside 'async with'")
        return self._message


@dataclass
class _AsyncMessages:
    sync: LlamaCppClient = field(repr=False)

    async def create(self, **kwargs: Any) -> LocalMessage:
        """Run one completion in a worker thread, so the event loop stays free."""
        return await asyncio.to_thread(self.sync.complete, **kwargs)

    def stream(self, **kwargs: Any) -> _AsyncStream:
        """Async-context-manager form of :meth:`create`."""
        return _AsyncStream(self.create(**kwargs))


@dataclass
class _AsyncBeta:
    messages: _AsyncMessages


class AsyncLlamaCppClient:
    """Async wrapper used in place of ``anthropic.AsyncAnthropic``.

    Shares one loaded model with a :class:`LlamaCppClient`, so a process holds the model
    in memory once.
    """

    def __init__(self, sync_client: LlamaCppClient) -> None:
        self.sync_client = sync_client
        self.model_name = sync_client.model_name
        self.beta = _AsyncBeta(messages=_AsyncMessages(sync_client))
        self.messages = self.beta.messages


# -- embeddings -------------------------------------------------------------------------

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LlamaCppEmbedder:
    """A local GGUF embedding model for :class:`longdoc.EmbeddingRetriever`.

    Example (MIT-licensed bge-small-en-v1.5, ~36 MB)::

        from longdoc import BM25Retriever, DocumentService, EmbeddingRetriever, HybridRetriever
        from longdoc.local import BGE_QUERY_PREFIX, LlamaCppEmbedder

        embed = LlamaCppEmbedder("models/bge-small-en-v1.5-q8_0.gguf")
        service = DocumentService(
            retriever_factory=lambda: HybridRetriever(
                BM25Retriever(), EmbeddingRetriever(embed, query_prefix=BGE_QUERY_PREFIX)
            ),
        )

    Args:
        model_path: Path to a GGUF embedding model.
        n_ctx: Maximum tokens per text; longer texts are truncated by the model.
        llama: An existing ``llama_cpp.Llama`` created with ``embedding=True``.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        n_ctx: int = 512,
        n_threads: int | None = None,
        llama: Any = None,
        **llama_kwargs: Any,
    ) -> None:
        if llama is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise ImportError(
                    'Local models need llama-cpp-python: pip install "longdoc-qa[local]"'
                ) from exc
            llama = Llama(
                model_path=str(model_path),
                embedding=True,
                n_ctx=n_ctx,
                n_threads=n_threads or os.cpu_count(),
                verbose=False,
                **llama_kwargs,
            )
        self._llama = llama
        self._lock = threading.Lock()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``; returns one vector per text, in order."""
        with self._lock:
            result = self._llama.create_embedding(texts)
        data = sorted(result["data"], key=lambda d: d["index"])
        return [list(map(float, d["embedding"])) for d in data]
