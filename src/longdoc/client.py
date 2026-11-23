"""High-level question answering over one document.

:class:`DocumentQA` (sync) and :class:`AsyncDocumentQA` (async) share all prompt and
parsing logic through :class:`_QACore`; they differ only in how they call the API.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import anthropic

from .chunking import ParentChildChunker
from .config import ChunkingConfig, QAConfig, Strategy
from .exceptions import ConfigurationError, RefusalError
from .loaders import load_document
from .local import AsyncLlamaCppClient, LlamaCppClient, LocalMessage
from .models import Answer, ChunkedDocument, Document, ParentChunk, TokenUsage
from .prompts import SYSTEM_PROMPT, clean_answer, extract_citations, system_blocks, user_message
from .retrieval import BM25Retriever, Retriever, select_parents

if TYPE_CHECKING:
    from anthropic.types.beta import BetaMessage

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_WARMUP_QUESTION = "(cache warm-up)"


class _QACore:
    """Chunking, retrieval, request building and response parsing (no I/O)."""

    def __init__(
        self,
        document: Document | ChunkedDocument,
        *,
        config: QAConfig | None,
        chunking: ChunkingConfig | None,
        retriever: Retriever | None,
    ) -> None:
        self.config = config or QAConfig()
        self.chunks = (
            document
            if isinstance(document, ChunkedDocument)
            else ParentChildChunker(chunking).chunk(document)
        )
        self.retriever: Retriever = retriever or BM25Retriever()
        self.retriever.index(self.chunks.children)
        self._usage = TokenUsage()
        self._lock = threading.Lock()

    @property
    def usage(self) -> TokenUsage:
        """Cumulative token usage across every call made by this instance."""
        return self._usage

    def reset_usage(self) -> None:
        """Reset the cumulative usage counter to zero."""
        with self._lock:
            self._usage = TokenUsage()

    def retrieve(self, question: str) -> tuple[ParentChunk, ...]:
        """Return the parent sections that would be sent for ``question``.

        Useful for debugging retrieval without calling the API.
        """
        hits = self.retriever.search(question, self.config.top_k_children)
        return select_parents(
            hits,
            self.chunks,
            max_parents=self.config.max_parents,
            max_chars=self.config.max_context_chars,
        )

    # -- request building ----------------------------------------------------------

    def _base_kwargs(self, question: str, context: Sequence[ParentChunk] | None) -> dict[str, Any]:
        """``context=None`` means: put the whole document in the cached system prefix."""
        instructions = self.config.system_prompt or SYSTEM_PROMPT
        if context is None:
            system = system_blocks(
                instructions=instructions, document=self.chunks, cache_ttl=self.config.cache_ttl
            )
        else:
            system = system_blocks(instructions=instructions)
        messages = [user_message(question, context)]
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.config.effort},
        }

    def _fallback_kwargs(self) -> dict[str, Any]:
        fallbacks = self.config.fallbacks
        if fallbacks is None:
            return {}
        value: Any = "default" if fallbacks == "default" else [{"model": m} for m in fallbacks]
        return {"betas": [FALLBACK_BETA], "fallbacks": value}

    def build_request(
        self, question: str, strategy: Strategy | None = None
    ) -> tuple[dict[str, Any], tuple[ParentChunk, ...]]:
        """Return the Messages API keyword arguments and the sections they contain."""
        if not question.strip():
            raise ConfigurationError("question must not be empty.")
        strategy = Strategy(strategy or self.config.strategy)
        if strategy is Strategy.CACHED_DOCUMENT:
            kwargs = self._base_kwargs(question, None)
            context = self.chunks.parents
        else:
            context = self.retrieve(question)
            kwargs = self._base_kwargs(question, context)
        return kwargs | self._fallback_kwargs(), context

    def build_warmup_request(self) -> dict[str, Any]:
        """Return a ``max_tokens=0`` request that writes the document to the cache."""
        kwargs = self._base_kwargs(_WARMUP_QUESTION, None)
        kwargs["max_tokens"] = 0
        return kwargs

    # -- response parsing ----------------------------------------------------------

    def record_usage(self, message: BetaMessage | LocalMessage) -> TokenUsage:
        u = message.usage
        usage = TokenUsage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
        )
        with self._lock:
            self._usage = self._usage + usage
        logger.debug(
            "request_id=%s model=%s usage=%s",
            getattr(message, "_request_id", None),
            message.model,
            usage,
        )
        return usage

    def to_answer(
        self, message: BetaMessage | LocalMessage, context: Sequence[ParentChunk]
    ) -> Answer:
        usage = self.record_usage(message)
        request_id = getattr(message, "_request_id", None)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise RefusalError(
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
                request_id=request_id,
            )

        text = clean_answer("".join(b.text for b in message.content if b.type == "text"))
        iterations = getattr(message.usage, "iterations", None) or []
        fallback_used = any(b.type == "fallback" for b in message.content) or any(
            getattr(entry, "type", None) == "fallback_message" for entry in iterations
        )
        if message.stop_reason == "max_tokens":
            logger.warning("Answer hit max_tokens=%d and may be cut off.", self.config.max_tokens)

        return Answer(
            text=text,
            citations=extract_citations(text, self.chunks),
            context=tuple(context),
            usage=usage,
            model=message.model,
            stop_reason=message.stop_reason,
            truncated=message.stop_reason == "max_tokens",
            fallback_used=fallback_used,
            request_id=request_id,
        )


class DocumentQA(_QACore):
    """Ask questions about one long document.

    Args:
        document: A loaded :class:`Document`, or an already-chunked document.
        config: Question-answering settings.
        chunking: Chunk sizes (ignored if ``document`` is already chunked).
        retriever: Custom search backend. Defaults to :class:`BM25Retriever`.
        client: A configured ``anthropic.Anthropic`` client, or a
            :class:`longdoc.local.LlamaCppClient` to run on a local model. Defaults to an
            Anthropic client that reads credentials from the environment
            (``ANTHROPIC_API_KEY`` and friends).

    Example:
        >>> qa = DocumentQA.from_file("contract.pdf")            # doctest: +SKIP
        >>> answer = qa.ask("What is the termination notice period?")  # doctest: +SKIP
        >>> print(answer.text, [s.id for s in answer.citations])  # doctest: +SKIP
    """

    def __init__(
        self,
        document: Document | ChunkedDocument,
        *,
        config: QAConfig | None = None,
        chunking: ChunkingConfig | None = None,
        retriever: Retriever | None = None,
        client: anthropic.Anthropic | LlamaCppClient | None = None,
    ) -> None:
        super().__init__(document, config=config, chunking=chunking, retriever=retriever)
        self.client: anthropic.Anthropic | LlamaCppClient = client or anthropic.Anthropic()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], **kwargs: Any) -> DocumentQA:
        """Load ``path`` (``.txt``, ``.md``, ``.pdf``) and build an instance."""
        return cls(load_document(path), **kwargs)

    @classmethod
    def from_text(cls, text: str, *, source: str | None = None, **kwargs: Any) -> DocumentQA:
        """Build an instance from a string."""
        return cls(Document.from_text(text, source=source), **kwargs)

    def ask(self, question: str, *, strategy: Strategy | None = None) -> Answer:
        """Answer ``question`` from the document.

        Raises:
            RefusalError: The model (and any fallback) declined to answer.
            anthropic.APIError: Any API failure, after the SDK's automatic retries.
        """
        kwargs, context = self.build_request(question, strategy)
        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        return self.to_answer(message, context)

    def warm_cache(self) -> TokenUsage:
        """Write the full document to the prompt cache ahead of the first question.

        Only useful with :attr:`Strategy.CACHED_DOCUMENT`, when the first question's
        latency matters. Costs one cache write; bills no output tokens.
        """
        message = self.client.beta.messages.create(**self.build_warmup_request())
        return self.record_usage(message)


class AsyncDocumentQA(_QACore):
    """Async version of :class:`DocumentQA`; takes an ``anthropic.AsyncAnthropic``."""

    def __init__(
        self,
        document: Document | ChunkedDocument,
        *,
        config: QAConfig | None = None,
        chunking: ChunkingConfig | None = None,
        retriever: Retriever | None = None,
        client: anthropic.AsyncAnthropic | AsyncLlamaCppClient | None = None,
    ) -> None:
        super().__init__(document, config=config, chunking=chunking, retriever=retriever)
        self.client: anthropic.AsyncAnthropic | AsyncLlamaCppClient = (
            client or anthropic.AsyncAnthropic()
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], **kwargs: Any) -> AsyncDocumentQA:
        """Load ``path`` (``.txt``, ``.md``, ``.pdf``) and build an instance."""
        return cls(load_document(path), **kwargs)

    @classmethod
    def from_text(cls, text: str, *, source: str | None = None, **kwargs: Any) -> AsyncDocumentQA:
        """Build an instance from a string."""
        return cls(Document.from_text(text, source=source), **kwargs)

    async def ask(self, question: str, *, strategy: Strategy | None = None) -> Answer:
        """Answer ``question`` from the document. See :meth:`DocumentQA.ask`."""
        kwargs, context = self.build_request(question, strategy)
        async with self.client.beta.messages.stream(**kwargs) as stream:
            message = await stream.get_final_message()
        return self.to_answer(message, context)

    async def warm_cache(self) -> TokenUsage:
        """Write the full document to the prompt cache. See :meth:`DocumentQA.warm_cache`."""
        message = await self.client.beta.messages.create(**self.build_warmup_request())
        return self.record_usage(message)
