"""A multi-document service: the main integration point for existing applications.

:class:`DocumentService` owns a :class:`DocumentStore`, one shared Anthropic client,
and a bounded cache of ready-to-query documents. It has no web-framework dependency,
so it drops into Django, Flask, FastAPI, Celery workers, Lambda handlers or a CLI:

.. code-block:: python

    service = DocumentService(store=JSONFileStore("/var/lib/longdoc"))
    info = service.add_file("contract.pdf")
    answer = service.ask(info.id, "What is the notice period?")

Instances are thread-safe. Create one per process and reuse it.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic

from .chunking import ParentChildChunker
from .client import AsyncDocumentQA, DocumentQA
from .config import ChunkingConfig, QAConfig, Strategy
from .exceptions import LongDocError
from .loaders import load_bytes, load_document
from .local import AsyncLlamaCppClient, LlamaCppClient
from .models import Answer, ChunkedDocument, Document, ParentChunk, TokenUsage
from .openai_client import AsyncOpenAIClient, OpenAIClient
from .retrieval import BM25Retriever, Retriever
from .store import DocumentStore, InMemoryStore, validate_document_id


class DocumentNotFoundError(LongDocError, KeyError):
    """No document with the requested ID exists in the store."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document {document_id!r} not found.")

    def __str__(self) -> str:  # KeyError would otherwise quote the message
        return str(self.args[0])


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """Summary of a stored document."""

    id: str
    source: str | None
    sections: int
    passages: int
    pages: int | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "id": self.id,
            "source": self.source,
            "sections": self.sections,
            "passages": self.passages,
            "pages": self.pages,
        }


def _info(document_id: str, chunked: ChunkedDocument) -> DocumentInfo:
    last_pages = [p.page_end for p in chunked.parents if p.page_end is not None]
    return DocumentInfo(
        id=document_id,
        source=chunked.source,
        sections=len(chunked.parents),
        passages=len(chunked.children),
        pages=max(last_pages) if last_pages else None,
    )


class DocumentService:
    """Add, query and remove many documents through one object.

    Args:
        store: Where chunked documents are kept. Defaults to :class:`InMemoryStore`.
        config: Question-answering settings shared by every document.
        chunking: Chunk sizes used when documents are added.
        client: Sync Anthropic client for :meth:`ask`. Created lazily from the
            environment if omitted.
        async_client: Async Anthropic client for :meth:`aask`. Created lazily if omitted.
        retriever_factory: Builds a fresh :class:`Retriever` per document. Defaults to
            :class:`BM25Retriever`.
        max_cached_documents: How many indexed documents to keep in memory (LRU).
    """

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        config: QAConfig | None = None,
        chunking: ChunkingConfig | None = None,
        client: anthropic.Anthropic | LlamaCppClient | OpenAIClient | None = None,
        async_client: (
            anthropic.AsyncAnthropic | AsyncLlamaCppClient | AsyncOpenAIClient | None
        ) = None,
        retriever_factory: Callable[[], Retriever] = BM25Retriever,
        max_cached_documents: int = 64,
    ) -> None:
        self.store: DocumentStore = store if store is not None else InMemoryStore()
        self.config = config or QAConfig()
        self.chunker = ParentChildChunker(chunking)
        self._client = client
        self._async_client = async_client
        if isinstance(client, LlamaCppClient) and async_client is None:
            # Share the loaded model rather than loading a second copy.
            self._async_client = AsyncLlamaCppClient(client)
        self._retriever_factory = retriever_factory
        self._max_cached = max(1, max_cached_documents)
        self._sync_qa: OrderedDict[str, DocumentQA] = OrderedDict()
        self._async_qa: OrderedDict[str, AsyncDocumentQA] = OrderedDict()
        self._lock = threading.RLock()
        self._usage = TokenUsage()

    # -- clients ---------------------------------------------------------------------

    @property
    def client(self) -> anthropic.Anthropic | LlamaCppClient | OpenAIClient:
        """The shared sync client (created on first use)."""
        with self._lock:
            if self._client is None:
                self._client = anthropic.Anthropic()
            return self._client

    @property
    def async_client(self) -> anthropic.AsyncAnthropic | AsyncLlamaCppClient | AsyncOpenAIClient:
        """The shared async client (created on first use)."""
        with self._lock:
            if self._async_client is None:
                if isinstance(self._client, LlamaCppClient):
                    self._async_client = AsyncLlamaCppClient(self._client)
                elif isinstance(self._client, OpenAIClient):
                    self._async_client = self._client.to_async()
                else:
                    self._async_client = anthropic.AsyncAnthropic()
            return self._async_client

    # -- documents -------------------------------------------------------------------

    def add_document(self, document: Document, *, document_id: str | None = None) -> DocumentInfo:
        """Chunk and store ``document``. Replaces any document with the same ID.

        Args:
            document: The loaded document.
            document_id: Your own ID (for example, your database key). A random ID is
                generated if omitted.
        """
        doc_id = validate_document_id(document_id) if document_id else uuid.uuid4().hex
        chunked = self.chunker.chunk(document)
        self.store.put(doc_id, chunked)
        self._evict(doc_id)
        return _info(doc_id, chunked)

    def add_text(
        self, text: str, *, source: str | None = None, document_id: str | None = None
    ) -> DocumentInfo:
        """Add a document from a string."""
        return self.add_document(Document.from_text(text, source=source), document_id=document_id)

    def add_file(
        self, path: str | os.PathLike[str], *, document_id: str | None = None
    ) -> DocumentInfo:
        """Add a document from a file on disk."""
        return self.add_document(load_document(path), document_id=document_id)

    def add_bytes(
        self, data: bytes, *, filename: str, document_id: str | None = None
    ) -> DocumentInfo:
        """Add a document from uploaded bytes; ``filename`` picks the parser."""
        return self.add_document(load_bytes(data, filename=filename), document_id=document_id)

    def get_info(self, document_id: str) -> DocumentInfo:
        """Return a summary of a stored document.

        Raises:
            DocumentNotFoundError: No such document.
        """
        return _info(document_id, self._load(document_id))

    def list_documents(self) -> list[DocumentInfo]:
        """Summaries of every stored document."""
        infos = []
        for doc_id in self.store.list_ids():
            chunked = self.store.get(doc_id)
            if chunked is not None:
                infos.append(_info(doc_id, chunked))
        return infos

    def delete(self, document_id: str) -> bool:
        """Remove a document; return ``True`` if it existed."""
        validate_document_id(document_id)
        self._evict(document_id)
        return self.store.delete(document_id)

    # -- questions -------------------------------------------------------------------

    def retrieve(self, document_id: str, question: str) -> tuple[ParentChunk, ...]:
        """Return the sections that would be sent for ``question`` (no API call)."""
        return self._qa_sync(document_id).retrieve(question)

    def ask(self, document_id: str, question: str, *, strategy: Strategy | None = None) -> Answer:
        """Answer ``question`` about one document. See :meth:`DocumentQA.ask`."""
        answer = self._qa_sync(document_id).ask(question, strategy=strategy)
        self._add_usage(answer.usage)
        return answer

    async def aask(
        self, document_id: str, question: str, *, strategy: Strategy | None = None
    ) -> Answer:
        """Async version of :meth:`ask`."""
        answer = await self._qa_async(document_id).ask(question, strategy=strategy)
        self._add_usage(answer.usage)
        return answer

    @property
    def usage(self) -> TokenUsage:
        """Cumulative token usage across every document."""
        return self._usage

    # -- internals -------------------------------------------------------------------

    def _add_usage(self, usage: TokenUsage) -> None:
        with self._lock:
            self._usage = self._usage + usage

    def _load(self, document_id: str) -> ChunkedDocument:
        validate_document_id(document_id)
        chunked = self.store.get(document_id)
        if chunked is None:
            raise DocumentNotFoundError(document_id)
        return chunked

    def _evict(self, document_id: str) -> None:
        with self._lock:
            self._sync_qa.pop(document_id, None)
            self._async_qa.pop(document_id, None)

    def _qa_sync(self, document_id: str) -> DocumentQA:
        with self._lock:
            qa = self._sync_qa.get(document_id)
            if qa is None:
                qa = DocumentQA(
                    self._load(document_id),
                    config=self.config,
                    retriever=self._retriever_factory(),
                    client=self.client,
                )
                self._sync_qa[document_id] = qa
                while len(self._sync_qa) > self._max_cached:
                    self._sync_qa.popitem(last=False)
            self._sync_qa.move_to_end(document_id)
            return qa

    def _qa_async(self, document_id: str) -> AsyncDocumentQA:
        with self._lock:
            qa = self._async_qa.get(document_id)
            if qa is None:
                qa = AsyncDocumentQA(
                    self._load(document_id),
                    config=self.config,
                    retriever=self._retriever_factory(),
                    client=self.async_client,
                )
                self._async_qa[document_id] = qa
                while len(self._async_qa) > self._max_cached:
                    self._async_qa.popitem(last=False)
            self._async_qa.move_to_end(document_id)
            return qa
