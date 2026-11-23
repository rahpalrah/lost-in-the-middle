"""longdoc: accurate, low-cost question answering over long documents with Claude.

Two techniques, usable separately or together:

* **Parent-child retrieval** - search small passages, send the model only the full
  sections they belong to (:attr:`Strategy.RETRIEVAL`, the default).
* **Prompt caching** - send the whole document once as a cached prefix and reuse it
  across questions (:attr:`Strategy.CACHED_DOCUMENT`).

Quick start::

    from longdoc import DocumentQA

    qa = DocumentQA.from_file("contract.pdf")
    answer = qa.ask("What is the termination notice period?")
    print(answer.text)

To integrate into an existing system, use :class:`DocumentService` (many documents,
pluggable storage), mount :func:`longdoc.server.create_router` in a FastAPI app, or
run the standalone HTTP API with ``longdoc serve``.
"""

import logging

from ._version import __version__
from .chunking import ParentChildChunker
from .client import AsyncDocumentQA, DocumentQA
from .config import DEFAULT_MODEL, ChunkingConfig, QAConfig, Strategy
from .exceptions import (
    ConfigurationError,
    DocumentLoadError,
    EmptyDocumentError,
    LongDocError,
    RefusalError,
    UnsupportedFormatError,
)
from .loaders import load_bytes, load_document
from .models import (
    Answer,
    ChildChunk,
    ChunkedDocument,
    Document,
    Page,
    ParentChunk,
    RetrievalHit,
    TokenUsage,
)
from .retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever, Retriever
from .serialization import answer_to_dict, chunked_from_dict, chunked_to_dict
from .service import DocumentInfo, DocumentNotFoundError, DocumentService
from .store import DocumentStore, InMemoryStore, JSONFileStore

# Library best practice: never configure logging for the application.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DEFAULT_MODEL",
    "Answer",
    "AsyncDocumentQA",
    "BM25Retriever",
    "ChildChunk",
    "ChunkedDocument",
    "ChunkingConfig",
    "ConfigurationError",
    "Document",
    "DocumentInfo",
    "DocumentLoadError",
    "DocumentNotFoundError",
    "DocumentQA",
    "DocumentService",
    "DocumentStore",
    "EmbeddingRetriever",
    "EmptyDocumentError",
    "HybridRetriever",
    "InMemoryStore",
    "JSONFileStore",
    "LongDocError",
    "Page",
    "ParentChildChunker",
    "ParentChunk",
    "QAConfig",
    "RefusalError",
    "RetrievalHit",
    "Retriever",
    "Strategy",
    "TokenUsage",
    "UnsupportedFormatError",
    "__version__",
    "answer_to_dict",
    "chunked_from_dict",
    "chunked_to_dict",
    "load_bytes",
    "load_document",
]
