"""JSON-safe conversion of library objects.

Use these to persist chunked documents (so a document is parsed and chunked once, not
on every process start) and to return answers from your own API.

The persisted format carries a ``schema_version``; :func:`chunked_from_dict` rejects
versions it does not understand instead of misreading them.
"""

from __future__ import annotations

from typing import Any

from .exceptions import DocumentLoadError
from .models import Answer, ChildChunk, ChunkedDocument, ParentChunk, TokenUsage

SCHEMA_VERSION = 1


def section_to_dict(section: ParentChunk, *, include_text: bool = False) -> dict[str, Any]:
    """Describe a parent section. Text is left out by default to keep payloads small."""
    data: dict[str, Any] = {
        "id": section.id,
        "heading": section.heading,
        "pages": section.page_label,
        "page_start": section.page_start,
        "page_end": section.page_end,
    }
    if include_text:
        data["text"] = section.text
    return data


def usage_to_dict(usage: TokenUsage) -> dict[str, Any]:
    """Token usage, including the derived total and cache hit ratio."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "total_input_tokens": usage.total_input_tokens,
        "cache_hit_ratio": round(usage.cache_hit_ratio, 4),
    }


def answer_to_dict(answer: Answer, *, include_context_text: bool = False) -> dict[str, Any]:
    """A JSON-ready view of an :class:`Answer`, suitable as an API response body."""
    return {
        "text": answer.text,
        "citations": [section_to_dict(s) for s in answer.citations],
        "context": [section_to_dict(s, include_text=include_context_text) for s in answer.context],
        "usage": usage_to_dict(answer.usage),
        "model": answer.model,
        "stop_reason": answer.stop_reason,
        "truncated": answer.truncated,
        "fallback_used": answer.fallback_used,
        "request_id": answer.request_id,
    }


def chunked_to_dict(document: ChunkedDocument) -> dict[str, Any]:
    """Serialize a chunked document for storage."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": document.source,
        "parents": [
            {
                "id": p.id,
                "text": p.text,
                "order": p.order,
                "heading": p.heading,
                "page_start": p.page_start,
                "page_end": p.page_end,
            }
            for p in document.parents
        ],
        "children": [
            {
                "id": c.id,
                "parent_id": c.parent_id,
                "text": c.text,
                "order": c.order,
                "heading": c.heading,
            }
            for c in document.children
        ],
    }


def chunked_from_dict(data: dict[str, Any]) -> ChunkedDocument:
    """Rebuild a chunked document from :func:`chunked_to_dict` output.

    Raises:
        DocumentLoadError: The data is malformed or from an unsupported schema version.
    """
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DocumentLoadError(
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION}."
        )
    try:
        return ChunkedDocument(
            parents=tuple(ParentChunk(**p) for p in data["parents"]),
            children=tuple(ChildChunk(**c) for c in data["children"]),
            source=data.get("source"),
        )
    except (KeyError, TypeError) as exc:
        raise DocumentLoadError(f"Malformed chunked document: {exc}") from exc
