"""Immutable data types used across the library."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Page:
    """One page (or the whole body, for formats without pages) of a document.

    Attributes:
        text: The page's extracted text.
        number: 1-based page number, or ``None`` when the source has no pages.
    """

    text: str
    number: int | None = None


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded document: an ordered sequence of pages plus where it came from."""

    pages: tuple[Page, ...]
    source: str | None = None

    @classmethod
    def from_text(cls, text: str, *, source: str | None = None) -> Document:
        """Build a single-page document from a string."""
        return cls(pages=(Page(text=text),), source=source)

    @classmethod
    def from_pages(cls, pages: Iterable[str], *, source: str | None = None) -> Document:
        """Build a paged document from page texts, numbered from 1."""
        return cls(
            pages=tuple(Page(text=t, number=i) for i, t in enumerate(pages, start=1)),
            source=source,
        )

    @property
    def text(self) -> str:
        """The full document text, pages joined by blank lines."""
        return "\n\n".join(p.text for p in self.pages)


@dataclass(frozen=True, slots=True)
class ParentChunk:
    """A large, self-contained section of the document - the unit sent to the model."""

    id: str
    text: str
    order: int
    heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    @property
    def page_label(self) -> str | None:
        """Human-readable page range such as ``"12"`` or ``"12-14"``."""
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


@dataclass(frozen=True, slots=True)
class ChildChunk:
    """A small passage inside a parent - the unit that is searched."""

    id: str
    parent_id: str
    text: str
    order: int
    heading: str | None = None
    """Heading of the parent section. Search backends should index it with the text:
    headings such as "Governing Law" are often the strongest match for a question."""


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    """A document split into parents and their children."""

    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]
    source: str | None = None
    _by_id: dict[str, ParentChunk] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {p.id: p for p in self.parents})

    def parent(self, parent_id: str) -> ParentChunk:
        """Return the parent with this ID.

        Raises:
            KeyError: If no parent has this ID.
        """
        return self._by_id[parent_id]

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        """Return the parent with this ID, or ``None``."""
        return self._by_id.get(parent_id)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A child chunk that matched a query, with its relevance score."""

    child: ChildChunk
    score: float


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by the API. Supports ``+`` for running totals."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    @property
    def total_input_tokens(self) -> int:
        """All prompt tokens: uncached + written to cache + read from cache."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prompt tokens served from cache, between 0.0 and 1.0."""
        total = self.total_input_tokens
        return self.cache_read_input_tokens / total if total else 0.0


@dataclass(frozen=True, slots=True)
class Answer:
    """The result of :meth:`longdoc.DocumentQA.ask`.

    Attributes:
        text: The model's answer.
        citations: Sections the answer cites, in the order first cited.
        context: Every section that was sent to the model for this question.
        usage: Token usage for this call.
        model: The model that produced the answer (may differ from the requested one
            if a server-side fallback ran).
        stop_reason: The API's stop reason.
        truncated: ``True`` if the answer hit ``max_tokens`` and may be cut off.
        fallback_used: ``True`` if a fallback model produced the answer.
        request_id: The Anthropic request ID, for logging and support.
    """

    text: str
    citations: tuple[ParentChunk, ...]
    context: tuple[ParentChunk, ...]
    usage: TokenUsage
    model: str
    stop_reason: str | None
    truncated: bool = False
    fallback_used: bool = False
    request_id: str | None = None

    def __str__(self) -> str:
        return self.text
