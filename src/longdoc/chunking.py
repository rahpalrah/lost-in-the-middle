"""Parent-child chunking.

Long documents are split twice:

* **Parents** are large sections, cut at headings and capped at
  :attr:`ChunkingConfig.parent_max_chars`. They are what the model reads, so each one
  keeps enough surrounding context to be understood on its own.
* **Children** are small, overlapping passages inside each parent. They are what the
  retriever searches, because small passages match a query far more precisely than
  whole sections do.

At question time the retriever finds the best children and the model is given their
*parents*. This keeps prompts small (avoiding "lost in the middle" accuracy loss on
huge prompts) without losing the context around the matching lines.

Chunk IDs are deterministic (``p0003``, ``p0003-c002``) so the same document always
produces byte-identical prompts, which prompt caching requires.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .config import ChunkingConfig
from .exceptions import EmptyDocumentError
from .models import ChildChunk, ChunkedDocument, Document, ParentChunk

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")
_NUMBERED_HEADING = re.compile(r"^(?:(?:[A-Z]\.)?\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+[A-Z]\S")
_KEYWORD_HEADING = re.compile(r"^(?:section|article|chapter|part|schedule|annex|appendix)\b", re.I)
_MAX_HEADING_CHARS = 120


@dataclass(slots=True)
class _Paragraph:
    text: str
    page: int | None
    is_heading: bool = False


def is_heading(line: str) -> bool:
    """Return ``True`` if a single line looks like a section heading.

    Recognises Markdown headings, numbered headings (``3.2 Termination``), keyword
    headings (``Section 9``, ``ARTICLE IV``) and short ALL-CAPS lines.
    """
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_CHARS or "\n" in line:
        return False
    if _MARKDOWN_HEADING.match(line):
        return True
    if line.endswith((",", ";", ":")):
        return False
    words = line.split()
    if _NUMBERED_HEADING.match(line) or _KEYWORD_HEADING.match(line):
        # Title-like: most words capitalised. This rejects wrapped body lines that
        # happen to start with a number or "Article 12", which PDFs produce often.
        return len(words) <= 12 and not line.endswith(".") and _is_title_like(words[1:])
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters) and len(words) <= 10


_MINOR_WORDS = frozenset(
    {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
        "of", "on", "or", "the", "to", "with", "without",
    }
)  # fmt: skip


def _is_title_like(words: list[str]) -> bool:
    significant = [w for w in words if w.lower() not in _MINOR_WORDS]
    if not significant:
        return True
    capitalised = sum(1 for w in significant if w[0].isupper() or w[0].isdigit())
    return capitalised / len(significant) >= 0.6


class ParentChildChunker:
    r"""Split a :class:`Document` into parent sections and child passages.

    Args:
        config: Chunk sizes. Defaults to :class:`ChunkingConfig` defaults.

    Example:
        >>> from longdoc import Document, ParentChildChunker
        >>> doc = Document.from_text("# Intro\\n\\nHello world.\\n\\n# Terms\\n\\nPay in 30 days.")
        >>> chunked = ParentChildChunker().chunk(doc)
        >>> [p.heading for p in chunked.parents]
        ['# Intro', '# Terms']
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> ChunkedDocument:
        """Chunk ``document``.

        Raises:
            EmptyDocumentError: If the document has no non-whitespace text.
        """
        paragraphs = list(self._paragraphs(document))
        if not paragraphs:
            raise EmptyDocumentError(f"No text found in {document.source or 'document'}.")

        parents = self._build_parents(paragraphs)
        children = tuple(child for parent in parents for child in self._build_children(parent))
        return ChunkedDocument(parents=parents, children=children, source=document.source)

    # -- parents -------------------------------------------------------------------

    def _paragraphs(self, document: Document) -> Iterator[_Paragraph]:
        limit = self.config.parent_max_chars
        for page in document.pages:
            for raw in _PARAGRAPH_SPLIT.split(page.text):
                text = _normalise(raw)
                if not text:
                    continue
                # Headings can sit on any line: PDF extraction rarely leaves blank lines
                # between a heading and its body, or between paragraphs.
                body: list[str] = []
                for line in text.split("\n"):
                    if is_heading(line):
                        if body:
                            yield from self._sized(" ".join(body), page.number, limit)
                            body = []
                        yield _Paragraph(line, page.number, is_heading=True)
                    else:
                        body.append(line)
                if body:
                    yield from self._sized("\n".join(body), page.number, limit)

    @staticmethod
    def _sized(text: str, page: int | None, limit: int) -> Iterator[_Paragraph]:
        for piece in _split_to_size(text, limit):
            yield _Paragraph(piece, page)

    def _build_parents(self, paragraphs: list[_Paragraph]) -> tuple[ParentChunk, ...]:
        parents: list[ParentChunk] = []
        buf: list[_Paragraph] = []
        heading: str | None = None
        size = 0

        def flush() -> None:
            nonlocal buf, size
            if buf:
                pages = [p.page for p in buf if p.page is not None]
                parents.append(
                    ParentChunk(
                        id=f"p{len(parents):04d}",
                        text="\n\n".join(p.text for p in buf),
                        order=len(parents),
                        heading=heading,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                    )
                )
            buf, size = [], 0

        for para in paragraphs:
            if para.is_heading or is_heading(para.text):
                if buf and all(p.is_heading for p in buf):
                    # "ARTICLE 19. TERMINATION" directly followed by "19.1 For Cause":
                    # one section, both headings kept.
                    heading = f"{heading} > {para.text}" if heading else para.text
                else:
                    flush()
                    heading = para.text
                para.is_heading = True
            elif size + len(para.text) > self.config.parent_max_chars:
                flush()
            buf.append(para)
            size += len(para.text) + 2
        flush()
        return tuple(parents)

    # -- children ------------------------------------------------------------------

    def _build_children(self, parent: ParentChunk) -> Iterator[ChildChunk]:
        max_chars = self.config.child_max_chars
        overlap = self.config.child_overlap_chars
        units = [
            piece
            for para in parent.text.split("\n\n")
            for sentence in _SENTENCE_SPLIT.split(para)
            for piece in _split_to_size(sentence.strip(), max_chars)
            if piece
        ]

        start = 0
        order = 0
        while start < len(units):
            end, size = start, 0
            while end < len(units) and (end == start or size + len(units[end]) + 1 <= max_chars):
                size += len(units[end]) + 1
                end += 1
            yield ChildChunk(
                id=f"{parent.id}-c{order:03d}",
                parent_id=parent.id,
                text=" ".join(units[start:end]),
                order=order,
                heading=parent.heading,
            )
            order += 1
            if end >= len(units):
                break
            # Step back so the next child repeats ~`overlap` characters of this one.
            back, carried = end, 0
            while back - 1 > start and carried + len(units[back - 1]) <= overlap:
                back -= 1
                carried += len(units[back]) + 1
            start = back


def _normalise(text: str) -> str:
    lines = (re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _split_to_size(text: str, limit: int) -> Iterator[str]:
    """Yield ``text`` in pieces of at most ``limit`` chars, preferring word boundaries."""
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit + 1)
        if cut <= limit // 2:
            cut = limit
        yield text[:cut].strip()
        text = text[cut:].strip()
    if text:
        yield text
