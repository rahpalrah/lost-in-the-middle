"""Load documents from disk or from bytes (for example, an HTTP upload).

Supported: ``.txt``, ``.md``/``.markdown``/``.rst`` and ``.pdf`` (requires the ``pdf``
extra: ``pip install "longdoc-qa[pdf]"``). For anything else, extract the text
yourself and use :meth:`Document.from_text` or :meth:`Document.from_pages`.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path, PurePath
from typing import BinaryIO

from .exceptions import DocumentLoadError, EmptyDocumentError, UnsupportedFormatError
from .models import Document

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def load_document(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> Document:
    """Load a document from disk, choosing the reader by file extension.

    Raises:
        DocumentLoadError: The file is missing or unreadable.
        UnsupportedFormatError: The extension is unsupported, or ``pypdf`` is missing.
        EmptyDocumentError: No text could be extracted.
    """
    p = Path(path)
    if not p.is_file():
        raise DocumentLoadError(f"File not found: {p}")
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read {p}: {exc}") from exc
    return load_bytes(data, filename=str(p), encoding=encoding)


def load_bytes(data: bytes, *, filename: str, encoding: str = "utf-8") -> Document:
    """Load a document from raw bytes. ``filename`` selects the reader by extension.

    Raises:
        DocumentLoadError: The content cannot be decoded or parsed.
        UnsupportedFormatError: The extension is unsupported, or ``pypdf`` is missing.
        EmptyDocumentError: No text could be extracted.
    """
    suffix = PurePath(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        try:
            doc = Document.from_text(data.decode(encoding), source=filename)
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(f"{filename} is not valid {encoding} text: {exc}") from exc
    elif suffix == ".pdf":
        doc = _load_pdf(io.BytesIO(data), filename)
    else:
        raise UnsupportedFormatError(
            f"Unsupported file type {suffix or '(none)'!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )
    if not doc.text.strip():
        raise EmptyDocumentError(
            f"No text found in {filename}. Scanned PDFs need OCR before they can be loaded."
        )
    return doc


def _load_pdf(stream: BinaryIO, filename: str) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise UnsupportedFormatError(
            'PDF support needs pypdf: pip install "longdoc-qa[pdf]"'
        ) from exc
    try:
        reader = PdfReader(stream)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises many unrelated exception types
        raise DocumentLoadError(f"Could not parse PDF {filename}: {exc}") from exc
    return Document.from_pages(strip_running_headers(pages), source=filename)


_PAGE_NUMBER = re.compile(
    r"^(?:page\s*)?[-\u2013(\[]?\s*\d{1,4}\s*(?:(?:of|/)\s*\d{1,4})?\s*[-\u2013)\]]?$", re.I
)


def strip_running_headers(pages: list[str], *, threshold: float = 0.5) -> list[str]:
    """Remove running headers, footers and page numbers from extracted PDF pages.

    Only the first and last two lines of each page are considered. A line is dropped
    when it is a page number ("7", "Page 7", "7 of 30", "- 7 -"), or when exactly the
    same line appears at the edge of at least ``threshold`` of all pages. Lines that
    merely differ by a number are kept, so real content is never removed. Documents
    with fewer than 3 pages are returned unchanged.
    """
    if len(pages) < 3:
        return pages
    edge = 2

    def norm(line: str) -> str:
        return " ".join(line.split())

    def edges(lines: list[str]) -> list[str]:
        return lines[:edge] + lines[-edge:] if len(lines) > edge else lines

    page_lines = [[ln for ln in text.splitlines() if ln.strip()] for text in pages]
    counts: dict[str, int] = {}
    for lines in page_lines:
        for k in {norm(ln) for ln in edges(lines)}:
            counts[k] = counts.get(k, 0) + 1
    repeated = {k for k, n in counts.items() if n >= threshold * len(pages)}

    def drop(line: str) -> bool:
        key = norm(line)
        return key in repeated or bool(_PAGE_NUMBER.match(key))

    cleaned = []
    for lines in page_lines:
        n = len(lines)
        kept = [ln for i, ln in enumerate(lines) if not ((i < edge or i >= n - edge) and drop(ln))]
        cleaned.append("\n".join(kept))
    return cleaned
