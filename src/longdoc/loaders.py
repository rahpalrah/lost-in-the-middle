"""Load documents from disk or from bytes (for example, an HTTP upload).

Supported: ``.txt``, ``.md``/``.markdown``/``.rst`` and ``.pdf``. PDFs are read by a built-in
standard-library reader (see :mod:`longdoc.pdf`); if ``pypdf`` is installed
(``pip install "longdoc-qa[pdf]"``) it is used instead, being more robust on unusual files.
For anything else, extract the text yourself and use :meth:`Document.from_text` or
:meth:`Document.from_pages`.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
from pathlib import Path, PurePath
from typing import Literal

from .exceptions import DocumentLoadError, EmptyDocumentError, UnsupportedFormatError
from .models import Document
from .pdf import extract_pages

PdfBackend = Literal["auto", "builtin", "pypdf"]

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def load_document(
    path: str | os.PathLike[str], *, encoding: str = "utf-8", pdf_backend: PdfBackend = "auto"
) -> Document:
    """Load a document from disk, choosing the reader by file extension.

    Raises:
        DocumentLoadError: The file is missing or unreadable.
        UnsupportedFormatError: The extension is unsupported, or the PDF is encrypted.
        EmptyDocumentError: No text could be extracted.
    """
    p = Path(path)
    if not p.is_file():
        raise DocumentLoadError(f"File not found: {p}")
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read {p}: {exc}") from exc
    return load_bytes(data, filename=str(p), encoding=encoding, pdf_backend=pdf_backend)


def load_bytes(
    data: bytes, *, filename: str, encoding: str = "utf-8", pdf_backend: PdfBackend = "auto"
) -> Document:
    """Load a document from raw bytes. ``filename`` selects the reader by extension.

    Raises:
        DocumentLoadError: The content cannot be decoded or parsed.
        UnsupportedFormatError: The extension is unsupported, or the PDF is encrypted.
        EmptyDocumentError: No text could be extracted.
    """
    suffix = PurePath(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        try:
            doc = Document.from_text(data.decode(encoding), source=filename)
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(f"{filename} is not valid {encoding} text: {exc}") from exc
    elif suffix == ".pdf":
        doc = _load_pdf(data, filename, pdf_backend)
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


def _pages_with_pypdf(data: bytes, filename: str) -> list[str]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises many unrelated exception types
        raise DocumentLoadError(f"Could not parse PDF {filename}: {exc}") from exc


def _pages_with_builtin(data: bytes, filename: str) -> list[str]:
    try:
        return extract_pages(data)
    except UnsupportedFormatError:
        raise
    except DocumentLoadError as exc:
        raise DocumentLoadError(f"{exc} ({filename})") from exc


def _has_pypdf() -> bool:
    return importlib.util.find_spec("pypdf") is not None


def _load_pdf(data: bytes, filename: str, backend: PdfBackend = "auto") -> Document:
    """Read a PDF with the built-in reader, ``pypdf``, or both.

    ``"auto"`` tries ``pypdf`` first when it is installed (more robust on unusual files) and then
    the built-in standard-library reader, using the first that finds text. Each covers cases the
    other rejects: ``pypdf`` is strict about file structure, while the built-in reader is lenient
    about it. ``"builtin"`` and ``"pypdf"`` force one reader.
    """
    if backend not in ("auto", "builtin", "pypdf"):
        raise UnsupportedFormatError(f"Unknown pdf_backend {backend!r}.")
    if backend in ("auto", "pypdf") and not _has_pypdf():
        if backend == "pypdf":
            raise UnsupportedFormatError('pdf_backend="pypdf" needs: pip install "longdoc-qa[pdf]"')
        readers = [_pages_with_builtin]
    elif backend == "pypdf":
        readers = [_pages_with_pypdf]
    elif backend == "builtin":
        readers = [_pages_with_builtin]
    else:
        readers = [_pages_with_pypdf, _pages_with_builtin]

    first_error: DocumentLoadError | None = None
    pages: list[str] = []
    for reader in readers:
        try:
            candidate = reader(data, filename)
        except DocumentLoadError as exc:  # includes encrypted files (UnsupportedFormatError)
            first_error = first_error or exc
            continue
        if any(p.strip() for p in candidate):
            pages = candidate
            break
        pages = pages or candidate  # remember an empty result in case nothing finds text
    else:
        if first_error is not None and not pages:
            raise first_error
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
