"""Where chunked documents live between requests.

:class:`DocumentStore` is a :class:`typing.Protocol`: back it with your own database,
object storage or cache by implementing four methods. Two implementations ship with
the library:

* :class:`InMemoryStore` - process-local; for tests and single-process apps.
* :class:`JSONFileStore` - one JSON file per document in a directory; survives
  restarts and can be shared between processes on one machine or a shared volume.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from .exceptions import ConfigurationError, DocumentLoadError
from .models import ChunkedDocument
from .serialization import chunked_from_dict, chunked_to_dict

DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_document_id(document_id: str) -> str:
    """Return ``document_id`` if it is 1-64 letters, digits, ``_`` or ``-``.

    The restriction keeps IDs safe to use in URLs and as file names.

    Raises:
        ConfigurationError: The ID is invalid.
    """
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise ConfigurationError(
            "Document IDs must be 1-64 characters of letters, digits, '_' or '-', "
            "starting with a letter or digit."
        )
    return document_id


@runtime_checkable
class DocumentStore(Protocol):
    """Persistent storage for chunked documents, keyed by document ID."""

    def get(self, document_id: str) -> ChunkedDocument | None:
        """Return the document, or ``None`` if it does not exist."""
        ...

    def put(self, document_id: str, document: ChunkedDocument) -> None:
        """Create or replace the document."""
        ...

    def delete(self, document_id: str) -> bool:
        """Delete the document; return ``True`` if it existed."""
        ...

    def list_ids(self) -> Iterator[str]:
        """Yield every stored document ID."""
        ...


class InMemoryStore:
    """Thread-safe, process-local document store."""

    def __init__(self) -> None:
        self._docs: dict[str, ChunkedDocument] = {}
        self._lock = threading.Lock()

    def get(self, document_id: str) -> ChunkedDocument | None:
        """Return the document, or ``None``."""
        with self._lock:
            return self._docs.get(document_id)

    def put(self, document_id: str, document: ChunkedDocument) -> None:
        """Create or replace the document."""
        with self._lock:
            self._docs[document_id] = document

    def delete(self, document_id: str) -> bool:
        """Delete the document; return ``True`` if it existed."""
        with self._lock:
            return self._docs.pop(document_id, None) is not None

    def list_ids(self) -> Iterator[str]:
        """Yield stored IDs in sorted order."""
        with self._lock:
            ids = sorted(self._docs)
        yield from ids


class JSONFileStore:
    """Stores each document as ``<directory>/<document_id>.json``.

    Writes are atomic (write to a temporary file, then rename), so a crash never
    leaves a half-written document behind.

    Args:
        directory: Created if it does not exist.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id: str) -> Path:
        return self.directory / f"{validate_document_id(document_id)}.json"

    def get(self, document_id: str) -> ChunkedDocument | None:
        """Return the document, or ``None``.

        Raises:
            DocumentLoadError: The file exists but is corrupt.
        """
        path = self._path(document_id)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return chunked_from_dict(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(f"Corrupt document file {path}: {exc}") from exc

    def put(self, document_id: str, document: ChunkedDocument) -> None:
        """Atomically create or replace the document."""
        path = self._path(document_id)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=f".{document_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(chunked_to_dict(document), fh, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def delete(self, document_id: str) -> bool:
        """Delete the document; return ``True`` if it existed."""
        try:
            self._path(document_id).unlink()
        except FileNotFoundError:
            return False
        return True

    def list_ids(self) -> Iterator[str]:
        """Yield stored IDs in sorted order."""
        for path in sorted(self.directory.glob("*.json")):
            if DOCUMENT_ID_PATTERN.fullmatch(path.stem):
                yield path.stem
