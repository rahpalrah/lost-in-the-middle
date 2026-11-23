"""A DocumentStore backed by SQLite (standard library only).

Shows how to keep chunked documents in your own database. The same four methods work
for Postgres (a JSONB column), MySQL, DynamoDB, Redis or S3: store
``chunked_to_dict(doc)`` as JSON and rebuild it with ``chunked_from_dict``.

    python examples/sqlite_store.py
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator

from longdoc import (
    ChunkedDocument,
    DocumentService,
    chunked_from_dict,
    chunked_to_dict,
)
from longdoc.store import validate_document_id


class SQLiteStore:
    """Thread-safe DocumentStore in a single SQLite file (or ``:memory:``)."""

    def __init__(self, path: str = "longdoc.db") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )

    def get(self, document_id: str) -> ChunkedDocument | None:
        with self._lock:
            row = self._db.execute(
                "SELECT body FROM documents WHERE id = ?", (validate_document_id(document_id),)
            ).fetchone()
        return chunked_from_dict(json.loads(row[0])) if row else None

    def put(self, document_id: str, document: ChunkedDocument) -> None:
        body = json.dumps(chunked_to_dict(document), ensure_ascii=False)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO documents (id, body) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET body = excluded.body",
                (validate_document_id(document_id), body),
            )

    def delete(self, document_id: str) -> bool:
        with self._lock, self._db:
            cur = self._db.execute(
                "DELETE FROM documents WHERE id = ?", (validate_document_id(document_id),)
            )
        return cur.rowcount > 0

    def list_ids(self) -> Iterator[str]:
        with self._lock:
            rows = self._db.execute("SELECT id FROM documents ORDER BY id").fetchall()
        yield from (r[0] for r in rows)


if __name__ == "__main__":
    service = DocumentService(store=SQLiteStore(":memory:"))
    info = service.add_text(
        "# 7. Payment\n\nInvoices are payable within 30 days.\n\n"
        "# 19. Termination\n\nEither party may terminate on 90 days notice.",
        document_id="demo",
    )
    print("stored:", info)
    print(
        "retrieved sections:", [s.heading for s in service.retrieve("demo", "termination notice")]
    )
