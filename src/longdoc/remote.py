"""A Python client for the longdoc HTTP API, using only the standard library.

Use it from another Python service that talks to a running ``longdoc serve``:

.. code-block:: python

    from longdoc.remote import RemoteClient

    client = RemoteClient("http://localhost:8000", api_key="...")   # key from LONGDOC_API_KEY
    client.upload("contract.pdf", document_id="contract-1042")
    answer = client.ask("contract-1042", "What is the notice period?")
    print(answer["text"], [c["heading"] for c in answer["citations"]])

Every method returns the API's JSON as plain ``dict``/``list`` values and raises
:class:`RemoteError` (with the status, error code and message) when the server reports an error.
Transient failures (connection errors, 502/503/504) are retried with backoff.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._version import __version__
from .exceptions import ConfigurationError, LongDocError
from .http import _OPENER, Secret, check_url


class RemoteError(LongDocError):
    """The longdoc server answered with an error (or could not be reached).

    Attributes:
        status: HTTP status, or ``None`` if the server could not be reached.
        code: The API's error code (for example ``not_found`` or ``rate_limited``).
        details: The API's ``details`` object, if any.
        retry_after: Seconds the server asked you to wait, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = dict(details) if details else None
        self.retry_after = retry_after


def encode_multipart(
    fields: Mapping[str, str], files: Mapping[str, tuple[str, bytes]]
) -> tuple[bytes, str]:
    """Encode a ``multipart/form-data`` body. Returns ``(body, content_type)``."""
    boundary = "longdoc-" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode("utf-8")
            + b"\r\n"
        )
    for name, (filename, content) in files.items():
        safe = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{safe}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
            + content
            + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class RemoteClient:
    """Client for a running longdoc HTTP API.

    Args:
        base_url: For example ``"http://localhost:8000"``.
        api_key: The server's bearer token. Defaults to the ``LONGDOC_API_KEY`` variable.
        timeout: Seconds to wait for each response (answers from a language model can be slow).
        max_retries: Retries for connection errors and 502/503/504 responses.
        allow_insecure: Permit plain ``http://`` to a non-local host (use https in production).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        allow_insecure: bool = False,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("LONGDOC_API_KEY")
        self.base_url = base_url.rstrip("/")
        check_url(self.base_url, allow_insecure=allow_insecure)
        self._key = Secret(key) if key else None
        self.timeout = timeout
        self.max_retries = max_retries

    def __repr__(self) -> str:  # never include credentials
        return f"RemoteClient({self.base_url!r})"

    # -- the API ---------------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Liveness check."""
        return self._request("GET", "/health")  # type: ignore[no-any-return]

    def add_text(
        self, text: str, *, source: str | None = None, document_id: str | None = None
    ) -> dict[str, Any]:
        """Index a document sent as text."""
        body: dict[str, Any] = {"text": text}
        if source is not None:
            body["source"] = source
        if document_id is not None:
            body["id"] = document_id
        return self._request("POST", "/v1/documents/text", json_body=body)  # type: ignore[no-any-return]

    def upload(
        self,
        file: str | os.PathLike[str] | bytes,
        *,
        filename: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a file (a path, or the bytes plus ``filename``) and index it."""
        if isinstance(file, bytes):
            if not filename:
                raise ConfigurationError("filename= is required when uploading bytes.")
            content, name = file, filename
        else:
            path = Path(file)
            content, name = path.read_bytes(), filename or path.name
        fields = {"id": document_id} if document_id else {}
        body, content_type = encode_multipart(fields, {"file": (name, content)})
        return self._request("POST", "/v1/documents", raw_body=body, content_type=content_type)  # type: ignore[no-any-return]

    def list_documents(self) -> list[dict[str, Any]]:
        """Summaries of every stored document."""
        return self._request("GET", "/v1/documents")  # type: ignore[no-any-return]

    def get_document(self, document_id: str) -> dict[str, Any]:
        """One document's summary."""
        return self._request("GET", f"/v1/documents/{_quote(document_id)}")  # type: ignore[no-any-return]

    def delete_document(self, document_id: str) -> None:
        """Delete a document."""
        self._request("DELETE", f"/v1/documents/{_quote(document_id)}")

    def ask(
        self,
        document_id: str,
        question: str,
        *,
        strategy: str | None = None,
        include_context_text: bool = False,
    ) -> dict[str, Any]:
        """Ask a question; returns the answer with citations and token usage."""
        body: dict[str, Any] = {"question": question, "include_context_text": include_context_text}
        if strategy is not None:
            body["strategy"] = strategy
        return self._request("POST", f"/v1/documents/{_quote(document_id)}/ask", json_body=body)  # type: ignore[no-any-return]

    def retrieve(self, document_id: str, question: str) -> list[dict[str, Any]]:
        """Preview the sections that would be sent for ``question`` (no model call)."""
        return self._request(  # type: ignore[no-any-return]
            "POST",
            f"/v1/documents/{_quote(document_id)}/retrieve",
            json_body={"question": question},
        )

    # -- transport -------------------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        raw_body: bytes | None = None,
        content_type: str | None = None,
    ) -> Any:
        data = raw_body
        headers = {"accept": "application/json", "user-agent": f"longdoc-qa/{__version__}"}
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"
        if content_type:
            headers["content-type"] = content_type
        if self._key is not None:
            headers["authorization"] = f"Bearer {self._key.reveal()}"

        attempt = 0
        while True:
            req = urllib.request.Request(
                self.base_url + path, data=data, headers=headers, method=method
            )
            try:
                with _OPENER.open(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                error = _to_error(exc)
                transient = exc.code in (502, 503, 504)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                error = RemoteError(f"Could not reach the server ({type(reason).__name__}).")
                transient = True
            if not transient or attempt >= self.max_retries:
                raise error
            delay = error.retry_after if error.retry_after is not None else 0.5 * (2**attempt)
            time.sleep(min(delay, 30.0) + random.uniform(0, 0.25))
            attempt += 1


def _quote(document_id: str) -> str:
    return urllib.parse.quote(document_id, safe="")


def _to_error(exc: urllib.error.HTTPError) -> RemoteError:
    try:
        body = json.loads(exc.read(1024 * 1024))
    except (ValueError, UnicodeDecodeError):
        body = {}
    retry_after = None
    try:
        if exc.headers and exc.headers.get("retry-after"):
            retry_after = float(exc.headers["retry-after"])
    except ValueError:
        retry_after = None
    code = body.get("error") if isinstance(body, dict) else None
    message = body.get("message") if isinstance(body, dict) else None
    return RemoteError(
        f"{exc.code} {code or 'error'}: {message or 'no details'}",
        status=exc.code,
        code=code if isinstance(code, str) else None,
        details=body.get("details") if isinstance(body, dict) else None,
        retry_after=retry_after,
    )
