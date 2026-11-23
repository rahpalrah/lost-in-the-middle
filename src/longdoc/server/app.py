"""FastAPI router and application factory."""

# No `from __future__ import annotations` here: FastAPI must evaluate the annotations of
# route functions at runtime to build request parsing and the OpenAPI schema.

import logging
import os
import secrets
import threading
import time
from collections import deque
from typing import Annotated, Any

import anthropic
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .._version import __version__
from ..config import Strategy
from ..exceptions import (
    ConfigurationError,
    DocumentLoadError,
    EmptyDocumentError,
    LongDocError,
    RefusalError,
    UnsupportedFormatError,
)
from ..serialization import answer_to_dict, section_to_dict
from ..service import DocumentNotFoundError, DocumentService

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
API_KEY_ENV = "LONGDOC_API_KEY"


# -- schemas --------------------------------------------------------------------------


class TextDocumentRequest(BaseModel):
    """Create a document from plain text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Full document text.")
    source: str | None = Field(default=None, max_length=512, description="Label, e.g. a file name.")
    id: str | None = Field(default=None, description="Your own document ID (1-64 chars).")


class AskRequest(BaseModel):
    """Ask a question about a document."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=8_000)
    strategy: Strategy | None = Field(
        default=None, description="Override the server's default strategy for this question."
    )
    include_context_text: bool = Field(
        default=False, description="Include the text of every section that was sent."
    )


class RetrieveRequest(BaseModel):
    """Preview which sections would be sent for a question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=8_000)


class DocumentResponse(BaseModel):
    """Summary of a stored document."""

    id: str
    source: str | None
    sections: int
    passages: int
    pages: int | None


class SectionResponse(BaseModel):
    """A document section."""

    id: str
    heading: str | None
    pages: str | None
    page_start: int | None
    page_end: int | None
    text: str | None = None


class UsageResponse(BaseModel):
    """Token usage for one call."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_input_tokens: int
    cache_hit_ratio: float


class AnswerResponse(BaseModel):
    """An answer with its citations."""

    text: str
    citations: list[SectionResponse]
    context: list[SectionResponse]
    usage: UsageResponse
    model: str
    stop_reason: str | None
    truncated: bool
    fallback_used: bool
    request_id: str | None


class ErrorResponse(BaseModel):
    """Error body returned for every non-2xx response."""

    error: str
    message: str
    details: dict[str, Any] | None = None


# -- error mapping --------------------------------------------------------------------


def _error(
    status: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    **details: Any,
) -> HTTPException:
    body = {"error": code, "message": message, "details": details or None}
    return HTTPException(status_code=status, detail=body, headers=headers)


class RateLimiter:
    """Sliding-window limiter: at most ``per_minute`` calls per caller in any 60 seconds.

    In-memory and per process, so with N workers the effective limit is up to N times
    higher. For a hard global limit, also enforce one at your gateway or load balancer.
    """

    def __init__(self, per_minute: int, *, window: float = 60.0) -> None:
        self.per_minute = per_minute
        self.window = window
        self._calls: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> int | None:
        """Record a call. Returns ``None`` if allowed, else seconds until it would be."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if len(self._calls) > 10_000:  # bound memory: drop callers with no recent calls
                self._calls = {
                    k: q for k, q in self._calls.items() if q and now - q[-1] < self.window
                }
            q = self._calls.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return None


def to_http_error(exc: Exception) -> HTTPException:
    """Translate library and Anthropic SDK errors into HTTP errors.

    Upstream error messages are logged, not returned, so internal details (such as
    credential problems) never leak to API callers.
    """
    if isinstance(exc, DocumentNotFoundError):
        return _error(404, "not_found", str(exc))
    if isinstance(exc, RefusalError):
        return _error(422, "refused", str(exc), category=exc.category, request_id=exc.request_id)
    if isinstance(exc, UnsupportedFormatError):
        return _error(415, "unsupported_format", str(exc))
    if isinstance(exc, (EmptyDocumentError, DocumentLoadError, ConfigurationError)):
        return _error(
            422,
            "invalid_document" if isinstance(exc, DocumentLoadError) else "invalid_request",
            str(exc),
        )
    if isinstance(exc, anthropic.RateLimitError):
        return _error(429, "rate_limited", "The model provider is rate limiting requests.")
    if isinstance(exc, anthropic.APIStatusError):
        logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        return _error(
            502,
            "upstream_error",
            "The model provider returned an error.",
            upstream_status=exc.status_code,
            request_id=exc.request_id,
        )
    if isinstance(exc, anthropic.APIConnectionError):
        logger.error("Could not reach the Anthropic API: %s", exc)
        return _error(503, "upstream_unavailable", "The model provider could not be reached.")
    if isinstance(exc, LongDocError):
        return _error(400, "bad_request", str(exc))
    raise exc


# -- router ---------------------------------------------------------------------------


def _auth_dependency(api_key: str | None) -> Any:
    bearer = HTTPBearer(auto_error=False)

    def check(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if api_key is None:
            return
        supplied = credentials.credentials if credentials else ""
        if not secrets.compare_digest(supplied.encode(), api_key.encode()):
            raise _error(401, "unauthorized", "Missing or invalid bearer token.")

    return check


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise _error(413, "too_large", f"Uploads are limited to {limit} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def create_router(
    service: DocumentService,
    *,
    api_key: str | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    rate_limit_per_minute: int | None = None,
) -> APIRouter:
    """Build an :class:`fastapi.APIRouter` exposing ``service`` under ``/v1``.

    Args:
        service: The service that stores and answers questions about documents.
        api_key: If set, every request must send ``Authorization: Bearer <api_key>``.
            Leave unset only when your own app already authenticates these routes.
        max_upload_bytes: Upper bound on uploaded file size.
        rate_limit_per_minute: If set, each caller (by client address) may make at most this
            many costly calls (uploads and questions) per minute; excess calls get HTTP 429
            with a ``Retry-After`` header. Questions spend model tokens, so set this for any
            API that untrusted or many users can reach. Behind a reverse proxy, make sure
            the proxy passes the real client address (for example with Uvicorn's
            ``--proxy-headers``), or every caller shares one bucket.
    """
    limiter = RateLimiter(rate_limit_per_minute) if rate_limit_per_minute else None

    def throttle(request: Request) -> None:
        if limiter is None:
            return
        wait = limiter.check(request.client.host if request.client else "unknown")
        if wait is not None:
            raise _error(
                429,
                "rate_limited",
                "Too many requests; slow down.",
                headers={"Retry-After": str(wait)},
                retry_after_seconds=wait,
            )

    router = APIRouter(
        prefix="/v1",
        tags=["documents"],
        dependencies=[Depends(_auth_dependency(api_key))],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )

    @router.post(
        "/documents",
        status_code=201,
        response_model=DocumentResponse,
        dependencies=[Depends(throttle)],
    )
    async def upload_document(
        file: Annotated[UploadFile, File(description=".pdf, .txt, .md or .rst")],
        id: Annotated[str | None, Form(description="Your own document ID")] = None,
    ) -> dict[str, Any]:
        """Upload a file and index it."""
        data = await _read_limited(file, max_upload_bytes)
        try:
            info = service.add_bytes(data, filename=file.filename or "upload", document_id=id)
        except Exception as exc:
            raise to_http_error(exc) from exc
        return info.to_dict()

    @router.post("/documents/text", status_code=201, response_model=DocumentResponse)
    def create_text_document(body: TextDocumentRequest) -> dict[str, Any]:
        """Index a document sent as plain text."""
        try:
            info = service.add_text(body.text, source=body.source, document_id=body.id)
        except Exception as exc:
            raise to_http_error(exc) from exc
        return info.to_dict()

    @router.get("/documents", response_model=list[DocumentResponse])
    def list_documents() -> list[dict[str, Any]]:
        """List stored documents."""
        return [info.to_dict() for info in service.list_documents()]

    @router.get("/documents/{document_id}", response_model=DocumentResponse)
    def get_document(document_id: str) -> dict[str, Any]:
        """Get one document's summary."""
        try:
            return service.get_info(document_id).to_dict()
        except Exception as exc:
            raise to_http_error(exc) from exc

    @router.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: str) -> None:
        """Delete a document."""
        try:
            found = service.delete(document_id)
        except Exception as exc:
            raise to_http_error(exc) from exc
        if not found:
            raise to_http_error(DocumentNotFoundError(document_id))

    @router.post(
        "/documents/{document_id}/ask",
        response_model=AnswerResponse,
        dependencies=[Depends(throttle)],
    )
    async def ask(document_id: str, body: AskRequest) -> dict[str, Any]:
        """Answer a question about a document, with citations."""
        try:
            answer = await service.aask(document_id, body.question, strategy=body.strategy)
        except Exception as exc:
            raise to_http_error(exc) from exc
        return answer_to_dict(answer, include_context_text=body.include_context_text)

    @router.post("/documents/{document_id}/retrieve", response_model=list[SectionResponse])
    def retrieve(document_id: str, body: RetrieveRequest) -> list[dict[str, Any]]:
        """Preview the sections retrieval would send for a question (no model call)."""
        try:
            sections = service.retrieve(document_id, body.question)
        except Exception as exc:
            raise to_http_error(exc) from exc
        return [section_to_dict(s, include_text=True) for s in sections]

    return router


def create_app(
    service: DocumentService | None = None,
    *,
    api_key: str | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    rate_limit_per_minute: int | None = None,
) -> FastAPI:
    """Build a standalone FastAPI application.

    Args:
        service: Defaults to a :class:`DocumentService` with an in-memory store.
        api_key: Bearer token callers must send. Defaults to the ``LONGDOC_API_KEY``
            environment variable; if neither is set, the API is unauthenticated and a
            warning is logged.
        max_upload_bytes: Upper bound on uploaded file size.
        rate_limit_per_minute: Maximum uploads plus questions per client address per
            minute, or ``None`` for no limit. See :func:`create_router`.
    """
    api_key = api_key or os.environ.get(API_KEY_ENV) or None
    if api_key is None:
        logger.warning(
            "longdoc API is running WITHOUT authentication. Set %s for any shared deployment.",
            API_KEY_ENV,
        )
    service = service or DocumentService()

    app = FastAPI(
        title="longdoc API",
        version=__version__,
        description="Question answering over long documents with Claude: parent-child "
        "retrieval, prompt caching and cited answers.",
        license_info={"name": "MIT", "identifier": "MIT"},
    )
    app.state.service = service
    app.include_router(
        create_router(
            service,
            api_key=api_key,
            max_upload_bytes=max_upload_bytes,
            rate_limit_per_minute=rate_limit_per_minute,
        )
    )

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail: Any = exc.detail  # we put structured dicts here; others are strings
        body = (
            detail
            if isinstance(detail, dict)
            else {"error": "http_error", "message": str(detail), "details": None}
        )
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    return app
