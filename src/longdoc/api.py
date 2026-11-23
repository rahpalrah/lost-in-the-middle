"""The HTTP API, independent of any web framework.

:class:`LongDocApi` takes a request (method, path, headers, body) and returns a response. It does
routing, authentication, rate limiting, input validation, JSON handling and error mapping, so a
web server only has to move bytes. The standard-library server in
:mod:`longdoc.server.stdlib` uses it, and so can any other server or test harness.

Endpoints (all JSON unless noted; ``/v1`` routes need ``Authorization: Bearer <key>`` when an
API key is configured):

====== ============================== ==========================================================
Method Path                           Purpose
====== ============================== ==========================================================
GET    ``/health``                    Liveness probe (no auth)
GET    ``/openapi.json``              The OpenAPI 3.1 description (no auth)
POST   ``/v1/documents``              Upload a file (multipart: ``file``, optional ``id``)
POST   ``/v1/documents/text``         Add a document as JSON ``{"text", "source"?, "id"?}``
GET    ``/v1/documents``              List documents
GET    ``/v1/documents/{id}``         Document summary
DELETE ``/v1/documents/{id}``         Delete a document
POST   ``/v1/documents/{id}/ask``     Ask a question; returns a cited answer
POST   ``/v1/documents/{id}/retrieve`` Preview the sections that would be sent (no model call)
====== ============================== ==========================================================
"""

from __future__ import annotations

import email.parser
import email.policy
import json
import logging
import re
import secrets
import threading
import time
import urllib.parse
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._version import __version__
from .config import Strategy
from .exceptions import (
    ConfigurationError,
    DocumentLoadError,
    LongDocError,
    RefusalError,
    UnsupportedFormatError,
)
from .http import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
)
from .serialization import answer_to_dict, section_to_dict
from .service import DocumentNotFoundError, DocumentService

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_QUESTION_CHARS = 8_000
MAX_SOURCE_CHARS = 512
API_KEY_ENV = "LONGDOC_API_KEY"
_SECURITY_HEADERS = {"x-content-type-options": "nosniff", "cache-control": "no-store"}


# -- responses and errors --------------------------------------------------------------------


@dataclass(slots=True)
class ApiResponse:
    """A response ready to send: status, headers and body bytes."""

    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


class ApiError(Exception):
    """An error that becomes a JSON response ``{"error", "message", "details"}``."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = dict(details) if details else None
        self.headers = dict(headers) if headers else {}


def _json(status: int, data: Any, headers: Mapping[str, str] | None = None) -> ApiResponse:
    return ApiResponse(
        status,
        json.dumps(data, ensure_ascii=False).encode("utf-8"),
        {"content-type": "application/json; charset=utf-8", **(headers or {})},
    )


def error_response(err: ApiError) -> ApiResponse:
    """Render an :class:`ApiError` in the API's standard error format."""
    body = {"error": err.code, "message": err.message, "details": err.details}
    return _json(err.status, body, err.headers)


def map_exception(exc: BaseException) -> ApiError:
    """Translate library, provider and SDK errors into API errors.

    Provider messages are never returned to callers (they can contain fragments of API keys);
    only generic descriptions and the status and request ID are.
    """
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, DocumentNotFoundError):
        return ApiError(404, "not_found", str(exc))
    if isinstance(exc, RefusalError):
        return ApiError(
            422,
            "refused",
            str(exc),
            details={"category": exc.category, "request_id": exc.request_id},
        )
    if isinstance(exc, UnsupportedFormatError):
        return ApiError(415, "unsupported_format", str(exc))
    if isinstance(exc, DocumentLoadError):
        return ApiError(422, "invalid_document", str(exc))
    if isinstance(exc, ConfigurationError):
        return ApiError(422, "invalid_request", str(exc))
    if isinstance(exc, ProviderQuotaError):
        logger.error("The model provider account has no remaining credit or quota")
        return ApiError(
            502,
            "upstream_quota_exceeded",
            "The model provider account has no remaining credit. Add billing credit; "
            "retrying will not help.",
        )
    if isinstance(exc, ProviderRateLimitError):
        headers = {"retry-after": str(max(1, round(exc.retry_after)))} if exc.retry_after else {}
        return ApiError(
            429, "rate_limited", "The model provider is rate limiting requests.", headers=headers
        )
    if isinstance(exc, ProviderConnectionError):
        return ApiError(503, "upstream_unavailable", "The model provider could not be reached.")
    if isinstance(exc, ProviderError):
        logger.error("Model provider error: %s", type(exc).__name__)
        kind = "credentials" if isinstance(exc, ProviderAuthError) else "error"
        return ApiError(
            502,
            "upstream_error",
            f"The model provider returned an {kind}."
            if kind == "error"
            else "The model provider rejected the server's credentials.",
            details={"upstream_status": exc.status, "request_id": exc.request_id},
        )
    sdk = _sdk_error(exc)
    if sdk is not None:
        return sdk
    if isinstance(exc, LongDocError):
        return ApiError(400, "bad_request", str(exc))
    raise exc


def _sdk_error(exc: BaseException) -> ApiError | None:
    """Map errors from the official ``anthropic`` / ``openai`` SDKs, if a caller passes those."""
    if type(exc).__module__.split(".")[0] not in ("anthropic", "openai"):
        return None
    names = {c.__name__ for c in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    if "RateLimitError" in names:
        return ApiError(429, "rate_limited", "The model provider is rate limiting requests.")
    if "APIConnectionError" in names:
        return ApiError(503, "upstream_unavailable", "The model provider could not be reached.")
    if status is not None or "APIError" in names:
        return ApiError(
            502,
            "upstream_error",
            "The model provider returned an error.",
            details={"upstream_status": status, "request_id": getattr(exc, "request_id", None)},
        )
    return None


# -- rate limiting ------------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limiter: at most ``per_minute`` calls per caller in any 60 seconds.

    In-memory and per process, so with N workers the effective limit is up to N times higher.
    For a hard global limit, also enforce one at your gateway or load balancer.
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


# -- request validation ---------------------------------------------------------------------------


def _check_fields(data: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ApiError(422, "invalid_request", "The request body must be a JSON object.")
    extra = sorted(set(data) - allowed)
    if extra:
        raise ApiError(422, "invalid_request", f"Unknown field(s): {', '.join(extra)}.")
    return data


def _string(
    data: dict[str, Any], name: str, *, required: bool, max_chars: int | None = None
) -> str | None:
    value = data.get(name)
    if value is None:
        if required:
            raise ApiError(422, "invalid_request", f"{name}: field is required.")
        return None
    if not isinstance(value, str):
        raise ApiError(422, "invalid_request", f"{name}: must be a string.")
    if required and not value.strip():
        raise ApiError(422, "invalid_request", f"{name}: must not be empty.")
    if max_chars is not None and len(value) > max_chars:
        raise ApiError(422, "invalid_request", f"{name}: at most {max_chars} characters.")
    return value


def _parse_multipart(
    content_type: str, body: bytes
) -> tuple[str | None, bytes | None, dict[str, str]]:
    """Return ``(filename, file_bytes, other_fields)`` from a multipart/form-data body."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not message.is_multipart():
        raise ApiError(400, "bad_request", "Expected a multipart/form-data body.")
    filename: str | None = None
    data: bytes | None = None
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if not isinstance(name, str) or not isinstance(payload, bytes):
            continue
        if name == "file":
            filename = part.get_filename() or "upload"
            data = payload
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return filename, data, fields


# -- the API ---------------------------------------------------------------------------------------

_ID = r"(?P<id>[^/]+)"
_ROUTES: list[tuple[re.Pattern[str], str, tuple[str, ...]]] = [
    (re.compile(r"^/health$"), "health", ("GET",)),
    (re.compile(r"^/openapi\.json$"), "openapi", ("GET",)),
    (re.compile(r"^/v1/documents$"), "documents", ("GET", "POST")),
    (re.compile(r"^/v1/documents/text$"), "text", ("POST",)),
    (re.compile(rf"^/v1/documents/{_ID}$"), "document", ("GET", "DELETE")),
    (re.compile(rf"^/v1/documents/{_ID}/ask$"), "ask", ("POST",)),
    (re.compile(rf"^/v1/documents/{_ID}/retrieve$"), "retrieve", ("POST",)),
]
_PUBLIC = {"health", "openapi"}
_THROTTLED = {"documents:POST", "text", "ask"}


class LongDocApi:
    """Framework-free request handler. See the module docstring for the endpoints.

    Args:
        service: The :class:`DocumentService` that stores and answers questions about documents.
        api_key: If set, ``/v1`` requests must send ``Authorization: Bearer <api_key>``.
        max_upload_bytes: Upper bound on a request body.
        rate_limit_per_minute: If set, each client may make at most this many uploads plus
            questions per minute.
    """

    def __init__(
        self,
        service: DocumentService,
        *,
        api_key: str | None = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        rate_limit_per_minute: int | None = None,
    ) -> None:
        self.service = service
        self.api_key = api_key
        self.max_upload_bytes = max_upload_bytes
        self.limiter = RateLimiter(rate_limit_per_minute) if rate_limit_per_minute else None
        if api_key is None:
            logger.warning(
                "longdoc API is running WITHOUT authentication. Set %s for any shared deployment.",
                API_KEY_ENV,
            )

    # -- entry point ----------------------------------------------------------------------------

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes = b"",
        client: str = "unknown",
    ) -> ApiResponse:
        """Handle one request and return the response. Never raises for a bad request."""
        request_id = uuid.uuid4().hex
        try:
            response = self._dispatch(method.upper(), path, _lower(headers), body, client)
        except ApiError as err:
            response = error_response(err)
        except Exception as exc:
            try:
                response = error_response(map_exception(exc))
            except Exception:
                logger.exception("Unhandled error while serving %s %s", method, path.split("?")[0])
                response = error_response(
                    ApiError(500, "internal_error", "An unexpected error occurred.")
                )
        response.headers.update({**_SECURITY_HEADERS, "x-request-id": request_id})
        return response

    def _dispatch(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes, client: str
    ) -> ApiResponse:
        clean = urllib.parse.urlsplit(path).path.rstrip("/") or "/"
        for pattern, name, methods in _ROUTES:
            match = pattern.match(clean)
            if match is None:
                continue
            if method not in methods:
                raise ApiError(
                    405,
                    "method_not_allowed",
                    f"{method} is not allowed here.",
                    headers={"allow": ", ".join(methods)},
                )
            if name not in _PUBLIC:
                self._authenticate(headers)
            if name in _THROTTLED or f"{name}:{method}" in _THROTTLED:
                self._throttle(client)
            if len(body) > self.max_upload_bytes:
                raise ApiError(
                    413, "too_large", f"Requests are limited to {self.max_upload_bytes} bytes."
                )
            doc_id = urllib.parse.unquote(match.groupdict().get("id", "")) or None
            handler: Callable[..., ApiResponse] = getattr(self, f"_{name}")
            return handler(method, headers, body, doc_id)
        raise ApiError(404, "not_found", "No such route.")

    def _authenticate(self, headers: Mapping[str, str]) -> None:
        if self.api_key is None:
            return
        auth = headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        supplied = token.strip() if scheme.lower() == "bearer" else ""
        if not secrets.compare_digest(supplied.encode(), self.api_key.encode()):
            raise ApiError(
                401,
                "unauthorized",
                "Missing or invalid bearer token.",
                headers={"www-authenticate": "Bearer"},
            )

    def _throttle(self, client: str) -> None:
        if self.limiter is None:
            return
        wait = self.limiter.check(client)
        if wait is not None:
            raise ApiError(
                429,
                "rate_limited",
                "Too many requests; slow down.",
                details={"retry_after_seconds": wait},
                headers={"retry-after": str(wait)},
            )

    # -- handlers --------------------------------------------------------------------------------

    def _health(self, *_: Any) -> ApiResponse:
        return _json(200, {"status": "ok", "version": __version__})

    def _openapi(self, *_: Any) -> ApiResponse:
        return _json(200, openapi_spec())

    def _documents(
        self, method: str, headers: Mapping[str, str], body: bytes, _id: str | None
    ) -> ApiResponse:
        if method == "GET":
            return _json(200, [info.to_dict() for info in self.service.list_documents()])
        content_type = headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ApiError(
                415, "unsupported_format", "Upload with Content-Type: multipart/form-data."
            )
        filename, data, fields = _parse_multipart(content_type, body)
        if data is None:
            raise ApiError(422, "invalid_request", "file: field is required.")
        info = self.service.add_bytes(
            data, filename=filename or "upload", document_id=fields.get("id") or None
        )
        return _json(201, info.to_dict())

    def _text(
        self, _m: str, headers: Mapping[str, str], body: bytes, _id: str | None
    ) -> ApiResponse:
        data = _check_fields(_loads(body), {"text", "source", "id"})
        text = _string(data, "text", required=True)
        source = _string(data, "source", required=False, max_chars=MAX_SOURCE_CHARS)
        doc_id = _string(data, "id", required=False)
        assert text is not None
        info = self.service.add_text(text, source=source, document_id=doc_id or None)
        return _json(201, info.to_dict())

    def _document(
        self, method: str, _h: Mapping[str, str], _b: bytes, doc_id: str | None
    ) -> ApiResponse:
        assert doc_id is not None
        if method == "GET":
            return _json(200, self.service.get_info(doc_id).to_dict())
        if not self.service.delete(doc_id):
            raise DocumentNotFoundError(doc_id)
        return ApiResponse(204)

    def _ask(self, _m: str, _h: Mapping[str, str], body: bytes, doc_id: str | None) -> ApiResponse:
        assert doc_id is not None
        data = _check_fields(_loads(body), {"question", "strategy", "include_context_text"})
        question = _string(data, "question", required=True, max_chars=MAX_QUESTION_CHARS)
        strategy = _strategy(data.get("strategy"))
        include = data.get("include_context_text", False)
        if not isinstance(include, bool):
            raise ApiError(422, "invalid_request", "include_context_text: must be true or false.")
        assert question is not None
        answer = self.service.ask(doc_id, question, strategy=strategy)
        return _json(200, answer_to_dict(answer, include_context_text=include))

    def _retrieve(
        self, _m: str, _h: Mapping[str, str], body: bytes, doc_id: str | None
    ) -> ApiResponse:
        assert doc_id is not None
        data = _check_fields(_loads(body), {"question"})
        question = _string(data, "question", required=True, max_chars=MAX_QUESTION_CHARS)
        assert question is not None
        sections = self.service.retrieve(doc_id, question)
        return _json(200, [section_to_dict(s, include_text=True) for s in sections])


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def _loads(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8") or "null")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, "bad_request", "The request body is not valid JSON.") from exc


def _strategy(value: Any) -> Strategy | None:
    if value is None:
        return None
    try:
        return Strategy(value)
    except ValueError as exc:
        names = ", ".join(s.value for s in Strategy)
        raise ApiError(422, "invalid_request", f"strategy: must be one of {names}.") from exc


# -- OpenAPI description ------------------------------------------------------------------------


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def openapi_spec() -> dict[str, Any]:
    """The OpenAPI 3.1 description of this API."""
    ref = _ref
    err = {"$ref": "#/components/schemas/ErrorResponse"}
    err_responses = {
        "401": {
            "description": "Missing or invalid bearer token",
            "content": {"application/json": {"schema": err}},
        },
        "404": {
            "description": "No such document",
            "content": {"application/json": {"schema": err}},
        },
        "422": {"description": "Invalid input", "content": {"application/json": {"schema": err}}},
        "429": {"description": "Rate limited", "content": {"application/json": {"schema": err}}},
    }
    doc_param = {
        "name": "document_id",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    }
    secured: list[dict[str, list[str]]] = [{"bearerAuth": []}]

    def op(summary: str, ok: str, schema: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
        content = {"application/json": {"schema": schema}} if schema else None
        response: dict[str, Any] = {"description": "Success"}
        if content:
            response["content"] = content
        return {
            "summary": summary,
            "security": secured,
            "responses": {ok: response, **err_responses},
            **extra,
        }

    def json_body(name: str) -> dict[str, Any]:
        return {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}},
        }

    string_or_null = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    int_or_null = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "longdoc API",
            "version": __version__,
            "description": "Question answering over long documents: parent-child retrieval, prompt "
            "caching and cited answers.",
            "license": {"name": "MIT", "identifier": "MIT"},
        },
        "paths": {
            "/health": {
                "get": {"summary": "Liveness probe", "responses": {"200": {"description": "OK"}}}
            },
            "/v1/documents": {
                "get": op(
                    "List documents", "200", {"type": "array", "items": ref("DocumentResponse")}
                ),
                "post": op(
                    "Upload a file and index it",
                    "201",
                    ref("DocumentResponse"),
                    requestBody={
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file"],
                                    "properties": {
                                        "file": {"type": "string", "format": "binary"},
                                        "id": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                ),
            },
            "/v1/documents/text": {
                "post": op(
                    "Index a document sent as plain text",
                    "201",
                    ref("DocumentResponse"),
                    requestBody=json_body("TextDocumentRequest"),
                ),
            },
            "/v1/documents/{document_id}": {
                "get": op(
                    "Get one document's summary",
                    "200",
                    ref("DocumentResponse"),
                    parameters=[doc_param],
                ),
                "delete": op("Delete a document", "204", None, parameters=[doc_param]),
            },
            "/v1/documents/{document_id}/ask": {
                "post": op(
                    "Answer a question about a document, with citations",
                    "200",
                    ref("AnswerResponse"),
                    parameters=[doc_param],
                    requestBody=json_body("AskRequest"),
                ),
            },
            "/v1/documents/{document_id}/retrieve": {
                "post": op(
                    "Preview the sections retrieval would send (no model call)",
                    "200",
                    {"type": "array", "items": ref("SectionResponse")},
                    parameters=[doc_param],
                    requestBody=json_body("RetrieveRequest"),
                ),
            },
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "required": ["error", "message"],
                    "properties": {
                        "error": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": ["object", "null"]},
                    },
                },
                "DocumentResponse": {
                    "type": "object",
                    "required": ["id", "source", "sections", "passages", "pages"],
                    "properties": {
                        "id": {"type": "string"},
                        "source": string_or_null,
                        "sections": {"type": "integer"},
                        "passages": {"type": "integer"},
                        "pages": int_or_null,
                    },
                },
                "SectionResponse": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string"},
                        "heading": string_or_null,
                        "pages": string_or_null,
                        "page_start": int_or_null,
                        "page_end": int_or_null,
                        "text": string_or_null,
                    },
                },
                "UsageResponse": {
                    "type": "object",
                    "properties": {
                        k: {"type": "integer"}
                        for k in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                            "total_input_tokens",
                        )
                    }
                    | {"cache_hit_ratio": {"type": "number"}},
                },
                "AnswerResponse": {
                    "type": "object",
                    "required": ["text", "citations", "context", "usage", "model"],
                    "properties": {
                        "text": {"type": "string"},
                        "citations": {"type": "array", "items": ref("SectionResponse")},
                        "context": {"type": "array", "items": ref("SectionResponse")},
                        "usage": ref("UsageResponse"),
                        "model": {"type": "string"},
                        "stop_reason": string_or_null,
                        "truncated": {"type": "boolean"},
                        "fallback_used": {"type": "boolean"},
                        "request_id": string_or_null,
                    },
                },
                "TextDocumentRequest": {
                    "type": "object",
                    "required": ["text"],
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "source": {"type": "string", "maxLength": MAX_SOURCE_CHARS},
                        "id": {"type": "string"},
                    },
                },
                "AskRequest": {
                    "type": "object",
                    "required": ["question"],
                    "additionalProperties": False,
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_QUESTION_CHARS,
                        },
                        "strategy": {"enum": [s.value for s in Strategy]},
                        "include_context_text": {"type": "boolean"},
                    },
                },
                "RetrieveRequest": {
                    "type": "object",
                    "required": ["question"],
                    "additionalProperties": False,
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_QUESTION_CHARS,
                        }
                    },
                },
            },
        },
    }
