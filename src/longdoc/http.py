"""A small, standard-library-only HTTP layer for talking to model providers.

It replaces the providers' own SDK packages, so longdoc needs no third-party dependency to
call Claude or OpenAI. It provides:

* :func:`post_json` - POST JSON, parse JSON, retry transient failures with backoff.
* A typed error hierarchy (:class:`ProviderError` and subclasses) that does the same job for
  every provider: bad key, no credit, rate limit, outage, network failure.

Safety rules built in, because the requests carry API keys:

* Redirects are never followed, so a key is never replayed to another host.
* Plain ``http://`` is refused except to the local machine (or with ``allow_insecure=True``).
* Error messages never contain the key, request headers or the provider's raw message for
  authentication failures (OpenAI echoes part of the key there).
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._version import __version__
from .exceptions import ConfigurationError, LongDocError

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_RETRY_AFTER_CAP = 30.0
_OUT_OF_CREDIT_CODES = frozenset(
    {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}
)


class Secret:
    """Holds an API key and refuses to show it: ``repr``, ``str`` and pickling hide the value.

    Call :meth:`reveal` at the one place the key is sent. This keeps keys out of logs,
    tracebacks, ``repr()`` output and accidental ``vars()`` dumps.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """Return the secret value. Use only when building a request."""
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("Secret values cannot be pickled")

    def __bool__(self) -> bool:
        return bool(self._value)


# -- errors ----------------------------------------------------------------------------


class ProviderError(LongDocError):
    """Base class for failures talking to a model provider (Claude, OpenAI, ...).

    Attributes:
        status: HTTP status, or ``None`` for network failures.
        code: The provider's error code, if any.
        error_type: The provider's error type, if any.
        request_id: The provider's request ID, for support tickets.
        retry_after: Seconds the provider asked us to wait, if any.
        provider_message: The provider's message. ``None`` for authentication failures,
            because some providers echo a fragment of the key there.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.error_type = error_type
        self.request_id = request_id
        self.retry_after = retry_after
        self.provider_message = provider_message


class ProviderAuthError(ProviderError):
    """The provider rejected the API key (HTTP 401)."""


class ProviderPermissionError(ProviderError):
    """The key is valid but not allowed to do this (HTTP 403)."""


class ProviderRateLimitError(ProviderError):
    """Too many requests (HTTP 429). Retrying later helps."""


class ProviderQuotaError(ProviderError):
    """The account has no credit or quota left. Retrying never helps; add credit."""


class ProviderBadRequestError(ProviderError):
    """The provider rejected the request itself (HTTP 400, 404, 422 and similar)."""


class ProviderServerError(ProviderError):
    """The provider failed or was overloaded (HTTP 5xx) even after retries."""


class ProviderConnectionError(ProviderError):
    """The provider could not be reached, or the connection timed out."""


# -- response / transport types ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Response:
    """A parsed provider response."""

    status: int
    body: dict[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def request_id(self) -> str | None:
        """The provider's request ID header, if present."""
        return self.headers.get("request-id") or self.headers.get("x-request-id")


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]], Response]
"""``(url, headers, json_body) -> Response``. Inject a fake in tests."""


# -- classification -----------------------------------------------------------------------


def classify_error(status: int, body: Any, headers: Mapping[str, str]) -> ProviderError:
    """Turn an HTTP error response into the right :class:`ProviderError`."""
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        err = {}
    code = err.get("code") if isinstance(err.get("code"), str) else None
    etype = err.get("type") if isinstance(err.get("type"), str) else None
    message = err.get("message") if isinstance(err.get("message"), str) else None
    request_id = (
        headers.get("request-id")
        or headers.get("x-request-id")
        or (body.get("request_id") if isinstance(body, dict) else None)
    )
    retry_after = _retry_after(headers)
    common: dict[str, Any] = {
        "status": status,
        "code": code,
        "error_type": etype,
        "request_id": request_id if isinstance(request_id, str) else None,
        "retry_after": retry_after,
    }

    if status == 401:
        return ProviderAuthError(
            "The provider rejected the API key (HTTP 401). Check that the key is correct, "
            "complete and active.",
            **common,
        )
    out_of_credit = (
        etype == "insufficient_quota"
        or code in _OUT_OF_CREDIT_CODES
        or status == 402
        or (message is not None and "credit balance" in message.lower())
    )
    if out_of_credit:
        return ProviderQuotaError(
            "The provider account has no remaining credit or quota. Add billing credit; "
            "retrying will not help.",
            provider_message=message,
            **common,
        )
    if status == 403:
        return ProviderPermissionError(
            "The provider denied access to this model or project for your key (HTTP 403).",
            provider_message=message,
            **common,
        )
    if status == 429:
        return ProviderRateLimitError(
            "The provider is rate limiting requests (HTTP 429).", provider_message=message, **common
        )
    if status >= 500:
        return ProviderServerError(
            f"The provider returned HTTP {status}.", provider_message=message, **common
        )
    return ProviderBadRequestError(
        f"The provider rejected the request (HTTP {status}): {message or 'no details'}",
        provider_message=message,
        **common,
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    ms = headers.get("retry-after-ms")
    seconds = headers.get("retry-after")
    try:
        if ms is not None:
            return max(0.0, float(ms) / 1000.0)
        if seconds is not None:
            return max(0.0, float(seconds))
    except ValueError:
        return None
    return None


def _is_retryable(status: int, error: ProviderError) -> bool:
    if isinstance(error, ProviderQuotaError):
        return False
    return status in (408, 409, 429) or status >= 500


# -- the transport -------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: never replay an Authorization header to another location."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def check_url(url: str, *, allow_insecure: bool = False) -> None:
    """Raise if sending credentials to ``url`` would be unsafe."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigurationError(f"Not a valid http(s) URL: {url!r}")
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK and not allow_insecure:
        raise ConfigurationError(
            f"Refusing to send credentials over plain HTTP to {parsed.hostname}. Use https://, "
            "or pass allow_insecure=True if this is a trusted private network."
        )


def post_json(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    *,
    timeout: float = 600.0,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    allow_insecure: bool = False,
) -> Response:
    """POST ``body`` as JSON and return the parsed JSON response.

    Retries connection failures, timeouts, HTTP 408/409/429 (not "out of credit") and 5xx up to
    ``max_retries`` times, with exponential backoff and jitter, honouring ``Retry-After``.

    Raises:
        ProviderError: A subclass describing why the call failed.
        ConfigurationError: ``url`` is unsafe or invalid.
    """
    check_url(url, allow_insecure=allow_insecure)
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": f"longdoc-qa/{__version__}",
        **{k.lower(): v for k, v in headers.items()},
    }

    attempt = 0
    while True:
        error: ProviderError
        try:
            req = urllib.request.Request(url, data=payload, headers=request_headers, method="POST")
            with _OPENER.open(req, timeout=timeout) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ProviderBadRequestError("The provider response was too large.")
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderServerError(
                        "The provider returned a response that is not valid JSON.",
                        status=resp.status,
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ProviderServerError(
                        "The provider returned an unexpected response.", status=resp.status
                    )
                return Response(status=resp.status, body=parsed, headers=resp_headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read(1024 * 1024)
            try:
                err_body: Any = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                err_body = None
            err_headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
            error = classify_error(exc.code, err_body, err_headers)
            status = exc.code
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            error = ProviderConnectionError(
                f"Could not reach the provider ({type(reason).__name__}).", provider_message=None
            )
            status = 0

        if attempt >= max_retries or (status and not _is_retryable(status, error)):
            raise error
        delay = error.retry_after if error.retry_after is not None else 0.5 * (2**attempt)
        delay = min(delay, _RETRY_AFTER_CAP) + random.uniform(0, 0.25)
        logger.warning("Provider call failed (%s); retrying in %.1fs", type(error).__name__, delay)
        sleep(delay)
        attempt += 1
