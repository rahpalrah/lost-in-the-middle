"""A production-capable HTTP server built on the Python standard library only.

``create_server`` wraps :class:`longdoc.api.LongDocApi` in :class:`http.server.ThreadingHTTPServer`:
no FastAPI, no Uvicorn, nothing to install. It adds what a bare ``http.server`` lacks:

* request size limits (a declared body larger than the limit is refused without being read);
* a cap on concurrent requests: beyond it the server answers ``503`` with ``Retry-After`` instead
  of running out of threads or memory;
* socket timeouts, so a stalled client cannot hold a thread forever;
* graceful shutdown on SIGTERM/SIGINT (for containers);
* security headers, no server version leak, and logging that never includes headers or bodies.

Usage::

    longdoc serve --port 8000

or from Python::

    from longdoc.server import create_server
    server = create_server(service, port=8000, api_key="...")
    server.serve_forever()

For TLS, terminate HTTPS at a reverse proxy or load balancer in front of this server.
"""

from __future__ import annotations

import logging
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..accounts import AccountService
from ..api import ApiError, ApiResponse, LongDocApi, error_response
from ..service import DocumentService
from .envconfig import Settings, settings_from_env

logger = logging.getLogger(__name__)

_BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class LongDocServer(ThreadingHTTPServer):
    """A threaded HTTP server for a :class:`LongDocApi`."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256

    def __init__(
        self,
        address: tuple[str, int],
        api: LongDocApi,
        *,
        max_concurrency: int = 32,
        trust_proxy: bool = False,
        request_timeout: float = 60.0,
    ) -> None:
        self.api = api
        self.trust_proxy = trust_proxy
        self.request_timeout = request_timeout
        self.slots = threading.BoundedSemaphore(max_concurrency)
        super().__init__(address, _Handler)

    @property
    def url(self) -> str:
        """The base URL this server listens on."""
        return f"http://{self.socket.getsockname()[0]}:{self.server_port}"

    def start_in_thread(self) -> threading.Thread:
        """Serve in a background thread (useful in tests and notebooks)."""
        thread = threading.Thread(target=self.serve_forever, daemon=True, name="longdoc-server")
        thread.start()
        return thread


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "longdoc"
    sys_version = ""
    server: LongDocServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.request_timeout)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s %s", self.address_string(), format % args)  # never headers or bodies

    def _client(self) -> str:
        if self.server.trust_proxy:
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return str(self.client_address[0])

    def _handle(self) -> None:
        srv = self.server
        try:
            if self.headers.get("Transfer-Encoding"):
                return self._send(
                    error_response(ApiError(411, "length_required", "Send a Content-Length.")),
                    close=True,
                )
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                return self._send(
                    error_response(ApiError(400, "bad_request", "Invalid Content-Length.")),
                    close=True,
                )
            if length > srv.api.max_upload_bytes:
                err = ApiError(
                    413, "too_large", f"Requests are limited to {srv.api.max_upload_bytes} bytes."
                )
                return self._send(error_response(err), close=True)  # body is not read
            body = self.rfile.read(length) if length and self.command in _BODY_METHODS else b""
            if len(body) < length:
                return self._send(
                    error_response(ApiError(400, "bad_request", "Incomplete request body.")),
                    close=True,
                )
            if not srv.slots.acquire(blocking=False):
                busy = ApiError(
                    503,
                    "server_busy",
                    "The server is at capacity; try again shortly.",
                    headers={"retry-after": "1"},
                )
                return self._send(error_response(busy))
            try:
                response = srv.api.handle(
                    self.command, self.path, dict(self.headers.items()), body, self._client()
                )
            finally:
                srv.slots.release()
            self._send(response)
        except (ConnectionError, TimeoutError, OSError):
            self.close_connection = True  # the client went away; nothing to send

    def _send(self, response: ApiResponse, *, close: bool = False) -> None:
        self.send_response(response.status)
        for name, value in response.headers.items():
            self.send_header(name, value)
        if response.status not in (204, 304):
            self.send_header("Content-Length", str(len(response.body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD" and response.status not in (204, 304):
            self.wfile.write(response.body)

    # The method names are fixed by BaseHTTPRequestHandler.
    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _handle  # noqa: N815


def create_server(
    service: DocumentService,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    max_upload_bytes: int | None = None,
    rate_limit_per_minute: int | None = None,
    max_concurrency: int = 32,
    trust_proxy: bool = False,
    accounts: AccountService | None = None,
) -> LongDocServer:
    """Build (but do not start) a server for ``service``. Use ``port=0`` for a free port."""
    kwargs: dict[str, Any] = {}
    if max_upload_bytes is not None:
        kwargs["max_upload_bytes"] = max_upload_bytes
    api = LongDocApi(
        service,
        api_key=api_key,
        rate_limit_per_minute=rate_limit_per_minute,
        accounts=accounts,
        **kwargs,
    )
    return LongDocServer(
        (host, port), api, max_concurrency=max_concurrency, trust_proxy=trust_proxy
    )


def serve(settings: Settings | None = None, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API until interrupted, configured from ``settings`` (default: the environment)."""
    settings = settings or settings_from_env()
    server = create_server(
        settings.service,
        host=host,
        port=port,
        api_key=settings.api_key,
        max_upload_bytes=settings.max_upload_bytes,
        rate_limit_per_minute=settings.rate_limit_per_minute,
        max_concurrency=settings.max_concurrency,
        trust_proxy=settings.trust_proxy,
        accounts=settings.accounts,
    )

    def stop(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("longdoc listening on %s (provider: %s)", server.url, settings.provider)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if settings.accounts is not None:
            settings.accounts.flush()
