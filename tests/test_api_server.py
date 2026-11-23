"""The built-in HTTP server, exercised over real sockets with only the standard library."""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from longdoc import DocumentService, QAConfig
from longdoc.extractive import ExtractiveClient
from longdoc.http import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderQuotaError,
    ProviderRateLimitError,
)
from longdoc.server import create_server

from .test_pdf import simple_pdf

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CONTRACT = (
    "# 19. Termination\n\nEither party may terminate on ninety (90) days written notice, provided "
    "that the fee in Schedule C is paid.\n\n# 21. Governing Law\n\n"
    "This Agreement is governed by the laws of the State of New York.\n"
)


class Client:
    """A tiny HTTP client over http.client (standard library)."""

    def __init__(self, server: Any) -> None:
        self.host, self.port = server.socket.getsockname()[:2]

    def request(
        self,
        method: str,
        path: str,
        body: bytes | dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: bool = True,
    ) -> tuple[int, dict[str, str], Any]:
        hdrs = dict(AUTH if auth else {}) | (headers or {})
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            conn.request(method, path, body=body, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            response_headers = {k.lower(): v for k, v in resp.getheaders()}
            ctype = response_headers.get("content-type", "")
            data: Any = json.loads(raw) if raw and "json" in ctype else raw
            return resp.status, response_headers, data
        finally:
            conn.close()

    def upload(self, filename: str, content: bytes, doc_id: str | None = None) -> tuple[int, Any]:
        boundary = "----longdoc" + uuid.uuid4().hex
        parts = []
        if doc_id:
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="id"\r\n\r\n{doc_id}\r\n'.encode()
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n".encode()
            + content
            + b"\r\n"
        )
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        status, _, data = self.request(
            "POST",
            "/v1/documents",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return status, data


def start(service: DocumentService | None = None, **kwargs: Any) -> Any:
    svc = service or DocumentService(client=ExtractiveClient(), config=QAConfig(fallbacks=None))
    kwargs.setdefault("api_key", TOKEN)
    server = create_server(svc, port=0, **kwargs)
    server.start_in_thread()
    return server


@pytest.fixture
def server() -> Iterator[Any]:
    s = start()
    yield s
    s.shutdown()
    s.server_close()


@pytest.fixture
def api(server: Any) -> Client:
    return Client(server)


# -- basics ---------------------------------------------------------------------------------


def test_health_and_openapi_are_public(api: Client) -> None:
    status, _, body = api.request("GET", "/health", auth=False)
    assert status == 200
    assert body["status"] == "ok"
    status, _, spec = api.request("GET", "/openapi.json", auth=False)
    assert status == 200
    assert spec["openapi"] == "3.1.0"
    assert "/v1/documents/{document_id}/ask" in spec["paths"]


def test_openapi_document_is_valid() -> None:
    validator = pytest.importorskip("openapi_spec_validator")
    from longdoc.api import openapi_spec

    validator.validate(openapi_spec())


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}]
)
def test_authentication_is_required(api: Client, headers: dict[str, str]) -> None:
    status, resp_headers, body = api.request("GET", "/v1/documents", headers=headers, auth=False)
    assert status == 401
    assert body["error"] == "unauthorized"
    assert resp_headers["www-authenticate"] == "Bearer"


def test_response_headers_are_safe_and_do_not_leak_the_runtime(api: Client) -> None:
    _, h1, _ = api.request("GET", "/health", auth=False)
    _, h2, _ = api.request("GET", "/health", auth=False)
    assert h1["x-content-type-options"] == "nosniff"
    assert h1["cache-control"] == "no-store"
    assert "python" not in h1.get("server", "").lower()
    assert re.fullmatch(r"[0-9a-f]{32}", h1["x-request-id"])
    assert h1["x-request-id"] != h2["x-request-id"]


def test_keep_alive_serves_several_requests_on_one_connection(server: Any) -> None:
    host, port = server.socket.getsockname()[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    for _ in range(3):
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
    conn.close()


# -- the full document workflow ---------------------------------------------------------------


def test_upload_ask_retrieve_delete_workflow(api: Client) -> None:
    status, info = api.upload("contract.md", CONTRACT.encode(), doc_id="acme-msa")
    assert status == 201, info
    assert info["id"] == "acme-msa"
    assert info["sections"] == 2

    assert [d["id"] for d in api.request("GET", "/v1/documents")[2]] == ["acme-msa"]
    assert api.request("GET", "/v1/documents/acme-msa")[2]["source"] == "contract.md"

    status, _, answer = api.request(
        "POST",
        "/v1/documents/acme-msa/ask",
        {"question": "How much notice is needed to terminate?"},
    )
    assert status == 200
    assert "ninety (90) days" in answer["text"]
    assert [c["heading"] for c in answer["citations"]] == ["# 19. Termination"]
    assert answer["model"] == "longdoc-extractive-1"
    assert "text" not in answer["context"][0]  # only sent when asked for

    status, _, answer = api.request(
        "POST",
        "/v1/documents/acme-msa/ask",
        {
            "question": "Which law governs?",
            "strategy": "cached_document",
            "include_context_text": True,
        },
    )
    assert "State of New York" in answer["text"]
    assert answer["context"][0]["text"]

    status, _, sections = api.request(
        "POST", "/v1/documents/acme-msa/retrieve", {"question": "governing law"}
    )
    assert status == 200
    assert "New York" in sections[0]["text"]

    assert api.request("DELETE", "/v1/documents/acme-msa")[0] == 204
    assert api.request("GET", "/v1/documents/acme-msa")[2]["error"] == "not_found"
    assert api.request("DELETE", "/v1/documents/acme-msa")[0] == 404


def test_text_documents_and_pdf_upload_need_no_libraries(api: Client) -> None:
    status, _, info = api.request(
        "POST", "/v1/documents/text", {"text": CONTRACT, "id": "t1", "source": "inline"}
    )
    assert status == 201
    assert info["source"] == "inline"

    pdf = simple_pdf(
        "BT /F1 12 Tf 72 700 Td (Payment is due within thirty \\(30\\) days of the invoice date.) Tj ET"
    )
    status, info = api.upload("invoice.pdf", pdf, doc_id="pdf1")
    assert status == 201, info
    assert info["pages"] == 1
    _, _, answer = api.request(
        "POST", "/v1/documents/pdf1/ask", {"question": "When is payment due?"}
    )
    assert "thirty (30) days" in answer["text"]
    assert answer["citations"][0]["pages"] == "1"


# -- validation and errors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "status", "code"),
    [
        ("/v1/documents/text", b"{not json", 400, "bad_request"),
        ("/v1/documents/text", {"text": ""}, 422, "invalid_request"),
        ("/v1/documents/text", {"text": "x", "surprise": 1}, 422, "invalid_request"),
        ("/v1/documents/text", {"text": "x", "id": "bad id"}, 422, "invalid_request"),
        ("/v1/documents/text", b"[1, 2]", 422, "invalid_request"),
        ("/v1/documents/d/ask", {"question": ""}, 422, "invalid_request"),
        ("/v1/documents/d/ask", {"question": "x" * 8001}, 422, "invalid_request"),
        ("/v1/documents/d/ask", {"question": "q", "strategy": "nope"}, 422, "invalid_request"),
        (
            "/v1/documents/d/ask",
            {"question": "q", "include_context_text": "yes"},
            422,
            "invalid_request",
        ),
        ("/v1/documents/missing/ask", {"question": "q"}, 404, "not_found"),
        ("/v1/documents/..%2Fetc%2Fpasswd/ask", {"question": "q"}, 422, "invalid_request"),
    ],
)
def test_input_validation(api: Client, path: str, body: Any, status: int, code: str) -> None:
    api.request("POST", "/v1/documents/text", {"text": "Some text about fees.", "id": "d"})
    got_status, _, data = api.request("POST", path, body)
    assert got_status == status
    assert data["error"] == code


def test_upload_errors(api: Client) -> None:
    assert api.upload("sheet.xlsx", b"x")[0] == 415
    status, data = api.upload("empty.txt", b"   ")
    assert (status, data["error"]) == (422, "invalid_document")
    status, data = api.upload("a.pdf", b"not really a pdf")
    assert status == 422
    status, _, data = api.request("POST", "/v1/documents", b"plain", {"Content-Type": "text/plain"})
    assert (status, data["error"]) == (415, "unsupported_format")
    boundary = "xyz"
    body = f'--{boundary}\r\nContent-Disposition: form-data; name="id"\r\n\r\nonly-id\r\n--{boundary}--\r\n'
    status, _, data = api.request(
        "POST",
        "/v1/documents",
        body.encode(),
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert (status, data["error"]) == (422, "invalid_request")


def test_routes_and_methods(api: Client) -> None:
    status, headers, data = api.request("PUT", "/v1/documents")
    assert (status, data["error"]) == (405, "method_not_allowed")
    assert "POST" in headers["allow"]
    assert api.request("GET", "/nope")[0] == 404
    assert api.request("GET", "/v1/documents/")[0] == 200  # trailing slash tolerated


def test_oversized_bodies_are_refused_without_reading_them() -> None:
    s = start(max_upload_bytes=1000)
    try:
        c = Client(s)
        status, _, data = c.request(
            "POST", "/v1/documents/text", b"x" * 2000, {"Content-Type": "application/json"}
        )
        assert (status, data["error"]) == (413, "too_large")
        assert c.request("GET", "/health", auth=False)[0] == 200  # still healthy
    finally:
        s.shutdown()
        s.server_close()


def test_chunked_requests_are_refused(server: Any) -> None:
    host, port = server.socket.getsockname()[:2]
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(
            b"POST /v1/documents/text HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
            b"Authorization: Bearer test-token\r\n\r\n0\r\n\r\n"
        )
        reply = sock.recv(4096)
    assert reply.startswith(b"HTTP/1.1 411")


def test_garbage_and_malformed_requests_do_not_kill_the_server(server: Any) -> None:
    host, port = server.socket.getsockname()[:2]
    for junk in (
        b"\x00\x01\x02garbage\r\n\r\n",
        b"GET\r\n\r\n",
        b"POST /x HTTP/1.1\r\nContent-Length: abc\r\n\r\n",
    ):
        with socket.create_connection((host, port), timeout=10) as sock:
            sock.sendall(junk)
            sock.settimeout(5)
            with contextlib.suppress(TimeoutError, ConnectionError):
                sock.recv(4096)
    assert Client(server).request("GET", "/health", auth=False)[0] == 200


# -- limits ---------------------------------------------------------------------------------------


def test_rate_limit_applies_per_client_and_costly_routes_only() -> None:
    s = start(rate_limit_per_minute=3)
    try:
        c = Client(s)
        c.request("POST", "/v1/documents/text", {"text": "Fees are due in 30 days.", "id": "d"})
        codes = [
            c.request("POST", "/v1/documents/d/ask", {"question": "fees?"})[0] for _ in range(4)
        ]
        assert codes == [200, 200, 429, 429]
        _, headers, data = c.request("POST", "/v1/documents/d/ask", {"question": "fees?"})
        assert data["error"] == "rate_limited"
        assert int(headers["retry-after"]) >= 1
        assert c.request("GET", "/v1/documents/d")[0] == 200  # cheap routes are not throttled
        assert c.request("GET", "/health", auth=False)[0] == 200
    finally:
        s.shutdown()
        s.server_close()


def test_forwarded_for_is_honoured_only_behind_a_trusted_proxy() -> None:
    def two_clients(trust: bool) -> list[int]:
        s = start(rate_limit_per_minute=2, trust_proxy=trust)
        try:
            c = Client(s)
            c.request("POST", "/v1/documents/text", {"text": "Fees are due.", "id": "d"})
            return [
                c.request(
                    "POST", "/v1/documents/d/ask", {"question": "fees?"}, {"X-Forwarded-For": ip}
                )[0]
                for ip in ("1.1.1.1", "2.2.2.2")
            ]
        finally:
            s.shutdown()
            s.server_close()

    assert two_clients(trust=True) == [200, 200]  # each forwarded address has its own budget
    assert two_clients(trust=False) == [200, 429]  # a spoofed header cannot dodge the limit


class _SlowClient:
    """A client whose answers block until released, to hold request slots."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.inner = ExtractiveClient()
        outer = self

        class Messages:
            def stream(self, **kwargs: Any) -> Any:
                outer.started.set()
                outer.release.wait(10)
                return outer.inner.beta.messages.stream(**kwargs)

            create = stream

        class Beta:
            messages = Messages()

        self.beta = Beta()

    def to_async(self) -> Any:
        raise NotImplementedError


def test_concurrency_cap_returns_503_instead_of_exhausting_the_server() -> None:
    slow = _SlowClient()
    svc = DocumentService(client=slow, config=QAConfig(fallbacks=None))
    svc.add_text("Fees are due in 30 days.", document_id="d")
    s = start(svc, max_concurrency=1)
    try:
        c = Client(s)
        first: list[tuple[int, Any]] = []
        t = threading.Thread(
            target=lambda: first.append(
                c.request("POST", "/v1/documents/d/ask", {"question": "fees?"})[::2]
            )
        )
        t.start()
        assert slow.started.wait(10)
        status, headers, data = c.request("POST", "/v1/documents/d/ask", {"question": "fees?"})
        assert (status, data["error"]) == (503, "server_busy")
        assert headers["retry-after"] == "1"
        slow.release.set()
        t.join(10)
        assert first[0][0] == 200
    finally:
        slow.release.set()
        s.shutdown()
        s.server_close()


def test_many_concurrent_clients_all_get_correct_answers() -> None:
    s = start(max_concurrency=64)
    try:
        c = Client(s)
        c.request("POST", "/v1/documents/text", {"text": CONTRACT, "id": "d"})
        results: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            status, _, data = c.request(
                "POST", "/v1/documents/d/ask", {"question": "Which law governs?"}
            )
            with lock:
                results.append(status if "State of New York" in data.get("text", "") else -1)

        threads = [threading.Thread(target=worker) for _ in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert results == [200] * 40
    finally:
        s.shutdown()
        s.server_close()


# -- error mapping from the model provider ---------------------------------------------------------


class _Failing:
    def __init__(self, exc: Exception) -> None:
        class Messages:
            def stream(self_inner, **kwargs: Any) -> Any:  # noqa: N805
                raise exc

            create = stream

        class Beta:
            messages = Messages()

        self.beta = Beta()


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (ProviderQuotaError("no credit", status=429), 502, "upstream_quota_exceeded"),
        (ProviderRateLimitError("slow", status=429, retry_after=5), 429, "rate_limited"),
        (ProviderAuthError("bad key", status=401, request_id="req_1"), 502, "upstream_error"),
        (ProviderConnectionError("offline"), 503, "upstream_unavailable"),
        (RuntimeError("secret internal detail sk-abc123"), 500, "internal_error"),
    ],
)
def test_provider_failures_become_clean_http_errors(exc: Exception, status: int, code: str) -> None:
    svc = DocumentService(client=_Failing(exc), config=QAConfig(fallbacks=None))
    svc.add_text("Fees are due in 30 days.", document_id="d")
    s = start(svc)
    try:
        got, headers, data = Client(s).request("POST", "/v1/documents/d/ask", {"question": "fees?"})
        assert got == status
        assert data["error"] == code
        assert "secret internal detail" not in json.dumps(data)
        assert "bad key" not in json.dumps(data)
        if code == "rate_limited":
            assert headers["retry-after"] == "5"
    finally:
        s.shutdown()
        s.server_close()


# -- the real command, end to end ---------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_the_real_command_serves_and_shuts_down_gracefully(tmp_path: Path) -> None:
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI"))}
    env["LONGDOC_API_KEY"] = TOKEN
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "longdoc.cli",
            "serve",
            "--port",
            str(port),
            "--store",
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("server did not start")
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        body = json.dumps({"text": CONTRACT, "id": "persisted"})
        c.request("POST", "/v1/documents/text", body, {**AUTH, "Content-Type": "application/json"})
        assert c.getresponse().status == 201
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 0  # clean exit, not killed
    finally:
        if proc.poll() is None:
            proc.kill()
    assert (tmp_path / "persisted.json").exists()  # the document survived on disk
