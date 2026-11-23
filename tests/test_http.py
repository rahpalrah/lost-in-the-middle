"""The standard-library HTTP layer, tested against a real local server (no mocks)."""

from __future__ import annotations

import json
import pickle
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from longdoc import ConfigurationError
from longdoc.http import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderServerError,
    Secret,
    check_url,
    post_json,
)

Reply = tuple[int, dict[str, str], bytes]


class Fake:
    """A scripted local HTTP server that records every request it receives."""

    def __init__(self) -> None:
        self.script: list[Reply] = []
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(self.rfile.read(length) or b"null"),
                    }
                )
                status, headers, body = outer.script.pop(0) if outer.script else (200, {}, b"{}")
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, status: int = 200, body: Any = None, **headers: str) -> None:
        raw = (
            body
            if isinstance(body, bytes)
            else json.dumps(body if body is not None else {}).encode()
        )
        self.script.append((status, headers, raw))


@pytest.fixture
def fake() -> Iterator[Fake]:
    f = Fake()
    yield f
    f.server.shutdown()
    f.server.server_close()


def call(fake: Fake, path: str = "/x", **kwargs: Any) -> Any:
    sleeps: list[float] = []
    kwargs.setdefault("sleep", sleeps.append)
    result = post_json(fake.url + path, {"X-Api-Key": "k"}, {"a": 1}, **kwargs)
    return result, sleeps


def test_success_sends_json_and_parses_response(fake: Fake) -> None:
    fake.reply(200, {"ok": True}, **{"request-id": "req_1"})
    (resp, sleeps) = call(fake)
    assert resp.body == {"ok": True}
    assert resp.request_id == "req_1"
    assert sleeps == []
    sent = fake.requests[0]
    assert sent["body"] == {"a": 1}
    assert sent["headers"]["x-api-key"] == "k"
    assert sent["headers"]["content-type"] == "application/json"
    assert sent["headers"]["user-agent"].startswith("longdoc-qa/")


def test_transient_errors_are_retried_with_backoff(fake: Fake) -> None:
    fake.reply(500, {"error": {"message": "boom"}})
    fake.reply(503, {})
    fake.reply(200, {"ok": 1})
    resp, sleeps = call(fake)
    assert resp.body == {"ok": 1}
    assert len(fake.requests) == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential


def test_retries_are_bounded(fake: Fake) -> None:
    for _ in range(5):
        fake.reply(502, {})
    with pytest.raises(ProviderServerError) as info:
        call(fake, max_retries=1)
    assert len(fake.requests) == 2
    assert info.value.status == 502


def test_retry_after_header_is_honoured_and_capped(fake: Fake) -> None:
    fake.reply(429, {"error": {"type": "rate_limit_error"}}, **{"retry-after": "7"})
    fake.reply(429, {}, **{"retry-after": "9999"})
    fake.reply(200, {})
    _, sleeps = call(fake, max_retries=2)
    assert 7 <= sleeps[0] < 8
    assert 30 <= sleeps[1] < 31


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}},
        {"error": {"code": "insufficient_quota", "message": "You exceeded your quota."}},
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Your credit balance is too low to access the API.",
            },
        },
    ],
)
def test_out_of_credit_is_a_distinct_error_and_never_retried(
    fake: Fake, body: dict[str, Any]
) -> None:
    fake.reply(
        429
        if "type" in body.get("error", {}) and body["error"]["type"] != "invalid_request_error"
        else 400,
        body,
    )
    with pytest.raises(ProviderQuotaError) as info:
        call(fake)
    assert len(fake.requests) == 1
    assert "retrying will not help" in str(info.value)


def test_ordinary_429_is_a_rate_limit_error(fake: Fake) -> None:
    for _ in range(3):
        fake.reply(429, {"error": {"type": "rate_limit_error", "message": "slow down"}})
    with pytest.raises(ProviderRateLimitError):
        call(fake)


def test_auth_error_never_exposes_the_providers_message(fake: Fake) -> None:
    echoed = "Incorrect API key provided: sk-abc***XYZ"
    fake.reply(401, {"error": {"message": echoed, "code": "invalid_api_key"}})
    with pytest.raises(ProviderAuthError) as info:
        call(fake)
    assert echoed not in str(info.value)
    assert "sk-abc" not in repr(info.value.__dict__)
    assert info.value.provider_message is None
    assert info.value.code == "invalid_api_key"
    assert len(fake.requests) == 1  # not retried


def test_other_client_errors_are_not_retried(fake: Fake) -> None:
    fake.reply(400, {"error": {"message": "bad field"}})
    with pytest.raises(ProviderBadRequestError, match="bad field"):
        call(fake)
    fake.reply(403, {"error": {"message": "no access"}})
    with pytest.raises(ProviderPermissionError):
        call(fake)
    assert len(fake.requests) == 2


def test_non_json_and_non_object_responses(fake: Fake) -> None:
    fake.reply(200, b"<html>oops</html>")
    with pytest.raises(ProviderServerError, match="not valid JSON"):
        call(fake)
    fake.reply(200, b"[1, 2]")
    with pytest.raises(ProviderServerError, match="unexpected"):
        call(fake)


def test_unreachable_server_is_a_connection_error() -> None:
    sleeps: list[float] = []
    with pytest.raises(ProviderConnectionError):
        post_json("http://127.0.0.1:9/x", {}, {}, max_retries=1, sleep=sleeps.append, timeout=2)
    assert len(sleeps) == 1


def test_redirects_are_not_followed_so_keys_cannot_be_replayed(fake: Fake) -> None:
    fake.reply(307, b"", location=fake.url + "/elsewhere")
    with pytest.raises(ProviderBadRequestError):
        call(fake)
    assert [r["path"] for r in fake.requests] == ["/x"]  # /elsewhere was never contacted


def test_plain_http_to_remote_hosts_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="plain HTTP"):
        check_url("http://api.example.com/v1")
    check_url("http://127.0.0.1:8000/v1")
    check_url("http://localhost/v1")
    check_url("http://10.0.0.5/v1", allow_insecure=True)
    check_url("https://api.example.com/v1")
    with pytest.raises(ConfigurationError):
        check_url("ftp://example.com")


def test_secret_hides_its_value() -> None:
    secret = Secret("super-secret-value")
    assert "super-secret" not in repr(secret)
    assert "super-secret" not in str(secret)
    assert "super-secret" not in f"{secret}"
    assert "super-secret" not in repr({"k": secret})
    assert secret.reveal() == "super-secret-value"
    assert bool(secret)
    assert not Secret("")
    with pytest.raises(TypeError):
        pickle.dumps(secret)
