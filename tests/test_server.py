from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx2  # the HTTP library the anthropic SDK uses
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from longdoc import DocumentService, RefusalError
from longdoc.server import create_app, create_router
from longdoc.server.asgi import app_from_env, config_from_env

from .conftest import FakeClient, make_contract

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def build_service(*responses: Any) -> DocumentService:
    return DocumentService(
        client=FakeClient(*responses),  # type: ignore[arg-type]
        async_client=FakeClient(*responses, is_async=True),  # type: ignore[arg-type]
    )


@pytest.fixture
def service() -> DocumentService:
    return build_service()


@pytest.fixture
def api(service: DocumentService) -> Iterator[TestClient]:
    with TestClient(create_app(service, api_key=TOKEN, max_upload_bytes=2_000_000)) as client:
        yield client


def test_health_is_public(api: TestClient) -> None:
    assert api.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_auth_required(api: TestClient, headers: dict[str, str]) -> None:
    r = api.get("/v1/documents", headers=headers)
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_text_document_flow(api: TestClient) -> None:
    text = make_contract().text
    r = api.post("/v1/documents/text", json={"text": text, "id": "c1"}, headers=AUTH)
    assert r.status_code == 201, r.text
    assert r.json()["id"] == "c1"

    assert [d["id"] for d in api.get("/v1/documents", headers=AUTH).json()] == ["c1"]
    assert api.get("/v1/documents/c1", headers=AUTH).json()["sections"] > 1

    r = api.post("/v1/documents/c1/retrieve", json={"question": "termination fee"}, headers=AUTH)
    assert r.status_code == 200
    assert any("90 days" in s["text"] for s in r.json())

    r = api.post("/v1/documents/c1/ask", json={"question": "Notice period?"}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]
    assert body["usage"]["input_tokens"] == 100
    assert body["context"][0]["text"] is None  # text omitted unless requested

    r = api.post(
        "/v1/documents/c1/ask",
        json={"question": "Notice?", "strategy": "cached_document", "include_context_text": True},
        headers=AUTH,
    )
    assert r.json()["context"][0]["text"]

    assert api.delete("/v1/documents/c1", headers=AUTH).status_code == 204
    assert api.delete("/v1/documents/c1", headers=AUTH).status_code == 404
    assert api.get("/v1/documents/c1", headers=AUTH).json()["error"] == "not_found"


def test_file_upload(api: TestClient) -> None:
    files = {"file": ("notes.md", b"# Title\n\nThe fee is 500 dollars.", "text/markdown")}
    r = api.post("/v1/documents", files=files, data={"id": "notes"}, headers=AUTH)
    assert r.status_code == 201, r.text
    assert r.json()["source"] == "notes.md"


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("sheet.xlsx", b"x", 415, "unsupported_format"),
        ("empty.txt", b"   ", 422, "invalid_document"),
        ("big.txt", b"x" * 2_000_001, 413, "too_large"),
    ],
)
def test_upload_errors(
    api: TestClient, filename: str, content: bytes, status: int, code: str
) -> None:
    r = api.post("/v1/documents", files={"file": (filename, content)}, headers=AUTH)
    assert r.status_code == status
    assert r.json()["error"] == code


def test_validation_errors(api: TestClient) -> None:
    assert api.post("/v1/documents/text", json={"text": ""}, headers=AUTH).status_code == 422
    r = api.post("/v1/documents/text", json={"text": "a", "id": "bad id"}, headers=AUTH)
    assert r.json()["error"] == "invalid_request"
    assert (
        api.post("/v1/documents/nope/ask", json={"question": "x"}, headers=AUTH).status_code == 404
    )


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, headers={"request-id": "req_up"})
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (RefusalError(category="cyber"), 422, "refused"),
        (_status_error(anthropic.RateLimitError, 429), 429, "rate_limited"),
        (_status_error(anthropic.InternalServerError, 500), 502, "upstream_error"),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            503,
            "upstream_unavailable",
        ),
    ],
)
def test_upstream_errors_are_mapped(
    service: DocumentService,
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    status: int,
    code: str,
) -> None:
    service.add_text("Some text about fees.", document_id="d")

    async def boom(*_: Any, **__: Any) -> Any:
        raise exc

    monkeypatch.setattr(service, "aask", boom)
    r = api.post("/v1/documents/d/ask", json={"question": "fees?"}, headers=AUTH)
    assert r.status_code == status
    assert r.json()["error"] == code
    assert "boom" not in r.text  # upstream messages are not leaked


def test_router_mounts_into_existing_app(service: DocumentService) -> None:
    app = FastAPI()
    app.include_router(create_router(service), prefix="/longdoc")
    with TestClient(app) as client:
        r = client.post("/longdoc/v1/documents/text", json={"text": "Hello there.", "id": "h"})
        assert r.status_code == 201
        assert client.get("/longdoc/v1/documents/h").json()["id"] == "h"


def test_openapi_schema(api: TestClient) -> None:
    schema = api.get("/openapi.json").json()
    assert schema["info"]["license"]["name"] == "MIT"
    assert "/v1/documents/{document_id}/ask" in schema["paths"]


def test_config_from_env() -> None:
    cfg = config_from_env(
        {
            "LONGDOC_MODEL": "claude-sonnet-5-5",
            "LONGDOC_EFFORT": "high",
            "LONGDOC_STRATEGY": "cached_document",
            "LONGDOC_CACHE_TTL": "1h",
            "LONGDOC_FALLBACKS": "none",
        }
    )
    assert cfg.model == "claude-sonnet-5-5"
    assert cfg.fallbacks is None
    assert cfg.cache_ttl == "1h"


def test_app_from_env_uses_file_store(tmp_path: Path) -> None:
    app = app_from_env({"LONGDOC_STORE_DIR": str(tmp_path), "LONGDOC_API_KEY": "k"})
    with TestClient(app) as client:
        r = client.post(
            "/v1/documents/text",
            json={"text": "Persist me.", "id": "p"},
            headers={"Authorization": "Bearer k"},
        )
        assert r.status_code == 201
    assert (tmp_path / "p.json").exists()
