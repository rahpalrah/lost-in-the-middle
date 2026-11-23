"""The standard-library Python client, against the real built-in server."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from longdoc import ConfigurationError
from longdoc.remote import RemoteClient, RemoteError, encode_multipart

from .test_api_server import CONTRACT, TOKEN, start
from .test_pdf import simple_pdf


@pytest.fixture
def server() -> Iterator[Any]:
    s = start(rate_limit_per_minute=100)
    yield s
    s.shutdown()
    s.server_close()


@pytest.fixture
def client(server: Any) -> RemoteClient:
    host, port = server.socket.getsockname()[:2]
    return RemoteClient(f"http://{host}:{port}", api_key=TOKEN, max_retries=0)


def test_full_workflow(client: RemoteClient, tmp_path: Path) -> None:
    assert client.health()["status"] == "ok"

    path = tmp_path / "contract.md"
    path.write_text(CONTRACT, encoding="utf-8")
    info = client.upload(path, document_id="c1")
    assert (info["id"], info["sections"], info["source"]) == ("c1", 2, "contract.md")
    assert (
        client.add_text("Fees are due in 30 days.", document_id="t1", source="inline")["id"] == "t1"
    )
    assert [d["id"] for d in client.list_documents()] == ["c1", "t1"]
    assert client.get_document("c1")["sections"] == 2

    answer = client.ask("c1", "How much notice is needed to terminate?")
    assert "ninety (90) days" in answer["text"]
    assert answer["citations"][0]["heading"] == "# 19. Termination"
    assert "text" in client.ask("c1", "Which law governs?", include_context_text=True)["context"][0]
    assert "New York" in client.ask("c1", "Which law governs?", strategy="cached_document")["text"]
    assert "New York" in client.retrieve("c1", "governing law")[0]["text"]

    client.delete_document("t1")
    with pytest.raises(RemoteError) as info_err:
        client.get_document("t1")
    assert (info_err.value.status, info_err.value.code) == (404, "not_found")


def test_upload_bytes_and_pdf(client: RemoteClient) -> None:
    pdf = simple_pdf("BT /F1 12 Tf 72 700 Td (The notice period is sixty days.) Tj ET")
    assert client.upload(pdf, filename="n.pdf", document_id="p")["pages"] == 1
    assert "sixty days" in client.ask("p", "What is the notice period?")["text"]
    with pytest.raises(ConfigurationError, match="filename"):
        client.upload(b"bytes without a name")


def test_errors_carry_status_code_and_details(client: RemoteClient) -> None:
    with pytest.raises(RemoteError) as e1:
        client.ask("nope", "q?")
    assert (e1.value.status, e1.value.code) == (404, "not_found")
    with pytest.raises(RemoteError) as e2:
        client.add_text("")
    assert (e2.value.status, e2.value.code) == (422, "invalid_request")
    with pytest.raises(RemoteError) as e3:
        client.upload(b"x", filename="sheet.xlsx")
    assert e3.value.status == 415
    bad = RemoteClient(client.base_url, api_key="wrong", max_retries=0)
    with pytest.raises(RemoteError) as e4:
        bad.list_documents()
    assert e4.value.status == 401


def test_ids_with_special_characters_are_url_encoded(client: RemoteClient) -> None:
    with pytest.raises(RemoteError) as exc:
        client.get_document("../secret")
    assert exc.value.status in (404, 422)  # never interpreted as a path


def test_unreachable_server_and_retries() -> None:
    dead = RemoteClient("http://127.0.0.1:9", timeout=2, max_retries=1)
    with pytest.raises(RemoteError, match="Could not reach"):
        dead.health()


def test_credentials_are_never_shown(client: RemoteClient) -> None:
    assert TOKEN not in repr(client)
    assert TOKEN not in repr(vars(client))


def test_plain_http_to_remote_hosts_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="plain HTTP"):
        RemoteClient("http://api.example.com")
    RemoteClient("https://api.example.com")
    RemoteClient("http://10.0.0.5:8000", allow_insecure=True)
    monkeypatch.setenv("LONGDOC_API_KEY", TOKEN)
    assert RemoteClient("http://127.0.0.1:1")._key is not None  # picked up from the environment


def test_multipart_encoding_is_safe() -> None:
    body, ctype = encode_multipart({"id": "a"}, {"file": ('we"ird\r\nname.txt', b"data")})
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'filename="we_ird__name.txt"' in body
    assert b"\r\nname" not in body.split(b"filename=")[1].split(b"\r\n")[0]
