"""Self-service keys: registration, verification, auth, usage, owner alerts, admin."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from longdoc import DocumentService
from longdoc.accounts import (
    AccountError,
    AccountService,
    AccountStore,
    Notifier,
    email_kind,
    normalise_email,
)
from longdoc.api import LongDocApi
from longdoc.server import create_server


class FakeMailer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, text: str) -> bool:
        self.sent.append((to, subject, text))
        return True

    def token(self) -> str:
        match = re.search(r"token=([\w\-]+)", self.sent[-1][2])
        assert match
        return match.group(1)


def make(tmp_path: Path, **kwargs: Any) -> tuple[LongDocApi, FakeMailer, AccountService]:
    mailer = FakeMailer()
    service = AccountService(
        store=AccountStore(tmp_path / "accounts.json"),
        notifier=Notifier(events_file=str(tmp_path / "events.log")),
        mailer=mailer,  # type: ignore[arg-type]
        public_url="http://test",
        admin_token="admin-secret",
        **kwargs,
    )
    return LongDocApi(DocumentService(), accounts=service), mailer, service


def call(api: LongDocApi, method: str, path: str, body: Any = None, token: str | None = None):
    headers = {"authorization": f"Bearer {token}"} if token else {}
    raw = b""
    if body is not None:
        raw = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    return api.handle(method, path, headers, raw, "1.2.3.4")


def register_and_verify(api: LongDocApi, mailer: FakeMailer, email: str) -> str:
    assert call(api, "POST", "/v1/register", {"email": email}).status == 202
    page = call(api, "GET", f"/verify?token={mailer.token()}")
    assert page.status == 200
    match = re.search(r"lmd_[\w\-]+", page.body.decode())
    assert match
    return match.group(0)


def test_email_validation_and_kind() -> None:
    assert normalise_email("  Ada@Example.COM ") == "ada@example.com"
    for bad in ("", "no-at", "a@b", "a b@c.com", "x" * 300 + "@a.com", "a@@b.com"):
        with pytest.raises(AccountError):
            normalise_email(bad)
    assert email_kind("a@gmail.com") == "personal"
    assert email_kind("a@acme.io") == "business"


def test_register_verify_use_flow(tmp_path: Path) -> None:
    api, mailer, service = make(tmp_path)
    assert call(api, "GET", "/v1/documents").status == 401  # a key is now required
    key = register_and_verify(api, mailer, "Ada@Acme.io")
    assert call(api, "GET", "/v1/documents", token=key).status == 200
    assert (
        call(api, "POST", "/v1/documents/text", {"text": "# A\n\nfee 5", "id": "d"}, key).status
        == 201
    )
    assert call(api, "POST", "/v1/documents/d/ask", {"question": "fee?"}, key).status == 200
    row = service.store.list()[0]
    assert (row["email"], row["kind"], row["status"]) == ("ada@acme.io", "business", "active")
    assert (row["requests"], row["documents"], row["questions"]) == (3, 1, 1)
    assert row["key_prefix"] == key[:8]
    assert "key_hash" not in row and "verify_hash" not in row


def test_key_is_shown_once_and_never_stored(tmp_path: Path) -> None:
    api, mailer, _ = make(tmp_path)
    key = register_and_verify(api, mailer, "ada@acme.io")
    token = mailer.token()
    stored = (tmp_path / "accounts.json").read_text()
    assert key not in stored and token not in stored  # only hashes are on disk
    again = call(api, "GET", f"/verify?token={token}")
    assert again.status == 400 and key not in again.body.decode()  # single use


def test_invalid_and_expired_links(tmp_path: Path) -> None:
    api, mailer, service = make(tmp_path)
    assert call(api, "GET", "/verify?token=nope").status == 400
    assert call(api, "GET", "/verify").status == 400
    call(api, "POST", "/v1/register", {"email": "ada@acme.io"})
    service.store._accounts["ada@acme.io"]["verify_expires"] = 1.0  # long ago
    assert call(api, "GET", f"/verify?token={mailer.token()}").status == 400


def test_registration_does_not_reveal_whether_an_address_exists(tmp_path: Path) -> None:
    api, mailer, _ = make(tmp_path)
    register_and_verify(api, mailer, "ada@acme.io")
    sent_before = len(mailer.sent)
    existing = call(api, "POST", "/v1/register", {"email": "ada@acme.io"})
    new = call(api, "POST", "/v1/register", {"email": "bob@acme.io"})
    assert existing.status == new.status == 202
    assert existing.body == new.body  # same message
    assert len(mailer.sent) == sent_before + 1  # but no second key email for the active account


def test_invalid_registration_input(tmp_path: Path) -> None:
    api, _, _ = make(tmp_path)
    assert call(api, "POST", "/v1/register", {"email": "nope"}).status == 422
    assert call(api, "POST", "/v1/register", ["x"]).status == 400
    assert call(api, "POST", "/v1/register").status == 400


def test_registration_is_rate_limited(tmp_path: Path) -> None:
    api, _, _ = make(tmp_path)
    codes = [
        call(api, "POST", "/v1/register", {"email": f"u{i}@acme.io"}).status for i in range(12)
    ]
    assert codes[:10] == [202] * 10 and 429 in codes[10:]


def test_master_key_still_works_alongside_account_keys(tmp_path: Path) -> None:
    _, mailer, service = make(tmp_path)
    api = LongDocApi(DocumentService(), api_key="master-key", accounts=service)
    assert call(api, "GET", "/v1/documents", token="master-key").status == 200
    assert call(api, "GET", "/v1/documents", token="wrong").status == 401
    key = register_and_verify(api, mailer, "ada@acme.io")
    assert call(api, "GET", "/v1/documents", token=key).status == 200


def test_revoked_keys_stop_working(tmp_path: Path) -> None:
    api, mailer, _ = make(tmp_path)
    key = register_and_verify(api, mailer, "ada@acme.io")
    assert (
        call(api, "POST", "/v1/admin/accounts/ada%40acme.io/revoke", token="admin-secret").status
        == 200
    )
    assert call(api, "GET", "/v1/documents", token=key).status == 401
    # and a revoked address cannot simply re-register for a new key
    call(api, "POST", "/v1/register", {"email": "ada@acme.io"})
    assert len(mailer.sent) == 1
    assert (
        call(api, "POST", "/v1/admin/accounts/nobody%40x.io/revoke", token="admin-secret").status
        == 404
    )


def test_admin_requires_its_own_token(tmp_path: Path) -> None:
    api, mailer, _ = make(tmp_path)
    key = register_and_verify(api, mailer, "ada@acme.io")
    assert call(api, "GET", "/v1/admin/accounts").status == 401
    assert (
        call(api, "GET", "/v1/admin/accounts", token=key).status == 401
    )  # a user key is not admin
    rows = json.loads(call(api, "GET", "/v1/admin/accounts", token="admin-secret").body)
    assert rows[0]["email"] == "ada@acme.io" and rows[0]["requests"] == 0
    no_admin = LongDocApi(
        DocumentService(),
        accounts=AccountService(
            store=AccountStore(), notifier=Notifier(), mailer=None, public_url="http://t"
        ),
    )
    assert call(no_admin, "GET", "/v1/admin/accounts", token="anything").status == 404


def test_disabled_when_accounts_are_not_configured() -> None:
    api = LongDocApi(DocumentService())
    for method, path in (
        ("GET", "/register"),
        ("POST", "/v1/register"),
        ("GET", "/verify?token=x"),
        ("GET", "/admin"),
    ):
        assert api.handle(method, path, {}, b"{}", "c").status == 404


def test_owner_is_notified_of_signup_key_issue_and_first_use(tmp_path: Path) -> None:
    api, mailer, _ = make(tmp_path)
    key = register_and_verify(api, mailer, "ada@gmail.com")
    call(api, "GET", "/v1/documents", token=key)
    call(api, "GET", "/v1/documents", token=key)
    log = (tmp_path / "events.log").read_text().splitlines()
    titles = [line.split(" | ")[1] for line in log]
    assert titles == ["longdoc: new signup", "longdoc: API key issued", "longdoc: first use"]
    assert "ada@gmail.com (personal)" in log[0]
    assert key not in "".join(log) and mailer.token() not in "".join(log)  # no secrets in alerts


def test_pages_escape_input_and_are_locked_down(tmp_path: Path) -> None:
    api, _, _ = make(tmp_path)
    for path in ("/register", "/admin"):
        page = call(api, "GET", path)
        assert (
            page.status == 200
            and "frame-ancestors 'none'" in page.headers["content-security-policy"]
        )
        assert (
            page.headers["referrer-policy"] == "no-referrer"
            and page.headers["cache-control"] == "no-store"
        )
    bad = call(api, "GET", "/verify?token=%3Cscript%3Ealert(1)%3C/script%3E")
    assert "<script>alert" not in bad.body.decode()


def test_accounts_persist_across_restarts(tmp_path: Path) -> None:
    api, mailer, service = make(tmp_path)
    key = register_and_verify(api, mailer, "ada@acme.io")
    call(api, "GET", "/v1/documents", token=key)
    service.flush()
    reborn = LongDocApi(DocumentService(), accounts=make(tmp_path)[2])
    assert call(reborn, "GET", "/v1/documents", token=key).status == 200
    assert reborn.accounts is not None and reborn.accounts.store.list()[0]["requests"] == 2


def test_from_env_is_off_by_default_and_validates() -> None:
    assert AccountService.from_env({}) is None
    on = AccountService.from_env({"LONGDOC_ACCOUNTS": "1", "LONGDOC_PUBLIC_URL": "https://x.io/"})
    assert on is not None and on.public_url == "https://x.io" and on.mailer is None
    from longdoc import ConfigurationError

    with pytest.raises(ConfigurationError):
        AccountService.from_env({"LONGDOC_ACCOUNTS": "1", "LONGDOC_GITHUB_REPO": "not a repo"})


def test_dev_mode_returns_the_link_only_when_enabled(tmp_path: Path) -> None:
    api, _, _ = make(tmp_path, dev_mode=True)
    assert "verify_url" in json.loads(
        call(api, "POST", "/v1/register", {"email": "a@acme.io"}).body
    )
    api2, _, _ = make(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path)
    assert "verify_url" not in json.loads(
        call(api2, "POST", "/v1/register", {"email": "b@acme.io"}).body
    )


def test_end_to_end_over_a_real_socket(tmp_path: Path) -> None:
    mailer = FakeMailer()
    service = AccountService(
        store=AccountStore(tmp_path / "a.json"),
        notifier=Notifier(),
        mailer=mailer,  # type: ignore[arg-type]
        public_url="http://127.0.0.1",
        admin_token="adm",
    )
    server = create_server(DocumentService(), port=0, accounts=service)
    server.start_in_thread()

    def req(
        method: str, path: str, body: Any = None, token: str | None = None
    ) -> tuple[int, bytes]:
        headers = {"authorization": f"Bearer {token}"} if token else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        r = urllib.request.Request(server.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    try:
        assert req("GET", "/register")[0] == 200
        assert req("POST", "/v1/register", {"email": "ada@acme.io"})[0] == 202
        status, page = req("GET", f"/verify?token={mailer.token()}")
        key = re.search(rb"lmd_[\w\-]+", page).group(0).decode()  # type: ignore[union-attr]
        assert status == 200
        assert (
            req(
                "POST",
                "/v1/documents/text",
                {"text": "# A\n\nThe fee is 5 dollars.", "id": "d"},
                key,
            )[0]
            == 201
        )
        status, body = req("POST", "/v1/documents/d/ask", {"question": "What is the fee?"}, key)
        assert status == 200 and "5 dollars" in json.loads(body)["text"]
        assert req("GET", "/v1/documents")[0] == 401
    finally:
        server.shutdown()
        server.server_close()
    assert threading.active_count() >= 1
