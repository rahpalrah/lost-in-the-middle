"""Self-service API keys: registration, email verification, usage tracking and owner alerts.

Flow:

1. A person enters an email address on ``/register``.
2. The server emails a one-time verification link (valid 24 hours).
3. Opening the link issues an API key. It is shown **once**; only its SHA-256 hash is stored.
4. The key is used as ``Authorization: Bearer <key>``. Each account's request, upload and
   question counts are recorded, and the owner is alerted on signup, key issue and first use.

Security notes:

* Tokens are 256-bit random values from :mod:`secrets`. A SHA-256 hash is enough for them (no
  password hashing needed), and the account file contains no usable secret.
* Registration answers the same way whether or not the address already exists, so the form cannot
  be used to discover who has an account.
* Registering again for an active address does not issue a second key.
* Verification links are single-use and expire.

This is a file-backed store for one server (or a shared volume). For several independent servers,
replace :class:`AccountStore` with a shared database; the interface is small.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import smtplib
import ssl
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError
from .http import _OPENER

logger = logging.getLogger(__name__)

KEY_PREFIX = "lmd_"
VERIFY_TTL_SECONDS = 24 * 3600
_EMAIL = re.compile(r"^[a-z0-9._%+\-]{1,64}@[a-z0-9.\-]{1,253}\.[a-z]{2,63}$")
_REPO = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
PERSONAL_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "msn.com",
        "icloud.com",
        "me.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
        "gmx.com",
        "yandex.com",
        "zoho.com",
    }
)
# Fields shown to the owner. Hashes and verification tokens are never listed.
_PUBLIC_FIELDS = (
    "email",
    "kind",
    "status",
    "created_at",
    "verified_at",
    "key_prefix",
    "requests",
    "documents",
    "questions",
    "last_used",
)


class AccountError(ValueError):
    """A registration or verification request that cannot be honoured (message is safe to show)."""


def normalise_email(raw: str) -> str:
    """Lower-case and validate an email address. Raises :class:`AccountError` if invalid."""
    email = (raw or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        raise AccountError("Enter a valid email address.")
    return email


def email_kind(email: str) -> str:
    """``"personal"`` for common consumer mail providers, otherwise ``"business"``."""
    return "personal" if email.rsplit("@", 1)[1] in PERSONAL_DOMAINS else "business"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    return (
        None
        if ts is None
        else datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")
    )


class AccountStore:
    """Accounts keyed by email, optionally persisted to a JSON file (written atomically)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.RLock()
        self._accounts: dict[str, dict[str, Any]] = {}
        self._by_key: dict[str, str] = {}  # key hash -> email
        self._last_write = 0.0
        self._dirty = False
        if self._path is not None and self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8") or "{}")
                self._accounts = dict(data.get("accounts", {}))
            except (OSError, ValueError) as exc:
                raise ConfigurationError(f"Cannot read account file {self._path}: {exc}") from exc
            for email, rec in self._accounts.items():
                if rec.get("key_hash") and rec.get("status") == "active":
                    self._by_key[rec["key_hash"]] = email

    # -- registration and verification ----------------------------------------------------------

    def begin(self, email: str) -> tuple[bool, str | None]:
        """Create a pending account or issue a fresh verification token.

        Returns ``(is_new, token)``. ``token`` is ``None`` when no email should be sent (the
        address already has an active or revoked account).
        """
        with self._lock:
            rec = self._accounts.get(email)
            is_new = rec is None
            if rec is None:
                rec = {
                    "email": email,
                    "kind": email_kind(email),
                    "status": "pending",
                    "created_at": _now(),
                    "verified_at": None,
                    "key_hash": None,
                    "key_prefix": None,
                    "requests": 0,
                    "documents": 0,
                    "questions": 0,
                    "last_used": None,
                    "verify_hash": None,
                    "verify_expires": None,
                }
                self._accounts[email] = rec
            elif rec["status"] != "pending":
                return False, None
            token = secrets.token_urlsafe(32)
            rec["verify_hash"] = _sha256(token)
            rec["verify_expires"] = _now() + VERIFY_TTL_SECONDS
            self._save()
            return is_new, token

    def verify(self, token: str) -> tuple[str, str]:
        """Consume a verification token and issue the key. Returns ``(email, key)``."""
        digest = _sha256(token)
        with self._lock:
            rec = next(
                (
                    r
                    for r in self._accounts.values()
                    if r.get("verify_hash") and secrets.compare_digest(r["verify_hash"], digest)
                ),
                None,
            )
            if rec is None or rec["status"] != "pending" or _now() > (rec["verify_expires"] or 0):
                raise AccountError(
                    "This verification link is invalid or has expired. Register again."
                )
            key = KEY_PREFIX + secrets.token_urlsafe(32)
            rec.update(
                status="active",
                verified_at=_now(),
                key_hash=_sha256(key),
                key_prefix=key[:8],
                verify_hash=None,
                verify_expires=None,
            )
            self._by_key[rec["key_hash"]] = rec["email"]
            self._save()
            return rec["email"], key

    # -- authentication and usage ---------------------------------------------------------------

    def authenticate(self, key: str) -> str | None:
        """Return the account's email for an active key, else ``None``."""
        if not key.startswith(KEY_PREFIX):
            return None
        email = self._by_key.get(_sha256(key))
        if email is None or self._accounts[email]["status"] != "active":
            return None
        return email

    def record_use(self, email: str, action: str) -> bool:
        """Count one request. Returns ``True`` if this was the account's first request."""
        with self._lock:
            rec = self._accounts[email]
            first = rec["requests"] == 0
            rec["requests"] += 1
            if action == "upload":
                rec["documents"] += 1
            elif action == "ask":
                rec["questions"] += 1
            rec["last_used"] = _now()
            self._save(debounce=True)
            return bool(first)

    def revoke(self, email: str) -> bool:
        """Disable an account's key. Returns ``False`` if the account does not exist."""
        with self._lock:
            rec = self._accounts.get(email)
            if rec is None:
                return False
            if rec.get("key_hash"):
                self._by_key.pop(rec["key_hash"], None)
            rec.update(status="revoked", key_hash=None, key_prefix=None, verify_hash=None)
            self._save()
            return True

    def list(self) -> list[dict[str, Any]]:
        """All accounts, newest first, without any secrets."""
        with self._lock:
            rows = []
            for rec in sorted(self._accounts.values(), key=lambda r: r["created_at"], reverse=True):
                row = {k: rec.get(k) for k in _PUBLIC_FIELDS}
                row["created_at"] = _iso(rec["created_at"])
                row["verified_at"] = _iso(rec.get("verified_at"))
                row["last_used"] = _iso(rec.get("last_used"))
                rows.append(row)
            return rows

    def flush(self) -> None:
        """Write any pending usage counts to disk."""
        with self._lock:
            if self._dirty:
                self._save()

    def _save(self, *, debounce: bool = False) -> None:
        if self._path is None:
            return
        now = time.monotonic()
        if debounce and now - self._last_write < 1.0:
            self._dirty = True  # written by the next change or flush(), at most ~1s late
            return
        self._last_write = now
        self._dirty = False
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "accounts": self._accounts}, indent=2), "utf-8")
        with contextlib.suppress(OSError):  # chmod is not supported everywhere (e.g. Windows)
            os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)


class Mailer:
    """Sends mail over SMTP with STARTTLS (standard library only)."""

    def __init__(self, *, host: str, port: int, user: str, password: str, sender: str) -> None:
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender = sender

    def send(self, to: str, subject: str, text: str) -> bool:
        """Send one message. Returns ``False`` (and logs) on failure; never raises."""
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(text)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                if self.user:
                    smtp.login(self.user, self.password)
                smtp.send_message(msg)
            return True
        except (smtplib.SMTPException, OSError) as exc:
            logger.warning("Email delivery failed (%s).", type(exc).__name__)
            return False


class Notifier:
    """Tells the owner about signups, key issues and first use.

    Every event is logged and appended to ``events_file`` (if set). If ``github_repo`` and
    ``github_token`` are set, an issue is opened in that repository. Use a **private** repository:
    issues contain email addresses. If ``owner_email`` and a mailer are set, the owner is emailed.
    Delivery runs in background threads and never blocks or fails a request.
    """

    def __init__(
        self,
        *,
        events_file: str | None = None,
        github_repo: str | None = None,
        github_token: str | None = None,
        mailer: Mailer | None = None,
        owner_email: str | None = None,
    ) -> None:
        if github_repo and not _REPO.match(github_repo):
            raise ConfigurationError("LONGDOC_GITHUB_REPO must look like owner/name.")
        self._events = Path(events_file) if events_file else None
        self._repo = github_repo
        self._token = github_token
        self._mailer = mailer
        self._owner = owner_email

    def notify(self, title: str, body: str) -> None:
        """Record an event and alert the owner through every configured channel."""
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info("%s: %s", title, body)
        if self._events is not None:
            try:
                with self._events.open("a", encoding="utf-8") as fh:
                    fh.write(f"{stamp} | {title} | {body}\n")
            except OSError as exc:
                logger.warning("Cannot write events file (%s).", type(exc).__name__)
        if self._repo and self._token:
            threading.Thread(target=self._github_issue, args=(title, body), daemon=True).start()
        if self._mailer is not None and self._owner:
            threading.Thread(
                target=self._mailer.send, args=(self._owner, title, body), daemon=True
            ).start()

    def _github_issue(self, title: str, body: str) -> None:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"https://api.github.com/repos/{self._repo}/issues",
            data=json.dumps({"title": title, "body": body}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "longdoc-qa",
            },
            method="POST",
        )
        try:
            with _OPENER.open(req, timeout=10) as resp:
                resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("GitHub notification failed (%s).", type(exc).__name__)


@dataclass
class AccountService:
    """Registration, verification, usage and notifications, wired together."""

    store: AccountStore
    notifier: Notifier
    mailer: Mailer | None
    public_url: str
    admin_token: str | None = None
    dev_mode: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> AccountService | None:
        """Build from ``LONGDOC_*`` variables. Returns ``None`` unless accounts are enabled."""
        enabled = env.get("LONGDOC_ACCOUNTS", "").lower() in ("1", "true", "yes")
        if not (enabled or env.get("LONGDOC_ACCOUNTS_FILE")):
            return None
        mailer = None
        if host := env.get("LONGDOC_SMTP_HOST"):
            user = env.get("LONGDOC_SMTP_USER", "")
            mailer = Mailer(
                host=host,
                port=int(env.get("LONGDOC_SMTP_PORT", "587")),
                user=user,
                password=env.get("LONGDOC_SMTP_PASSWORD", ""),
                sender=env.get("LONGDOC_SMTP_FROM") or user,
            )
        notifier = Notifier(
            events_file=env.get("LONGDOC_EVENTS_FILE") or None,
            github_repo=env.get("LONGDOC_GITHUB_REPO") or None,
            github_token=env.get("LONGDOC_GITHUB_TOKEN") or None,
            mailer=mailer,
            owner_email=env.get("LONGDOC_OWNER_EMAIL") or None,
        )
        return cls(
            store=AccountStore(env.get("LONGDOC_ACCOUNTS_FILE") or None),
            notifier=notifier,
            mailer=mailer,
            public_url=env.get("LONGDOC_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/"),
            admin_token=env.get("LONGDOC_ADMIN_TOKEN") or None,
            dev_mode=env.get("LONGDOC_ACCOUNTS_DEV", "").lower() in ("1", "true", "yes"),
        )

    def start_registration(self, raw_email: str) -> dict[str, Any]:
        """Begin registration. The reply is identical whether or not the address is known."""
        email = normalise_email(raw_email)
        is_new, token = self.store.begin(email)
        result: dict[str, Any] = {
            "status": "check_email",
            "message": "If this address can register, a verification link has been sent. "
            "The link expires in 24 hours.",
        }
        if token is not None:
            url = f"{self.public_url}/verify?token={token}"
            sent = False
            if self.mailer is not None:
                sent = self.mailer.send(
                    email,
                    "Verify your longdoc API key",
                    "Open this link once to get your API key. It is shown only once.\n\n"
                    f"{url}\n\nIf you did not request this, ignore this email.",
                )
            if not sent:
                logger.warning("Verification link for a new registration (no email sent): %s", url)
            if is_new:
                kind = email_kind(email)
                self.notifier.notify(
                    "longdoc: new signup",
                    f"{email} ({kind}) registered. Verification email sent: {sent}.",
                )
            if self.dev_mode:
                result["verify_url"] = url  # only when LONGDOC_ACCOUNTS_DEV is set
        return result

    def complete_verification(self, token: str) -> tuple[str, str]:
        """Consume a verification link and return ``(email, key)``; the key is shown once."""
        email, key = self.store.verify(token)
        self.notifier.notify(
            "longdoc: API key issued", f"{email} ({email_kind(email)}) verified and received a key."
        )
        return email, key

    def authenticate(self, key: str) -> str | None:
        """The account email for an active key, else ``None``."""
        return self.store.authenticate(key)

    def record_use(self, email: str, action: str) -> None:
        """Count a request and alert the owner on the account's first one."""
        if self.store.record_use(email, action):
            self.notifier.notify(
                "longdoc: first use",
                f"{email} ({email_kind(email)}) made its first request ({action}).",
            )

    def revoke(self, email: str) -> bool:
        """Revoke an account's key and alert the owner."""
        ok = self.store.revoke(email)
        if ok:
            self.notifier.notify("longdoc: key revoked", f"{email} was revoked by the owner.")
        return ok

    def flush(self) -> None:
        """Persist pending usage counts."""
        self.store.flush()
