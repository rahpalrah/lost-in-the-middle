# Self-service API keys

People register with an email address and receive their own API key. You, the owner, get an alert
when someone signs up, when a key is issued, and when it is first used, and can see every account
and its usage on an admin page. Off by default; no extra packages needed.

## Turn it on

```bash
export LONGDOC_ACCOUNTS_FILE=/data/accounts.json      # persists accounts (also enables the feature)
export LONGDOC_PUBLIC_URL=https://your-domain         # used in the emailed link
export LONGDOC_ADMIN_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
longdoc serve --host 0.0.0.0 --port 8000
```

Pages: `/register` (sign-up), `/verify?token=...` (shows the key once), `/admin` (you).

## Email (verification link)

Set an SMTP server and the link is emailed:

| Variable | Meaning |
|---|---|
| `LONGDOC_SMTP_HOST`, `LONGDOC_SMTP_PORT` (587) | SMTP server (STARTTLS) |
| `LONGDOC_SMTP_USER`, `LONGDOC_SMTP_PASSWORD` | credentials |
| `LONGDOC_SMTP_FROM` | sender address |

**Without SMTP no email is sent.** The link is then written to the server log (a warning), so you
can test locally. For local testing you can also set `LONGDOC_ACCOUNTS_DEV=1` to have the register
call return the link. Never use that setting on a public server.

### On Azure with Gmail (simplest)

Create a Gmail **app password** (Google Account > Security > 2-Step Verification on, then
https://myaccount.google.com/apppasswords). In Azure Cloud Shell, enter it at the hidden prompt:

```bash
read -rsp "Gmail app password: " GPW; echo; GPW=${GPW// /}
az containerapp secret set -n longdoc-app -g longdoc-rg --secrets smtp-password="$GPW" -o none
az containerapp update -n longdoc-app -g longdoc-rg --set-env-vars \
  LONGDOC_SMTP_HOST=smtp.gmail.com LONGDOC_SMTP_PORT=587 \
  LONGDOC_SMTP_USER=you@gmail.com LONGDOC_SMTP_FROM=you@gmail.com \
  LONGDOC_SMTP_PASSWORD=secretref:smtp-password LONGDOC_OWNER_EMAIL=you@gmail.com -o none
unset GPW
```

Gmail allows roughly 500 messages a day and sends from your Gmail address. For your own domain
or higher volume, use a transactional service (Azure Communication Services, SendGrid, ...).

## Alerts to you

Every event is logged, and written to `LONGDOC_EVENTS_FILE` if set. Add either or both:

| Variable | Result |
|---|---|
| `LONGDOC_GITHUB_REPO=owner/name` + `LONGDOC_GITHUB_TOKEN` | opens an issue per event (GitHub notifies you) |
| `LONGDOC_OWNER_EMAIL` (needs SMTP) | emails you per event |

Use a **private** repository for alerts: the issues contain people's email addresses. The token
needs only "Issues: read and write" on that repository (a fine-grained token). Events: new signup,
key issued, first use, key revoked. Each includes whether the address is personal (gmail, outlook,
...) or business.

"Has downloaded" cannot be detected: the package is installed from GitHub or PyPI, which does not
tell you who. What is tracked is use: requests, documents uploaded and questions asked, per account.

## Admin

Open `/admin`, enter `LONGDOC_ADMIN_TOKEN`, and see each account: email, type, status, key prefix,
request / document / question counts and last use. **Revoke** disables a key at once. The same data
as JSON: `GET /v1/admin/accounts` with `Authorization: Bearer <admin token>`.

## How it behaves

* The key is random, 256-bit, shown once, and only its SHA-256 hash is stored.
* The verification link is single-use and expires after 24 hours.
* Registering gives the same reply whether or not the address exists; registering again for an
  active address sends nothing and issues no second key; a revoked address cannot re-register.
* Registration and verification are limited to 10 attempts a minute per client address.
* Keys work in addition to `LONGDOC_API_KEY` (which keeps working as a master key).
* Pages send a strict Content-Security-Policy and `Referrer-Policy: no-referrer`.

## Limits

* One server process owns the account file. Several replicas would overwrite each other's usage
  counts: run one replica, or replace `AccountStore` with a database.
* Usage counts are written at most once a second (a crash can lose the last second).
* Anyone with a valid email can register: there is no approval step, CAPTCHA or domain
  allow-list. Add `LONGDOC_RATE_LIMIT_PER_MIN` and your own gateway limits for a public site.
* Not tested against a real SMTP server or the real GitHub API.
