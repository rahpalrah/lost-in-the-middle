# Security policy

## Reporting a vulnerability

Please report security issues privately through the repository's
**Security → Report a vulnerability** page (GitHub private vulnerability reporting),
not in a public issue. We aim to acknowledge reports within 3 working days.

## Deployment guidance

- **Always set `LONGDOC_API_KEY`** (or pass `api_key=` to `create_app`) when the HTTP
  API is reachable by anyone but you. Without it the API is unauthenticated and logs a
  warning at startup. When you mount `create_router()` in your own app, protect it with
  your app's existing authentication.
- **Keep `ANTHROPIC_API_KEY` server-side.** Never ship it to browsers or mobile apps;
  call this API from your backend instead.
- **Run behind TLS** (a reverse proxy or load balancer) in production.
- **Uploads are size-limited** (25 MB by default, `LONGDOC_MAX_UPLOAD_MB`). Document IDs
  are restricted to `[A-Za-z0-9_-]`, which prevents path traversal in `JSONFileStore`.
- **Documents are sent to the Anthropic API** to answer questions. Make sure that is
  allowed for the data you load, under your organisation's policies and Anthropic's
  terms.
- Error responses never include upstream error messages; those are logged instead.
