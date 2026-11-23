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

## Threat model and built-in protections

| Risk (OWASP Top 10 for LLM Applications, 2025 / API Security Top 10, 2023) | What longdoc does | What remains yours |
|---|---|---|
| **Prompt injection (LLM01)** | Document text, headings and user questions are escaped before they enter the prompt, so they cannot close a tag or forge a `<section>`. Citation IDs that do not exist in the document are ignored. | A model can still be *persuaded* by instruction-like text inside a document. Treat answers as untrusted output; do not give the model tools or actions based on them. |
| **Unbounded consumption (LLM10, API4)** | Questions are limited to 8,000 characters, uploads to `LONGDOC_MAX_UPLOAD_MB`, output to `max_tokens`. Optional per-client rate limit (`LONGDOC_RATE_LIMIT_PER_MIN`) on uploads and questions. | The rate limit is per process; also limit at your gateway. There is no cap on the number or total size of stored documents. |
| **Broken object-level authorisation (API1)** | None. The API has one shared bearer token and does not know which of your users may read which document. | Enforce per-user access in your application (for example, namespace document IDs per tenant and check ownership before calling longdoc). |
| **Broken authentication (API2)** | Constant-time bearer-token comparison. | One shared token, no scopes or rotation. Use your gateway or identity provider for anything richer. |
| **Sensitive information disclosure (LLM02)** | Upstream error text is logged, not returned. | Documents are sent to the Anthropic API (or stay on the machine with a local model). Confirm that is permitted. |
| **Improper output handling (LLM05)** | The API returns JSON; citation IDs are checked against the real document. | Escape answer text before rendering it as HTML. |
| **Supply chain (LLM03)** | CI runs Bandit and `pip-audit`; Dependabot proposes updates. | Pin versions and review updates. |

## Known advisories

- **PYSEC-2026-2447 (`diskcache` 5.6.3, no fixed version at time of writing).** `diskcache`
  is installed only through the optional `local` extra (`llama-cpp-python`). The advisory
  requires an attacker with write access to a cache directory that `diskcache` reads
  (pickle deserialisation). longdoc never enables llama.cpp's disk cache. Do not enable
  `LlamaDiskCache` on a directory other users can write to, and re-check when a fixed
  release appears.
