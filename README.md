# longdoc-qa (lost-in-the-middle)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Typed](https://img.shields.io/badge/typing-strict-informational)

An SDK for adding accurate, low-cost question answering over long documents
(contracts, policies, manuals, reports) to your existing system, powered by Claude.
Use it as a **Python library**, **inside your existing web app**, or as a
**standalone HTTP service** that any language can call.

## Why

Putting a 100-page PDF straight into a prompt "because the context window is big
enough" works in a demo and fails in production:

- **Accuracy drops on the middle pages.** Models pay less attention to content buried
  deep in a very long prompt (the "lost in the middle" problem). A condition on page 54
  gets missed.
- **Cost multiplies.** Re-sending the whole document with every question bills every
  page, every time.
- **"Read every page carefully" in the prompt doesn't fix attention.**

`longdoc` packages the two production fixes:

| Strategy | How it works | Use it when |
|---|---|---|
| `retrieval` (default) | **Parent-child chunking.** Small overlapping *child* passages are searched; the model receives only the full *parent* sections they belong to. | One-off questions, very large documents, small focused prompts. |
| `cached_document` | **Prefix prompt caching.** The whole sectioned document is sent once as a cached prompt prefix; later questions read it from cache at a small fraction of the input price, with faster responses. | Many questions about the same document in a short window. |

Every answer comes back with **section citations and page numbers**, plus token usage
including cache reads and writes.

## Installation

**No third-party packages are required.** The library, CLI, HTTP server, PDF reader and the
built-in answer engine use only the Python standard library (3.10+), so this works on a locked-down
machine and answers questions with no API key and no model:

```bash
pip install git+https://github.com/rahpalrah/lost-in-the-middle   # not yet on PyPI
longdoc ask contract.pdf "What is the notice period?"             # built-in engine, $0, offline
longdoc serve --port 8000                                          # built-in HTTP server
```

Optional extras (none are needed for the above):

```bash
pip install "longdoc-qa[pdf] @ git+https://github.com/rahpalrah/lost-in-the-middle"      # pypdf as an alternative PDF reader
pip install "longdoc-qa[fastapi] @ git+https://github.com/rahpalrah/lost-in-the-middle"  # FastAPI/Uvicorn instead of the built-in server
pip install "longdoc-qa[local] @ git+https://github.com/rahpalrah/lost-in-the-middle"    # run a local GGUF model
```

Claude and OpenAI are called directly over HTTPS by the SDK itself (no vendor SDK needed): set
`ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` with `--provider openai --model <model>`. With neither,
the built-in extractive engine answers by quoting the document, with citations, and says so when
the document does not contain the answer.

> **Status: alpha (0.8).** See [docs/QUALITY_REPORT.md](docs/QUALITY_REPORT.md) for what is and is
> not verified. The Claude and OpenAI request paths have been tested against local stand-in
> servers only, never against the live services.

## Choose your integration

> **Step-by-step instructions for your own systems, applications and architecture are in the
> [Integration guide](docs/INTEGRATION.md).** Measured quality, security checks and known
> gaps are in the [Quality report](docs/QUALITY_REPORT.md).

| You have... | Use | Section |
|---|---|---|
| A Python script or one document | `DocumentQA` | [Library](#1-library) |
| A Python backend (Django, Flask, FastAPI, Celery, Lambda) | `DocumentService` | [Existing Python system](#2-existing-python-system) |
| A FastAPI app | `create_router()` | [Mount in FastAPI](#3-mount-into-an-existing-fastapi-app) |
| A non-Python system (Node, Java, Go, .NET, PHP...) | HTTP API / Docker | [HTTP service](#4-standalone-http-service) |
| A terminal | `longdoc ask` | [CLI](#5-cli) |

### 1. Library

```python
from longdoc import DocumentQA

qa = DocumentQA.from_file("contract.pdf")
answer = qa.ask("What is the termination notice period, and are there conditions?")

print(answer.text)
for section in answer.citations:
    print(f"  [{section.id}] {section.heading} (pages {section.page_label})")
print(answer.usage)
```

Many questions about one document? Cache it:

```python
from longdoc import DocumentQA, QAConfig, Strategy

qa = DocumentQA.from_file(
    "contract.pdf", config=QAConfig(strategy=Strategy.CACHED_DOCUMENT, cache_ttl="1h")
)
qa.warm_cache()  # optional: pay the cache write before the first user is waiting
for q in ["Who are the parties?", "What are the payment terms?", "Any penalties?"]:
    print(q, "->", qa.ask(q).text)
print(f"Cache hit ratio: {qa.usage.cache_hit_ratio:.0%}")
```

Async: `AsyncDocumentQA` has the same API with `await qa.ask(...)`.

### 2. Existing Python system

`DocumentService` manages many documents behind one thread-safe object, with no
web-framework dependency. Create one per process and call it from your own routes,
views, tasks or handlers.

```python
from longdoc import DocumentService, JSONFileStore, answer_to_dict

service = DocumentService(store=JSONFileStore("/var/lib/myapp/longdoc"))

# Index once, using your own primary key as the document ID.
service.add_file("uploads/acme-msa.pdf", document_id="contract-1042")
# ...or service.add_bytes(upload_bytes, filename="acme.pdf", document_id="contract-1042")

answer = service.ask("contract-1042", "What is the liability cap?")  # or: await service.aask(...)
return_json = answer_to_dict(answer)  # JSON-ready response body
```

- **Chunk once, query many times.** Documents are parsed and chunked when added and
  stored as JSON; indexes are rebuilt in memory on demand (LRU-cached).
- **Bring your own storage.** Implement the four-method `DocumentStore` protocol
  (`get`, `put`, `delete`, `list_ids`) to keep documents in Postgres, S3 or Redis:

  ```python
  class PostgresStore:
      def get(self, document_id): ...  # -> ChunkedDocument | None, via chunked_from_dict()
      def put(self, document_id, doc): ...  # store chunked_to_dict(doc) in a JSONB column
      def delete(self, document_id): ...  # -> bool
      def list_ids(self): ...  # -> iterator of IDs


  service = DocumentService(store=PostgresStore())
  ```
- **Bring your own search.** Pass `retriever_factory=` to use embeddings or a vector
  database instead of the built-in BM25 keyword search (see [Retrieval](#retrieval)).

Complete examples: [`examples/flask_integration.py`](examples/flask_integration.py) and
[`examples/django_integration.py`](examples/django_integration.py).

### 3. Mount into an existing FastAPI app

```python
from fastapi import Depends, FastAPI
from longdoc import DocumentService, JSONFileStore
from longdoc.server import create_router

service = DocumentService(store=JSONFileStore("./longdoc-data"))
app = FastAPI()
app.include_router(
    create_router(service),
    prefix="/longdoc",
    dependencies=[Depends(your_existing_auth)],  # reuse your own authentication
)
```

The routes and their schemas appear in your app's OpenAPI docs. See
[`examples/fastapi_mount.py`](examples/fastapi_mount.py).

### 4. Standalone HTTP service

Run it next to your system and call it from any language.

```bash
export ANTHROPIC_API_KEY=...            # optional: without it the built-in engine answers
export LONGDOC_API_KEY=choose-a-token   # callers must send this as a bearer token
longdoc serve --store ./longdoc-data --host 0.0.0.0 --port 8000
# one process, many threads, standard library only. More processes: run several instances behind
# a load balancer, or `--framework fastapi --workers 4` (needs the fastapi extra).
```

Or with Docker:

```bash
docker build -t longdoc .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e LONGDOC_API_KEY -v longdoc-data:/data longdoc
```

Interactive docs are at `http://localhost:8000/docs` and the OpenAPI spec at
`/openapi.json`; generate a typed client for your language from the spec with any
OpenAPI generator.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe (no auth) |
| `POST` | `/v1/documents` | Upload a file (multipart: `file`, optional `id`) |
| `POST` | `/v1/documents/text` | Add a document as JSON `{"text", "source"?, "id"?}` |
| `GET` | `/v1/documents` | List documents |
| `GET` | `/v1/documents/{id}` | Document summary |
| `DELETE` | `/v1/documents/{id}` | Delete a document |
| `POST` | `/v1/documents/{id}/ask` | `{"question", "strategy"?, "include_context_text"?}` → cited answer |
| `POST` | `/v1/documents/{id}/retrieve` | Preview the sections retrieval would send (no model call) |

```bash
curl -H "Authorization: Bearer $LONGDOC_API_KEY" \
     -F file=@contract.pdf -F id=contract-1042 \
     http://localhost:8000/v1/documents

curl -H "Authorization: Bearer $LONGDOC_API_KEY" -H "Content-Type: application/json" \
     -d '{"question": "What is the termination notice period?"}' \
     http://localhost:8000/v1/documents/contract-1042/ask
```

```js
// Node.js / browser backend
const res = await fetch(`${LONGDOC_URL}/v1/documents/contract-1042/ask`, {
  method: "POST",
  headers: { Authorization: `Bearer ${process.env.LONGDOC_API_KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What is the termination notice period?" }),
});
const { text, citations } = await res.json();
```

Errors always have the shape `{"error": "<code>", "message": "...", "details": {...}}`:

| Status | `error` | Meaning |
|---|---|---|
| 401 | `unauthorized` | Missing or wrong bearer token |
| 404 | `not_found` | No document with that ID |
| 413 | `too_large` | Upload over the size limit |
| 415 | `unsupported_format` | File type not supported |
| 422 | `invalid_document` / `invalid_request` / `refused` | Unreadable file, bad input, or the model declined |
| 429 | `rate_limited` | Claude rate limit reached; retry with backoff |
| 502 / 503 | `upstream_error` / `upstream_unavailable` | Claude API error or unreachable |

**Server configuration** (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Claude credential |
| `LONGDOC_API_KEY` | none (unauthenticated, logs a warning) | Bearer token for callers |
| `LONGDOC_STORE_DIR` | in-memory | Document storage directory (required for more than one worker) |
| `LONGDOC_MODEL` | `claude-opus-5-5` | Claude model ID |
| `LONGDOC_EFFORT` | `medium` | `low` … `max` |
| `LONGDOC_STRATEGY` | `retrieval` | Default strategy |
| `LONGDOC_CACHE_TTL` | `5m` | `5m` or `1h` |
| `LONGDOC_FALLBACKS` | `default` | `none` to disable refusal fallbacks |
| `LONGDOC_MAX_UPLOAD_MB` | `25` | Upload size limit |

Any ASGI server works: `uvicorn longdoc.server.asgi:app --workers 4`.

### 5. CLI

```bash
longdoc ask contract.pdf "What is the notice period?" "Is there a termination fee?"
longdoc ask contract.pdf "Summarise the payment terms" --strategy cached_document --json
```

## Using OpenAI instead of Claude

```python
from longdoc import DocumentQA, QAConfig
from longdoc.openai_client import OpenAIClient

qa = DocumentQA.from_file(
    "contract.pdf",
    client=OpenAIClient(),  # key comes from the OPENAI_API_KEY variable
    config=QAConfig(model="<your chat model>", fallbacks=None),
)
```

```bash
cp .env.example .env && chmod 600 .env        # put your own key in .env (git-ignored)
longdoc --env-file .env ask contract.pdf "What is the notice period?" --provider openai --model <model>
```

**Never commit or paste a key.** Step-by-step setup, safe ways to provide the key, spending
controls and what is and is not verified: [docs/OPENAI.md](docs/OPENAI.md). The adapter has been
tested against a local stand-in server only, not against the real OpenAI service.

## Without an API key: local models

Run the whole SDK (library, `DocumentService`, HTTP API, CLI) on an open-source model
on your own hardware: no Anthropic API key, no per-token cost, nothing leaves the machine.
Any GGUF chat model works through [llama.cpp](https://github.com/ggml-org/llama.cpp); for
example MIT-licensed [Phi-3.5-mini-instruct](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF).

```python
from longdoc import DocumentQA, QAConfig
from longdoc.local import LlamaCppClient

llm = LlamaCppClient("models/Phi-3.5-mini-instruct-Q4_K_M.gguf", n_ctx=16384)
qa = DocumentQA.from_file(
    "contract.pdf", client=llm, config=QAConfig(model=llm.model_name, fallbacks=None)
)
print(qa.ask("What is the termination notice period?").text)
```

```bash
longdoc ask contract.pdf "What is the notice period?" --local-model models/phi.gguf
LONGDOC_LOCAL_MODEL=models/phi.gguf longdoc serve --store ./data
```

For paraphrased questions, add local semantic search (MIT-licensed bge-small, 36 MB;
`scripts/download_models.sh` fetches both models):

```python
from longdoc import BM25Retriever, DocumentService, EmbeddingRetriever, HybridRetriever
from longdoc.local import BGE_QUERY_PREFIX, LlamaCppEmbedder

embed = LlamaCppEmbedder("models/bge-small-en-v1.5-q8_0.gguf")
service = DocumentService(
    client=llm,
    retriever_factory=lambda: HybridRetriever(
        BM25Retriever(), EmbeddingRetriever(embed, query_prefix=BGE_QUERY_PREFIX)
    ),
)
```

Measured on the contract-review benchmark: keyword search 14/16 correct, hybrid 15/16
([details](benchmarks/contract_review/README.md)).

Trade-offs: small local models are less accurate than Claude, and much slower on a CPU
(about 1-2 minutes per question on 4 cores). Their context windows are small, so use
`Strategy.RETRIEVAL`: whole-document strategies only fit short documents. Claude-only
options (prompt caching, effort, refusal fallbacks) are ignored.

## Configuration

```python
from longdoc import ChunkingConfig, QAConfig

QAConfig(
    model="claude-opus-5-5",  # any Claude model ID
    effort="medium",  # low | medium | high | xhigh | max
    max_tokens=16_000,  # answers stream, so large values are safe
    top_k_children=12,  # child passages retrieved per question
    max_parents=4,  # parent sections sent per question
    max_context_chars=40_000,  # character budget for those sections
    cache_ttl="5m",  # "5m" or "1h" for cached_document
    fallbacks="default",  # server-side refusal fallback; None to disable
)
ChunkingConfig(parent_max_chars=6_000, child_max_chars=800, child_overlap_chars=150)
```

**Refusal fallbacks** are on by default: if the model declines for a policy reason, the
API retries on a fallback model in the same call, and `answer.fallback_used` tells you
it happened. Set `fallbacks=None` (or `LONGDOC_FALLBACKS=none`) for models or
platforms that don't support it, such as Amazon Bedrock, Vertex AI and Microsoft
Foundry. If every model declines, you get `RefusalError` (HTTP 422 `refused`).

**Your own Anthropic client** (proxies, retries, timeouts, other platforms):

```python
import anthropic

service = DocumentService(
    client=anthropic.Anthropic(max_retries=5, timeout=120),
    async_client=anthropic.AsyncAnthropic(max_retries=5, timeout=120),
)
```

## Retrieval

The default `BM25Retriever` is pure-Python keyword search: strong for exact terms such
as clause numbers, names and defined terms. For paraphrased questions, implement the
`Retriever` protocol with embeddings or a vector database:

```python
from longdoc import RetrievalHit


class MyVectorRetriever:
    def index(self, children): ...  # embed and store child passages
    def search(self, query, k): ...  # -> list[RetrievalHit], best first


qa = DocumentQA.from_file("contract.pdf", retriever=MyVectorRetriever())
service = DocumentService(retriever_factory=MyVectorRetriever)
```

Preview retrieval without calling the model: `qa.retrieve(question)`,
`service.retrieve(doc_id, question)` or `POST /v1/documents/{id}/retrieve`.

## Errors (library)

All library errors inherit from `longdoc.LongDocError`:

| Exception | Raised when |
|---|---|
| `DocumentNotFoundError` | No document with that ID in the store |
| `DocumentLoadError` | The file is missing, unreadable or corrupt |
| `UnsupportedFormatError` | Unknown extension, or `pypdf` isn't installed |
| `EmptyDocumentError` | No text could be extracted (for example, a scanned PDF without OCR) |
| `ConfigurationError` | An invalid setting, document ID or empty question |
| `RefusalError` | The model and any fallbacks declined; has `.category` and `.request_id` |

Claude API errors are **not wrapped**: they propagate as the Anthropic SDK's typed
exceptions (`anthropic.RateLimitError`, and so on) after the SDK's built-in retries.
If an answer hit `max_tokens`, `answer.truncated` is `True` and a warning is logged.

## How caching stays effective

Prompt caching is a byte-exact prefix match. `longdoc` keeps the prefix stable: chunk
IDs are deterministic, sections are rendered in document order, nothing volatile goes
in the system prompt, and the question always comes after the cached document. If you
pass a custom `system_prompt`, keep it constant. Watch
`usage.cache_read_input_tokens`: if it stays at 0 on repeated questions within the TTL,
something in the prefix is changing.

## Logging and security

The library logs to the `longdoc` logger (request IDs and token usage at `DEBUG`;
truncation and missing API authentication at `WARNING`) and never configures handlers
itself. See [SECURITY.md](SECURITY.md) for deployment guidance and how to report
vulnerabilities.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest --cov
```

The tests use a fake client and make no network calls. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). You may use, modify and redistribute this SDK, including in
commercial and closed-source products, provided the copyright and license notice is
kept.
