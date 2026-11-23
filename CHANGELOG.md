# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `docs/INTEGRATION.md`: step-by-step integration guide (library, FastAPI, standalone service,
  local model, architecture patterns, production checklist, troubleshooting).
- `examples/sqlite_store.py`: a runnable custom `DocumentStore`.

## [0.5.0] - 2026-11-23

### Added

- `EmbeddingRetriever` (semantic search over any embedding function) and
  `HybridRetriever` (reciprocal rank fusion of keyword and semantic search).
- `longdoc.local.LlamaCppEmbedder`: local embeddings, e.g. MIT-licensed bge-small.
- `scripts/download_models.sh` for the local answer and embedding models.
- `clean_answer()`: strips prompt markup that models echo into answers.

### Changed

- Citations are also recognised as bare IDs (`p0007`), not only `[p0007]`.

## [0.4.0] - 2026-11-23

### Added

- `longdoc.local`: `LlamaCppClient` and `AsyncLlamaCppClient` run the whole SDK on a
  local GGUF model through llama.cpp: no API key, no network, no per-token cost
  (`local` extra). They work anywhere an Anthropic client is accepted.
- `ContextWindowExceededError` when a prompt doesn't fit the local model's context.
- `longdoc ask --local-model` and `LONGDOC_LOCAL_MODEL` / `LONGDOC_LOCAL_CTX` for the
  HTTP server.
- `benchmarks/contract_review/run.py --local MODEL.gguf`.

## [0.3.0] - 2026-11-22

Fixes found by testing on a real-world PDF contract pack (`benchmarks/contract_review`).

### Fixed

- Section headings inside PDF-extracted text are now detected. PDF extraction leaves no
  blank lines between headings and body text, so every section previously came out
  with no heading and boundaries fell at arbitrary points.
- Exhibit-style headings such as `A.3 Personal Data Breach` are recognised.
- Wrapped body lines that start with a number or "Article 12" are no longer mistaken
  for headings.
- An article heading followed directly by its first sub-heading now forms one section
  (`ARTICLE 19. TERMINATION > 19.1 For Cause`) instead of a heading-only section.

### Added

- Running headers, footers and page numbers are stripped from PDF pages
  (`strip_running_headers`).
- `ChildChunk.heading`: passages carry their section heading, and `BM25Retriever`
  indexes it.
- Light stemming in `BM25Retriever` ("governs", "governed", "governing" match).
  On the benchmark, retrieval recall rose from 67% to 87%.
- `benchmarks/contract_review`: an end-to-end business benchmark (generator, 16
  questions with known answers, runner over the real HTTP API, cost guard, report).

## [0.2.0] - 2026-11-21

### Added

- `DocumentService`: thread-safe, framework-agnostic multi-document service with sync
  `ask()` and async `aask()`, LRU-cached indexes and a shared Anthropic client.
- `DocumentStore` protocol with `InMemoryStore` and `JSONFileStore` (atomic writes,
  path-traversal-safe IDs).
- `chunked_to_dict` / `chunked_from_dict` (versioned schema) and `answer_to_dict`.
- `load_bytes()` for uploaded files.
- HTTP API (`server` extra): `create_router()` to mount in an existing FastAPI app,
  `create_app()` for a standalone service, bearer-token auth, upload size limits,
  consistent error bodies and an OpenAPI spec.
- `longdoc.server.asgi:app`, configured from environment variables, for Uvicorn,
  Gunicorn and Docker.
- `longdoc` command-line tool: `longdoc ask` and `longdoc serve`.
- Dockerfile (non-root, health check), `SECURITY.md`, and Flask, Django and FastAPI
  integration examples.

### Changed

- `load_document()` now reads files through `load_bytes()`; behaviour is unchanged.

## [0.1.0] - 2026-11-21

### Added

- `DocumentQA` and `AsyncDocumentQA` for question answering over one long document.
- `Strategy.RETRIEVAL`: parent-child chunking with BM25 search over child passages.
- `Strategy.CACHED_DOCUMENT`: the whole document as a cached prompt prefix, with
  `warm_cache()` for pre-warming.
- Section citations with page numbers on every `Answer`.
- Cumulative token usage, including cache reads and writes, and a cache hit ratio.
- Server-side refusal fallbacks, on by default; `RefusalError` when every model declines.
- Loaders for `.txt`, `.md` and `.pdf` (optional `pdf` extra).
- Pluggable `Retriever` protocol for custom search backends.
