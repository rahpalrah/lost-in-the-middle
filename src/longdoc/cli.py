"""Command-line interface: ``longdoc ask`` and ``longdoc serve``.

Both work with no extra packages and no API key: ``ask`` falls back to the built-in extractive
engine, and ``serve`` uses the built-in standard-library HTTP server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

from ._version import __version__
from .client import DocumentQA
from .config import QAConfig, Strategy
from .env import load_env_file
from .exceptions import ConfigurationError, LongDocError
from .http import ProviderError
from .loaders import load_document
from .providers import PROVIDERS, make_client
from .serialization import answer_to_dict

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longdoc",
        description="Question answering over long documents. Works with no API key and no extra "
        "packages (built-in engine); optionally uses Claude, OpenAI or a local model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help="Load variables such as OPENAI_API_KEY from this .env file (never printed)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask questions about a file")
    ask.add_argument("path", help=".pdf, .txt, .md or .rst file")
    ask.add_argument("questions", nargs="+", help="One or more questions")
    ask.add_argument(
        "--strategy",
        choices=[s.value for s in Strategy],
        default=Strategy.RETRIEVAL.value,
        help="retrieval (default) or cached_document",
    )
    ask.add_argument(
        "--provider",
        choices=[p for p in PROVIDERS if p != "local"],
        default="auto",
        help="auto (default): Claude if ANTHROPIC_API_KEY is set, else the built-in engine",
    )
    ask.add_argument("--model", help="Model ID (required with --provider openai)")
    ask.add_argument(
        "--openai-base-url", help="Use an OpenAI-compatible server instead of api.openai.com"
    )
    ask.add_argument(
        "--local-model",
        metavar="MODEL.gguf",
        help="Answer with a local GGUF model (needs the 'local' extra)",
    )
    ask.add_argument("--n-ctx", type=int, default=16_384, help="Local model context window")
    ask.add_argument(
        "--pdf-backend",
        choices=["auto", "builtin", "pypdf"],
        default="auto",
        help="PDF reader: built-in (default; pypdf is used instead if installed)",
    )
    ask.add_argument("--json", action="store_true", help="Print answers as JSON lines")

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--store", help="Directory for persisted documents (default: in memory)")
    serve.add_argument(
        "--framework",
        choices=["builtin", "fastapi"],
        default="builtin",
        help="builtin (default): standard library only. fastapi: Uvicorn workers (needs the "
        "'fastapi' extra)",
    )
    serve.add_argument("--workers", type=int, default=1, help="Worker processes (fastapi only)")
    return parser


def _ask(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {"model": args.model} if args.model else {}
    provider = "local" if args.local_model else args.provider
    if provider == "openai":
        if not args.model:
            raise ConfigurationError("--provider openai needs --model (a chat model you can use).")
        client = make_client("openai", base_url=args.openai_base_url)
        overrides = {"fallbacks": None, **overrides}
    elif provider == "local":
        client = make_client("local", model_path=args.local_model, n_ctx=args.n_ctx)
        overrides = {"model": client.model_name, "fallbacks": None, **overrides}
    else:
        client = make_client(provider)
        if provider == "extractive" or type(client).__name__ == "ExtractiveClient":
            overrides = {"fallbacks": None, **overrides}
    config = QAConfig(strategy=Strategy(args.strategy), **overrides)
    document = load_document(args.path, pdf_backend=args.pdf_backend)
    qa = DocumentQA(document, config=config, client=client)
    for question in args.questions:
        answer = qa.ask(question)
        if args.json:
            print(json.dumps({"question": question, **answer_to_dict(answer)}, ensure_ascii=False))
            continue
        print(f"Q: {question}\nA: {answer.text}")
        for s in answer.citations:
            pages = f", pages {s.page_label}" if s.page_label else ""
            print(f"   - [{s.id}] {s.heading or '(untitled)'}{pages}")
        print(f"   (answered by: {answer.model})\n")
    return 0


def _serve(args: argparse.Namespace) -> int:
    if args.store:
        os.environ["LONGDOC_STORE_DIR"] = args.store  # workers are separate processes
    if args.framework == "fastapi":
        try:
            import uvicorn
        except ImportError:
            print(
                'error: --framework fastapi needs: pip install "longdoc-qa[fastapi]"',
                file=sys.stderr,
            )
            return 1
        if args.workers > 1 and not args.store:
            print("error: --workers > 1 needs --store so workers share documents", file=sys.stderr)
            return 1
        uvicorn.run("longdoc.server.asgi:app", host=args.host, port=args.port, workers=args.workers)
        return 0
    if args.workers != 1:
        print(
            "error: the built-in server is one process with many threads. For several processes "
            "use --framework fastapi, or run several instances behind a load balancer.",
            file=sys.stderr,
        )
        return 1
    from .server import serve, settings_from_env

    settings = settings_from_env()
    if settings.api_key is None and args.host not in _LOOPBACK:
        print(
            "warning: listening on a non-local address WITHOUT authentication. "
            "Set LONGDOC_API_KEY.",
            file=sys.stderr,
        )
    print(f"longdoc serving on http://{args.host}:{args.port} (engine: {settings.provider})")
    serve(settings, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``longdoc`` command."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if args.env_file:
        try:
            load_env_file(args.env_file)
        except ConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        return _ask(args) if args.command == "ask" else _serve(args)
    except ProviderError as exc:  # the message is safe: it never contains the key
        print(f"API error: {exc}", file=sys.stderr)
        return 2
    except LongDocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
