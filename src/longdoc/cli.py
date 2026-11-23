"""Command-line interface: ``longdoc ask`` and ``longdoc serve``."""

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
from .serialization import answer_to_dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longdoc", description="Question answering over long documents with Claude."
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
    ask.add_argument("--model", help="Model ID (required with --provider openai)")
    ask.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Which API answers the questions (default: anthropic)",
    )
    ask.add_argument(
        "--openai-base-url", help="Use an OpenAI-compatible server instead of api.openai.com"
    )
    ask.add_argument("--json", action="store_true", help="Print answers as JSON lines")
    ask.add_argument(
        "--local-model",
        metavar="MODEL.gguf",
        help="Answer with a local GGUF model instead of Claude (no API key needed)",
    )
    ask.add_argument("--n-ctx", type=int, default=16_384, help="Local model context window")

    serve = sub.add_parser("serve", help="Run the HTTP API (needs the 'server' extra)")
    serve.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--store", help="Directory for persisted documents (default: in memory only)"
    )
    serve.add_argument("--workers", type=int, default=1)
    return parser


def _ask(args: argparse.Namespace) -> int:
    overrides = {"model": args.model} if args.model else {}
    client: Any = None
    if args.provider == "openai":
        from .openai_client import OpenAIClient

        if args.local_model:
            raise ConfigurationError("Use either --provider openai or --local-model, not both.")
        if not args.model:
            raise ConfigurationError("--provider openai needs --model (a chat model you can use).")
        client = OpenAIClient(base_url=args.openai_base_url)
        overrides = {"fallbacks": None, **overrides}
    if args.local_model:
        from .local import LlamaCppClient

        client = LlamaCppClient(args.local_model, n_ctx=args.n_ctx)
        overrides = {"model": client.model_name, "fallbacks": None, **overrides}
    config = QAConfig(strategy=Strategy(args.strategy), **overrides)
    qa = DocumentQA.from_file(args.path, config=config, client=client)
    for question in args.questions:
        answer = qa.ask(question)
        if args.json:
            print(json.dumps({"question": question, **answer_to_dict(answer)}, ensure_ascii=False))
            continue
        print(f"Q: {question}\nA: {answer.text}")
        for s in answer.citations:
            pages = f", pages {s.page_label}" if s.page_label else ""
            print(f"   - [{s.id}] {s.heading or '(untitled)'}{pages}")
        print()
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print('error: the server needs extras: pip install "longdoc-qa[server]"', file=sys.stderr)
        return 1
    if args.workers > 1 and not args.store:
        print("error: --workers > 1 needs --store so workers share documents", file=sys.stderr)
        return 1
    # Workers are separate processes, so configuration travels via the environment.
    if args.store:
        os.environ["LONGDOC_STORE_DIR"] = args.store
    uvicorn.run("longdoc.server.asgi:app", host=args.host, port=args.port, workers=args.workers)
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
    except LongDocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        import anthropic

        if isinstance(exc, anthropic.APIError):
            print(f"API error: {exc}", file=sys.stderr)
            return 2
        message = _openai_error_message(exc)
        if message:
            print(f"API error: {message}", file=sys.stderr)
            return 2
        raise


def _openai_error_message(exc: Exception) -> str | None:
    """Describe an OpenAI SDK error without risking an echoed fragment of the API key."""
    try:
        import openai
    except ImportError:
        return None
    from .openai_client import is_out_of_credit

    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY (see docs/OPENAI.md)."
    if isinstance(exc, openai.PermissionDeniedError):
        return "OpenAI denied access to this model or project for your key."
    if isinstance(exc, openai.RateLimitError):
        if is_out_of_credit(exc):
            return (
                "Your OpenAI account has no credit left. Add credit at "
                "https://platform.openai.com/settings/organization/billing/ and try again."
            )
        return "OpenAI rate limit reached. Wait a moment and try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach the OpenAI API."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI returned HTTP {exc.status_code}."
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
