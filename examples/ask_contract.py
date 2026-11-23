"""Ask questions about a long document from the command line.

Usage:
    python examples/ask_contract.py contract.pdf "What is the notice period?" ["Another?" ...]
    python examples/ask_contract.py contract.pdf --cached "Q1?" "Q2?" "Q3?"

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile) in the environment.
"""

from __future__ import annotations

import argparse
import logging
import sys

import anthropic

from longdoc import DocumentQA, LongDocError, QAConfig, Strategy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="Path to a .pdf, .txt or .md file")
    parser.add_argument("questions", nargs="+", help="One or more questions")
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Send the whole document as a cached prefix (best for several questions)",
    )
    parser.add_argument("--verbose", action="store_true", help="Log request IDs and usage")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    strategy = Strategy.CACHED_DOCUMENT if args.cached else Strategy.RETRIEVAL

    try:
        qa = DocumentQA.from_file(args.path, config=QAConfig(strategy=strategy))
        print(f"Loaded {args.path}: {len(qa.chunks.parents)} sections\n")
        for question in args.questions:
            answer = qa.ask(question)
            print(f"Q: {question}\nA: {answer.text}")
            for s in answer.citations:
                pages = f", pages {s.page_label}" if s.page_label else ""
                print(f"   - [{s.id}] {s.heading or '(untitled)'}{pages}")
            print()
    except LongDocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except anthropic.APIError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 2

    u = qa.usage
    print(
        f"Tokens - input: {u.input_tokens}, output: {u.output_tokens}, "
        f"cache write: {u.cache_creation_input_tokens}, cache read: {u.cache_read_input_tokens} "
        f"({u.cache_hit_ratio:.0%} from cache)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
