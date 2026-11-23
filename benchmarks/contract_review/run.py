"""Contract-review benchmark: test the whole longdoc system on a real business workflow.

Scenario: a buyer's contract-management system sends a vendor contract pack (MSA with a
DPA and Order Form attached) and an unrelated NDA to the longdoc HTTP API, then a
review analyst asks 16 questions with known answers. Every question is run three ways:

* ``retrieval``        - longdoc parent-child retrieval (via the HTTP API)
* ``cached_document``  - longdoc whole-document prompt caching (via the HTTP API)
* ``naive``            - the common anti-pattern: the full extracted PDF text pasted into
                         every prompt, no caching, no sections (direct Anthropic SDK call)

Each answer is scored for correctness, correct citation (longdoc only), cost, cache use
and latency. Results go to ``results/<timestamp>/`` as JSON and a Markdown report.

Usage:
    # Offline: checks the plumbing with a fake model. Answers are placeholders and are
    # NOT results. Costs nothing.
    python benchmarks/contract_review/run.py --offline

    # Live: real Claude calls. Needs ANTHROPIC_API_KEY. Stops before exceeding --max-cost.
    python benchmarks/contract_review/run.py --live --max-cost 5

    # OpenAI (your own key, billed): key from OPENAI_API_KEY or --env-file; spend is capped.
    LONGDOC_ALLOW_LIVE_API=1 python benchmarks/contract_review/run.py --openai <model> \
        --price-in <usd/1M> --price-out <usd/1M> --strategies retrieval --max-cost 1

    # Local: an open-source GGUF model on this machine. No API key, no cost.
    python benchmarks/contract_review/run.py --local models/Phi-3.5-mini-instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_pack import FACTS, build_contract_pack, build_nda  # noqa: E402

from longdoc import (  # noqa: E402
    BM25Retriever,
    DocumentService,
    EmbeddingRetriever,
    HybridRetriever,
    JSONFileStore,
    ParentChildChunker,
    QAConfig,
    load_document,
)
from longdoc.remote import RemoteClient, RemoteError  # noqa: E402
from longdoc.server import create_server  # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "medium"

# US$ per million tokens for claude-opus-5-5 (Anthropic first-party pricing, Nov 2026).
# Check current pricing before relying on these figures.
PRICE_INPUT = 4.00
PRICE_OUTPUT = 20.00
PRICE_CACHE_WRITE_5M = PRICE_INPUT * 1.25
PRICE_CACHE_READ = 0.20

NAIVE_SYSTEM = "You are a contract analyst. Answer the question using the contract text provided."
NOT_FOUND = re.compile(
    r"(does not|doesn't|do not|don't|no)\b[^.]{0,80}\b(address|mention|specif|contain|include|"
    r"provi|cover|requir|impose|state|set out|refer|obligation|clause|provision)"
    r"|not (?:addressed|specified|covered|mentioned|found)|is silent|no (?:such|express)",
    re.I,
)
API_TOKEN = "benchmark-token"


# -- cost ------------------------------------------------------------------------------


BILLED = True  # False for local models: they cost nothing per token
NAIVE_MAX_TOKENS = 16_000


def cost_usd(usage: dict[str, int]) -> float:
    if not BILLED:
        return 0.0
    return (
        usage.get("input_tokens", 0) * PRICE_INPUT
        + usage.get("output_tokens", 0) * PRICE_OUTPUT
        + usage.get("cache_creation_input_tokens", 0) * PRICE_CACHE_WRITE_5M
        + usage.get("cache_read_input_tokens", 0) * PRICE_CACHE_READ
    ) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    limit: float
    spent: float = 0.0

    def check(self, next_call_estimate: float) -> None:
        if self.spent + next_call_estimate > self.limit:
            raise BudgetExceeded(
                f"Stopping: spent ${self.spent:.2f}; next call (~${next_call_estimate:.2f}) "
                f"would exceed the ${self.limit:.2f} cap."
            )

    def add(self, amount: float) -> None:
        self.spent += amount


# -- offline fake model (plumbing check only) -----------------------------------------

PLACEHOLDER = "OFFLINE PLACEHOLDER - not a model answer."


def _fake_message(kwargs: dict[str, Any]) -> Any:
    prompt_chars = len(json.dumps(kwargs.get("system", ""))) + len(json.dumps(kwargs["messages"]))
    cached = any(
        isinstance(b, dict) and "cache_control" in b for b in kwargs.get("system", []) or []
    )
    tokens = prompt_chars // 4
    return SimpleNamespace(
        content=[]
        if kwargs["max_tokens"] == 0
        else [SimpleNamespace(type="text", text=PLACEHOLDER)],
        stop_reason="end_turn",
        stop_details=None,
        model=kwargs["model"],
        _request_id="offline",
        usage=SimpleNamespace(
            input_tokens=200 if cached else tokens,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=tokens - 200 if cached else 0,
            iterations=None,
        ),
    )


class _FakeStream:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self._msg = _fake_message(kwargs)

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self._msg


class _AsyncFakeStream(_FakeStream):
    async def get_final_message(self) -> Any:  # type: ignore[override]
        return self._msg


class FakeAnthropic:
    def __init__(self, is_async: bool = False) -> None:
        stream_cls = _AsyncFakeStream if is_async else _FakeStream
        messages = SimpleNamespace(stream=lambda **kw: stream_cls(kw), create=_fake_message)
        self.beta = SimpleNamespace(messages=messages)


# -- server ----------------------------------------------------------------------------


class ServerThread:
    """Runs the real longdoc HTTP API (the built-in server) on a local port."""

    def __init__(self, service: DocumentService) -> None:
        self.server = create_server(service, port=0, api_key=API_TOKEN, max_concurrency=8)

    def __enter__(self) -> str:
        self.server.start_in_thread()
        return self.server.url

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


# -- grading ---------------------------------------------------------------------------


@dataclass
class Result:
    question_id: str
    strategy: str
    category: str
    correct: bool
    cited_correctly: bool | None
    retrieved_correctly: bool | None
    latency_s: float
    usage: dict[str, int]
    cost_usd: float
    answer: str
    failures: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


def grade(q: dict[str, Any], answer: str) -> tuple[bool, list[str]]:
    failures = []
    if q.get("expect_not_found") and not NOT_FOUND.search(answer):
        failures.append("did not say the documents are silent")
    for group in q.get("expect", []):
        if not any(re.search(p, answer, re.I) for p in group):
            failures.append(f"missing one of {group}")
    for pattern in q.get("forbid", []):
        if re.search(pattern, answer, re.I):
            failures.append(f"contains forbidden {pattern!r}")
    return not failures, failures


def fact_sections(pdf: Path) -> dict[str, set[str]]:
    """Map each fact to the section IDs that contain it (same chunker as the server)."""
    chunked = ParentChildChunker().chunk(load_document(pdf))
    norm = lambda s: " ".join(s.split())  # noqa: E731
    return {
        key: {p.id for p in chunked.parents if norm(text)[:60] in norm(p.text)}
        for key, text in FACTS.items()
    }


# -- runs ------------------------------------------------------------------------------


def run_sdk_strategy(
    remote: RemoteClient,
    strategy: str,
    questions: list[dict[str, Any]],
    doc_ids: dict[str, str],
    truth: dict[str, dict[str, set[str]]],
    budget: Budget,
    estimate: float,
) -> list[Result]:
    results = []
    for q in questions:
        budget.check(estimate)
        start = time.perf_counter()
        try:
            body = remote.ask(doc_ids[q["doc"]], q["question"], strategy=strategy)
        except RemoteError as exc:
            raise RuntimeError(f"{q['id']} {strategy}: {exc}") from exc
        latency = time.perf_counter() - start
        usage = {
            k: body["usage"][k]
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        }
        cost = cost_usd(usage)
        budget.add(cost)
        correct, failures = grade(q, body["text"])
        expected = set().union(*(truth[q["doc"]].get(f, set()) for f in q.get("facts", [])))
        cited = [c["id"] for c in body["citations"]]
        context = {c["id"] for c in body["context"]}
        results.append(
            Result(
                question_id=q["id"],
                strategy=strategy,
                category=q["category"],
                correct=correct,
                cited_correctly=bool(expected & set(cited)) if expected else None,
                retrieved_correctly=bool(expected & context) if expected else None,
                latency_s=round(latency, 2),
                usage=usage,
                cost_usd=round(cost, 5),
                answer=body["text"],
                failures=failures,
                citations=cited,
            )
        )
        print(
            f"  {q['id']} {strategy:16s} {'PASS' if correct else 'FAIL'} "
            f"{latency:5.1f}s ${cost:.4f}  spent ${budget.spent:.2f}"
        )
    return results


def run_naive(
    client: Any,
    questions: list[dict[str, Any]],
    raw_texts: dict[str, str],
    budget: Budget,
    estimate: float,
) -> list[Result]:
    results = []
    for q in questions:
        budget.check(estimate)
        start = time.perf_counter()
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=NAIVE_MAX_TOKENS,
            system=NAIVE_SYSTEM,
            output_config={"effort": EFFORT},
            messages=[
                {
                    "role": "user",
                    "content": f"<contract>\n{raw_texts[q['doc']]}\n</contract>\n\n{q['question']}",
                }
            ],
        ) as stream:
            msg = stream.get_final_message()
        latency = time.perf_counter() - start
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_creation_input_tokens": msg.usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": msg.usage.cache_read_input_tokens or 0,
        }
        cost = cost_usd(usage)
        budget.add(cost)
        correct, failures = grade(q, text)
        results.append(
            Result(
                question_id=q["id"],
                strategy="naive",
                category=q["category"],
                correct=correct,
                cited_correctly=None,
                retrieved_correctly=None,
                latency_s=round(latency, 2),
                usage=usage,
                cost_usd=round(cost, 5),
                answer=text,
                failures=failures,
            )
        )
        print(
            f"  {q['id']} {'naive':16s} {'PASS' if correct else 'FAIL'} "
            f"{latency:5.1f}s ${cost:.4f}  spent ${budget.spent:.2f}"
        )
    return results


def whole_doc_tokens(
    client: Any,
    service: DocumentService,
    raw: dict[str, str],
    strategy: str,
    doc_ids: dict[str, str],
) -> int:
    """Real prompt size, in the local model's tokens, for a whole-document strategy."""
    if strategy == "naive":
        chat = [{"role": "user", "content": raw["msa"]}]
    else:
        from longdoc import Strategy
        from longdoc.local import to_chat_messages

        qa = service._qa_sync(doc_ids["msa"])
        kwargs, _ = qa.build_request("question", Strategy.CACHED_DOCUMENT)
        chat = to_chat_messages(kwargs["system"], kwargs["messages"])
    return int(client.count_tokens(chat))


# -- report ----------------------------------------------------------------------------


def summarize(results: list[Result]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for strategy in dict.fromkeys(r.strategy for r in results):
        rs = [r for r in results if r.strategy == strategy]
        cites = [r.cited_correctly for r in rs if r.cited_correctly is not None]
        retr = [r.retrieved_correctly for r in rs if r.retrieved_correctly is not None]
        lat = sorted(r.latency_s for r in rs)
        tokens = {k: sum(r.usage[k] for r in rs) for k in rs[0].usage}
        prompt = sum(
            tokens[k]
            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        out[strategy] = {
            "questions": len(rs),
            "correct": sum(r.correct for r in rs),
            "accuracy": round(sum(r.correct for r in rs) / len(rs), 3),
            "citation_accuracy": round(sum(cites) / len(cites), 3) if cites else None,
            "retrieval_recall": round(sum(retr) / len(retr), 3) if retr else None,
            "latency_p50_s": round(statistics.median(lat), 2),
            "latency_max_s": lat[-1],
            "cost_usd": round(sum(r.cost_usd for r in rs), 4),
            "tokens": tokens,
            "cache_hit_ratio": round(tokens["cache_read_input_tokens"] / prompt, 3)
            if prompt
            else 0,
        }
    return out


def write_report(
    out_dir: Path,
    mode: str,
    summary: dict[str, dict[str, Any]],
    results: list[Result],
    questions: list[dict[str, Any]],
    pack_info: dict[str, Any],
    model_label: str = MODEL,
    skipped: dict[str, str] | None = None,
) -> Path:
    lines = [f"# Contract-review benchmark ({mode})", ""]
    if mode == "extractive":
        lines += [
            "> **Built-in extractive engine.** Answers are verbatim quoted passages chosen by word "
            "matching: no model, no API key, cost $0. It cannot reason, paraphrase or do "
            "arithmetic, so questions whose answer needs that, or whose wording shares no words "
            "with the clause, fail by design. Prompt-token and cost figures do not apply.",
            "",
        ]
    if mode == "local":
        lines += [
            f"> **Local model run.** Answers come from `{model_label}` running on this "
            "machine's CPU, not from Claude. Accuracy reflects that model; cost is $0.",
            "",
        ]
    if mode == "offline":
        lines += [
            "> **Offline plumbing check.** The model was a fake that returns a placeholder, "
            "so every answer fails by design. Accuracy, latency and cost below are NOT "
            "results. Run with `--live` for real numbers.",
            "",
        ]
    lines += [
        f"- Model: `{model_label}`" + (f", effort `{EFFORT}`" if mode == "live" else ""),
        f"- Retrieval: {pack_info.get('retriever', 'BM25 keyword')}",
        f"- Contract pack: {pack_info['pages']} pages, {pack_info['sections']} sections, "
        f"~{pack_info['approx_tokens']:,} tokens",
        f"- Run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        "| Strategy | Correct | Citations correct | Retrieval recall | p50 latency | Cost | Cache hit |",
        "|---|---|---|---|---|---|---|",
    ]
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    for s, m in summary.items():
        lines.append(
            f"| {s} | {m['correct']}/{m['questions']} ({m['accuracy']:.0%}) | "
            f"{pct(m['citation_accuracy'])} | {pct(m['retrieval_recall'])} | "
            f"{m['latency_p50_s']}s | ${m['cost_usd']:.2f} | {m['cache_hit_ratio']:.0%} |"
        )
    strategies = list(summary)
    lines += [
        "",
        "## Per question",
        "",
        "| ID | Category | " + " | ".join(strategies) + " |",
        "|---|---|" + "---|" * len(strategies),
    ]
    by = {(r.question_id, r.strategy): r for r in results}
    for q in questions:
        cells = []
        for s in strategies:
            r = by.get((q["id"], s))
            cells.append("-" if r is None else ("PASS" if r.correct else "FAIL"))
        lines.append(f"| {q['id']} | {q['category']} | " + " | ".join(cells) + " |")
    if skipped:
        lines += ["", "## Not run", ""]
        lines += [f"- **{k}**: {why}" for k, why in skipped.items()]
    failures = [r for r in results if not r.correct]
    if failures and mode in ("live", "openai", "local", "extractive"):
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [
                f"**{r.question_id} / {r.strategy}**: {'; '.join(r.failures)}",
                "",
                "> " + r.answer.replace("\n", "\n> ")[:1500],
                "",
            ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- main ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Real Claude calls (costs money)")
    mode.add_argument("--offline", action="store_true", help="Fake model; plumbing check only")
    mode.add_argument(
        "--openai",
        metavar="MODEL",
        help="Real OpenAI calls with YOUR key (costs money; needs --price-in/--price-out)",
    )
    mode.add_argument(
        "--extractive",
        action="store_true",
        help="Built-in engine that quotes passages: no model, no key, no cost, runs in seconds",
    )
    mode.add_argument(
        "--local", type=Path, metavar="MODEL.gguf", help="Local GGUF model; no API key, no cost"
    )
    parser.add_argument("--n-ctx", type=int, default=16_384, help="Local model context window")
    parser.add_argument("--env-file", help="Load OPENAI_API_KEY etc. from this .env file")
    parser.add_argument("--price-in", type=float, help="OpenAI: US$ per 1M input tokens")
    parser.add_argument("--price-out", type=float, help="OpenAI: US$ per 1M output tokens")
    parser.add_argument(
        "--price-cached-in",
        type=float,
        help="OpenAI: US$ per 1M cached input tokens (default: same as --price-in)",
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=4_000,
        help="OpenAI: cap on tokens per answer (reasoning models count thinking tokens too)",
    )
    parser.add_argument(
        "--embed-model",
        type=Path,
        metavar="EMBED.gguf",
        help="Local embedding model: use hybrid (keyword + semantic) retrieval",
    )
    parser.add_argument(
        "--kv-quant",
        choices=["f16", "q8", "q4"],
        default="f16",
        help="Compress the local model's KV cache so long contexts fit in memory",
    )
    parser.add_argument(
        "--naive-limit",
        type=int,
        default=0,
        help="Run the naive baseline on only the first N questions (0 = all)",
    )
    parser.add_argument("--max-cost", type=float, default=5.0, help="US$ cap for --live/--openai")
    parser.add_argument("--strategies", default="retrieval,cached_document,naive")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)

    if args.env_file:
        from longdoc.env import load_env_file
        from longdoc.exceptions import ConfigurationError

        try:
            load_env_file(args.env_file)
        except ConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    billed_run = bool(args.live or args.openai)
    if billed_run and os.environ.get("LONGDOC_ALLOW_LIVE_API") != "1":
        print(
            "error: --live and --openai make billed API calls. Set LONGDOC_ALLOW_LIVE_API=1 to "
            "confirm you intend that. Use --local or --offline for free runs.",
            file=sys.stderr,
        )
        return 1
    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: --live needs ANTHROPIC_API_KEY in the environment", file=sys.stderr)
        return 1
    if args.openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "error: --openai needs OPENAI_API_KEY in the environment (or --env-file). "
                "See docs/OPENAI.md.",
                file=sys.stderr,
            )
            return 1
        if args.price_in is None or args.price_out is None:
            print(
                "error: --openai needs --price-in and --price-out (US$ per 1M tokens, from "
                "your plan) so the --max-cost spending cap can be enforced.",
                file=sys.stderr,
            )
            return 1
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    questions = json.loads((HERE / "questions.json").read_text())["questions"]

    work = Path(tempfile.mkdtemp(prefix="longdoc-bench-"))
    pdfs = {"msa": work / "northwind_msa_pack.pdf", "nda": work / "fabrikam_nda.pdf"}
    build_contract_pack(pdfs["msa"])
    build_nda(pdfs["nda"])
    truth = {k: fact_sections(p) for k, p in pdfs.items()}
    missing = [f for f in FACTS if not (truth["msa"][f] or truth["nda"][f])]
    if missing:
        print(f"error: facts not found in any section: {missing}", file=sys.stderr)
        return 1

    # The naive baseline pastes raw extracted text, as a team without longdoc would.
    from longdoc.pdf import extract_pages

    raw = {k: "\n".join(extract_pages(p.read_bytes())) for k, p in pdfs.items()}
    pack = load_document(pdfs["msa"])
    pack_info = {
        "pages": len(pack.pages),
        "approx_tokens": len(pack.text) // 4,
        "sections": len(ParentChildChunker().chunk(pack).parents),
    }

    global BILLED, MODEL, PRICE_INPUT, PRICE_OUTPUT, PRICE_CACHE_READ, PRICE_CACHE_WRITE_5M
    global NAIVE_MAX_TOKENS
    run_mode = (
        "live"
        if args.live
        else "openai"
        if args.openai
        else "local"
        if args.local
        else "extractive"
        if args.extractive
        else "offline"
    )
    model_label = MODEL
    config = QAConfig(model=MODEL, effort=EFFORT)
    if args.live:
        from longdoc.anthropic_client import AnthropicClient

        sync_client: Any = AnthropicClient()
        async_client: Any = sync_client.to_async()
    elif args.openai:
        from longdoc.openai_client import OpenAIClient

        MODEL = model_label = args.openai
        PRICE_INPUT, PRICE_OUTPUT = args.price_in, args.price_out
        PRICE_CACHE_READ = args.price_cached_in if args.price_cached_in is not None else PRICE_INPUT
        PRICE_CACHE_WRITE_5M = PRICE_INPUT  # OpenAI does not bill a separate cache write
        NAIVE_MAX_TOKENS = args.max_answer_tokens
        sync_client = OpenAIClient()
        async_client = sync_client.to_async()
        config = QAConfig(model=MODEL, max_tokens=args.max_answer_tokens, fallbacks=None)
    elif args.local:
        from longdoc.local import AsyncLlamaCppClient, LlamaCppClient

        print(f"Loading local model {args.local} (n_ctx={args.n_ctx}) ...")
        kv: dict[str, Any] = {}
        if args.kv_quant != "f16":
            # Keys at q8_0 (8) hold quality; values at q4_0 (2) save the most memory.
            kv = {"flash_attn": True, "type_k": 8, "type_v": 8 if args.kv_quant == "q8" else 2}
        sync_client = LlamaCppClient(args.local, n_ctx=args.n_ctx, **kv)
        async_client = AsyncLlamaCppClient(sync_client)
        model_label = sync_client.model_name
        BILLED = False
        # Smaller prompts and answers: CPU inference is slow and context is limited.
        config = QAConfig(
            model=model_label,
            max_tokens=700,
            max_parents=3,
            max_context_chars=12_000,
            fallbacks=None,
        )
    elif args.extractive:
        from longdoc.extractive import ExtractiveClient

        sync_client = ExtractiveClient()
        async_client = sync_client.to_async()
        model_label = sync_client.model_name
        BILLED = False
        config = QAConfig(model=model_label, fallbacks=None)
    else:
        sync_client, async_client = FakeAnthropic(), FakeAnthropic(is_async=True)

    retriever_factory: Any = BM25Retriever
    retriever_label = "BM25 keyword"
    if args.embed_model:
        from longdoc.local import BGE_QUERY_PREFIX, LlamaCppEmbedder

        embedder = LlamaCppEmbedder(args.embed_model)
        prefix = BGE_QUERY_PREFIX if "bge" in args.embed_model.name.lower() else ""

        def retriever_factory() -> HybridRetriever:
            return HybridRetriever(
                BM25Retriever(), EmbeddingRetriever(embedder, query_prefix=prefix)
            )

        retriever_label = f"hybrid: BM25 + {args.embed_model.name}"
    pack_info["retriever"] = retriever_label

    service = DocumentService(
        store=JSONFileStore(work / "store"),
        config=config,
        client=sync_client,
        async_client=async_client,
        retriever_factory=retriever_factory,
    )
    budget = Budget(limit=args.max_cost if billed_run else float("inf"))
    skipped: dict[str, str] = {}
    # Conservative per-call estimates used only by the budget guard.
    doc_tokens = pack_info["approx_tokens"]
    est = {
        "retrieval": (12_000 * PRICE_INPUT + 3_000 * PRICE_OUTPUT) / 1e6,
        "cached_document": (doc_tokens * PRICE_CACHE_WRITE_5M + 3_000 * PRICE_OUTPUT) / 1e6,
        "naive": (doc_tokens * 1.1 * PRICE_INPUT + 3_000 * PRICE_OUTPUT) / 1e6,
    }

    results: list[Result] = []
    print(f"Contract pack: {pack_info}")
    try:
        with ServerThread(service) as url:
            # A cold 45k-token read on a CPU takes well over 10 minutes.
            remote = RemoteClient(
                url, api_key=API_TOKEN, timeout=7200 if args.local else 600, max_retries=0
            )
            doc_ids = {}
            for key, path in pdfs.items():
                info = remote.upload(path, document_id=f"bench-{key}")
                doc_ids[key] = info["id"]
                print(f"Uploaded {path.name}: {info}")
            for strategy in strategies:
                print(f"\n== {strategy} ==")
                needs_whole_doc = strategy in ("cached_document", "naive")
                if args.extractive and strategy == "naive":
                    skipped[strategy] = (
                        "pastes the raw document into a prompt for a language model; the "
                        "built-in engine has no model to read it."
                    )
                    print(f"  skipped: {skipped[strategy]}")
                    continue
                if args.local and needs_whole_doc:
                    # Use the model's own tokenizer: the chars/4 estimate is ~15% high here.
                    doc_tokens = whole_doc_tokens(sync_client, service, raw, strategy, doc_ids)
                if args.local and needs_whole_doc and doc_tokens + 1_000 > args.n_ctx:
                    skipped[strategy] = (
                        f"needs the whole ~{doc_tokens:,}-token document in every prompt, which "
                        f"exceeds the local model's {args.n_ctx:,}-token context window. Only "
                        "parent-child retrieval works at this document size."
                    )
                    print(f"  skipped: {skipped[strategy]}")
                    continue
                if strategy == "naive":
                    subset = questions[: args.naive_limit] if args.naive_limit else questions
                    if len(subset) < len(questions):
                        skipped["naive (partial)"] = (
                            f"run on the first {len(subset)} of {len(questions)} questions only, "
                            "because each naive call re-reads the whole document from scratch."
                        )
                    results += run_naive(sync_client, subset, raw, budget, est["naive"])
                else:
                    results += run_sdk_strategy(
                        remote, strategy, questions, doc_ids, truth, budget, est[strategy]
                    )
    except BudgetExceeded as exc:
        print(f"\n{exc}")

    if not results:
        print("No results recorded.")
        return 1
    summary = summarize(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / f"{stamp}-{run_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "mode": run_mode,
                "model": model_label,
                "skipped": skipped,
                "qa_config": {
                    "max_parents": config.max_parents,
                    "max_context_chars": config.max_context_chars,
                },
                "effort": EFFORT,
                "pack": pack_info,
                "summary": summary,
                "spent_usd": round(budget.spent, 4),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = write_report(
        out_dir, run_mode, summary, results, questions, pack_info, model_label, skipped
    )
    if args.local or args.extractive:
        print("\nCost: $0.")
    elif billed_run:
        print(f"\nTotal spent: ${budget.spent:.2f}")
    else:
        print(
            f"\nNothing spent (offline). Prompt-size cost projection: ${budget.spent:.2f} "
            "plus output tokens, which the fake model does not produce."
        )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
