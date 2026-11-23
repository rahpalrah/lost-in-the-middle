"""Load test for the longdoc HTTP API, using only the standard library.

By default it starts the built-in server in this process (built-in answer engine, no key, no
network beyond localhost), indexes a synthetic ~150-page contract, then drives ``/ask`` with an
increasing number of concurrent clients and reports throughput and latency percentiles per level.

    python benchmarks/load_test.py                       # self-contained
    python benchmarks/load_test.py --url http://host:8000 --api-key ... --levels 1,8,32 --seconds 20

Point ``--url`` at a deployed instance to test the real stack (a real model's latency will dominate;
the built-in engine measures the service itself: parsing, retrieval, locking, HTTP).
Exit status is 1 when any level has errors other than the deliberate 503/429 back-pressure, or
when ``--max-p95-ms`` is exceeded.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longdoc import DocumentService
from longdoc.remote import RemoteClient, RemoteError
from longdoc.server import create_server

TOPICS = [
    (
        "Payment",
        "Invoices are payable within {n} days of the invoice date. Late amounts accrue "
        "interest at {m} percent per month.",
    ),
    (
        "Termination",
        "Either party may terminate this Agreement for convenience by giving {n} days "
        "prior written notice.",
    ),
    ("Liability", "Total liability under this Agreement is capped at {n} times the annual fees."),
    (
        "Confidentiality",
        "Confidential information must be protected for {n} years after termination.",
    ),
    ("Audit", "The customer may audit the supplier no more than once every {n} months."),
    ("Insurance", "The supplier shall maintain insurance of not less than {n} million dollars."),
    ("Governing law", "This Agreement is governed by the laws of the State number {n}."),
    ("Data protection", "Personal data must be deleted within {n} days of a written request."),
]
FILLER = (
    "The parties acknowledge that the provisions of this section are material and that each has "
    "relied on them in entering into this Agreement. No waiver of any provision shall be "
    "effective unless made in writing and signed by the waiving party. "
)
QUESTIONS = [
    "How many days notice are needed to terminate for convenience?",
    "When must invoices be paid?",
    "What is the liability cap?",
    "How long must confidential information be protected?",
    "How often may the customer audit the supplier?",
    "How much insurance must the supplier maintain?",
    "Within how many days must personal data be deleted?",
    "What carbon emissions reporting is required?",  # deliberately unanswerable
]


def synthetic_contract(sections: int = 400, seed: int = 7) -> str:
    """A deterministic contract of roughly 150 pages of text (about 60k words)."""
    rng = random.Random(seed)
    parts = []
    for i in range(1, sections + 1):
        title, body = TOPICS[i % len(TOPICS)]
        text = body.format(n=rng.randint(5, 120), m=rng.randint(1, 3))
        filler = FILLER * rng.randint(8, 14)
        parts.append(f"# {i}. {title} ({i})\n\n{text} {filler}\n")
    return "\n".join(parts)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    k = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[k]


def run_level(
    url: str, api_key: str | None, doc_id: str, concurrency: int, seconds: float
) -> dict[str, object]:
    latencies: list[float] = []
    outcomes: Counter[str] = Counter()
    lock = threading.Lock()
    stop_at = time.monotonic() + seconds

    def worker(seed: int) -> None:
        # max_retries=0: a load test must see 503/429 rather than hide them behind retries
        client = RemoteClient(url, api_key=api_key, max_retries=0, timeout=120, allow_insecure=True)
        rng = random.Random(seed)
        while time.monotonic() < stop_at:
            question = rng.choice(QUESTIONS)
            began = time.perf_counter()
            try:
                answer = client.ask(doc_id, question)
                ok = bool(answer.get("text"))
                label = "ok" if ok else "empty_answer"
            except RemoteError as exc:
                label = f"http_{exc.status}_{exc.code}" if exc.status else "connection_error"
            elapsed = (time.perf_counter() - began) * 1000
            with lock:
                outcomes[label] += 1
                if label == "ok":
                    latencies.append(elapsed)

    began = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - began
    latencies.sort()
    total = sum(outcomes.values())
    return {
        "concurrency": concurrency,
        "seconds": round(wall, 1),
        "requests": total,
        "throughput_rps": round(outcomes["ok"] / wall, 1),
        "outcomes": dict(outcomes),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "p99": round(percentile(latencies, 0.99), 1),
            "max": round(latencies[-1], 1) if latencies else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", help="Test this running server instead of starting one")
    ap.add_argument("--api-key", help="Bearer token for --url (default: none)")
    ap.add_argument("--levels", default="1,4,16,32", help="Comma-separated concurrent clients")
    ap.add_argument("--seconds", type=float, default=10.0, help="Duration of each level")
    ap.add_argument(
        "--max-concurrency", type=int, default=64, help="Server limit (self-hosted run)"
    )
    ap.add_argument("--max-p95-ms", type=float, help="Fail if any level's p95 exceeds this")
    ap.add_argument("--out", type=Path, help="Write the JSON report here")
    args = ap.parse_args(argv)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    contract = synthetic_contract()
    server = None
    url, key = args.url, args.api_key
    if url is None:
        service = DocumentService()
        # no rate limit here: this measures capacity, not the limiter (which has its own tests)
        server = create_server(
            service,
            port=0,
            api_key="load-test-token",
            rate_limit_per_minute=0,
            max_concurrency=args.max_concurrency,
        )
        server.start_in_thread()
        url, key = server.url, "load-test-token"

    try:
        setup = RemoteClient(url, api_key=key, allow_insecure=True)
        doc_id = "load-test-contract"
        began = time.perf_counter()
        info = setup.add_text(contract, source="synthetic-contract.txt", document_id=doc_id)
        index_s = time.perf_counter() - began
        print(
            f"indexed {len(contract):,} characters / {info.get('sections', '?')} sections "
            f"in {index_s:.2f}s -> {url}"
        )
        results = []
        print(
            f"{'clients':>7} {'req':>7} {'rps':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}  outcomes"
        )
        for level in levels:
            r = run_level(url, key, doc_id, level, args.seconds)
            results.append(r)
            lat = r["latency_ms"]
            assert isinstance(lat, dict)
            print(
                f"{level:>7} {r['requests']:>7} {r['throughput_rps']:>8} {lat['p50']:>8} "
                f"{lat['p95']:>8} {lat['p99']:>8}  {r['outcomes']}"
            )
        setup.delete_document(doc_id)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

    report = {
        "target": "self-hosted built-in server" if args.url is None else "external",
        "engine": "built-in extractive" if args.url is None else "whatever the server uses",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "document_chars": len(contract),
        "index_seconds": round(index_s, 2),
        "levels": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    bad = 0
    for r in results:
        allowed = {"ok", "http_503_server_busy", "http_429_rate_limited"}
        unexpected = {k: v for k, v in r["outcomes"].items() if k not in allowed}  # type: ignore[attr-defined]
        if unexpected:
            print(f"FAIL at {r['concurrency']} clients: unexpected outcomes {unexpected}")
            bad = 1
        lat = r["latency_ms"]
        if args.max_p95_ms and isinstance(lat, dict) and (lat["p95"] or 0) > args.max_p95_ms:
            print(f"FAIL at {r['concurrency']} clients: p95 {lat['p95']} ms > {args.max_p95_ms} ms")
            bad = 1
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
