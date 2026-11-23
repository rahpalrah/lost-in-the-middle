"""Retrieval quality with standard information-retrieval metrics. No model, no API key.

For each answerable question the *relevant* parent sections are the ones that contain the
planted fact(s). Each retriever ranks child passages; they are collapsed to parent
sections in rank order (the unit the model actually reads), then scored with:

* Hit@k      - is at least one relevant section in the top k? (a.k.a. success@k)
* Recall@k   - share of the relevant sections found in the top k
* MRR        - mean reciprocal rank of the first relevant section
* nDCG@k     - rank-weighted gain with binary relevance (BEIR / MTEB convention)

With only 15 queries, differences of one question are noise, so every headline number
carries a 95% bootstrap confidence interval (10,000 resamples of the questions).

Usage:
    python benchmarks/contract_review/eval_retrieval.py --models-dir models
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_pack import FACTS, build_contract_pack, build_nda  # noqa: E402

from longdoc import (  # noqa: E402
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
    ParentChildChunker,
    load_document,
)
from longdoc.local import BGE_QUERY_PREFIX, LlamaCppEmbedder  # noqa: E402

KS = (1, 3, 4, 5, 10)
SEED = 20261123


def rank_parents(hits: list[Any]) -> list[str]:
    """Collapse ranked child hits to parent IDs, keeping first-appearance order."""
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h.child.parent_id, None)
    return list(seen)


def metrics_for(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    first = next((i for i, p in enumerate(ranked, 1) if p in relevant), None)
    out["mrr"] = 1.0 / first if first else 0.0
    for k in KS:
        top = ranked[:k]
        found = len(set(top) & relevant)
        out[f"hit@{k}"] = 1.0 if found else 0.0
        out[f"recall@{k}"] = found / len(relevant)
        dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(top, 1) if p in relevant)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return out


def bootstrap_ci(values: list[float], rng: random.Random, n: int = 10_000) -> tuple[float, float]:
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=HERE / "eval" / "retrieval_metrics.json")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="longdoc-eval-"))
    build_contract_pack(work / "msa.pdf")
    build_nda(work / "nda.pdf")
    chunked = {
        k: ParentChildChunker().chunk(load_document(work / f"{k}.pdf")) for k in ("msa", "nda")
    }
    norm = lambda s: " ".join(s.split())  # noqa: E731
    fact_parents = {
        doc: {
            key: {p.id for p in ch.parents if norm(text)[:60] in norm(p.text)}
            for key, text in FACTS.items()
        }
        for doc, ch in chunked.items()
    }
    questions = [
        q for q in json.loads((HERE / "questions.json").read_text())["questions"] if q.get("facts")
    ]

    bge = LlamaCppEmbedder(args.models_dir / "bge-small-en-v1.5-q8_0.gguf")
    nomic = LlamaCppEmbedder(args.models_dir / "nomic-embed-text-v1.5.Q8_0.gguf", n_ctx=2048)

    def emb_bge() -> EmbeddingRetriever:
        return EmbeddingRetriever(bge, query_prefix=BGE_QUERY_PREFIX)

    def emb_nomic() -> EmbeddingRetriever:
        return EmbeddingRetriever(
            nomic, query_prefix="search_query: ", document_prefix="search_document: "
        )

    systems = {
        "bm25": BM25Retriever,
        "embedding (bge-small)": emb_bge,
        "embedding (nomic-embed)": emb_nomic,
        "hybrid (bm25 + bge-small)": lambda: HybridRetriever(BM25Retriever(), emb_bge()),
        "hybrid (bm25 + nomic)": lambda: HybridRetriever(BM25Retriever(), emb_nomic()),
    }

    rng = random.Random(SEED)
    report: dict[str, Any] = {
        "n_queries": len(questions),
        "ks": KS,
        "seed": SEED,
        "note": "Relevance = parent sections containing the planted fact(s). Synthetic corpus.",
        "systems": {},
    }
    for name, make in systems.items():
        retrievers = {}
        for doc, ch in chunked.items():
            r = make()
            r.index(ch.children)
            retrievers[doc] = r
        per_q: list[dict[str, float]] = []
        misses = []
        for q in questions:
            relevant = set().union(*(fact_parents[q["doc"]][f] for f in q["facts"]))
            ranked = rank_parents(retrievers[q["doc"]].search(q["question"], 400))
            m = metrics_for(ranked, relevant)
            per_q.append(m)
            if m["hit@4"] == 0:
                misses.append(q["id"])
        summary: dict[str, Any] = {"misses_at_4": misses}
        for key in per_q[0]:
            vals = [m[key] for m in per_q]
            mean = sum(vals) / len(vals)
            lo, hi = bootstrap_ci(vals, rng)
            summary[key] = {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        report["systems"][name] = summary
        print(
            f"{name:28s} hit@4 {summary['hit@4']['mean']:.2f} "
            f"[{summary['hit@4']['ci95'][0]:.2f}-{summary['hit@4']['ci95'][1]:.2f}]  "
            f"recall@4 {summary['recall@4']['mean']:.2f}  MRR {summary['mrr']['mean']:.2f}  "
            f"nDCG@4 {summary['ndcg@4']['mean']:.2f}  misses {misses}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
