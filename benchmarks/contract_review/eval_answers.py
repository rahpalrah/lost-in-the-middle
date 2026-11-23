"""Answer-quality metrics for saved benchmark runs. No model and no API key needed.

Reads ``results.json`` files written by ``run.py`` and re-derives, deterministically, the
context each question was given. Metrics (names follow common RAG-evaluation practice):

* Answer accuracy         - required facts present, nothing forbidden (the run's own grader)
* Conditional accuracy    - accuracy when the right section *was* in the context. Separates
                            generation errors from retrieval errors.
* Citation recall/precision - section-level, in the spirit of ALCE (Gao et al., 2023):
                            recall = share of needed sections cited; precision = share of
                            cited sections that are needed.
* Groundedness (lexical)  - share of answers whose numbers all appear in the context. A cheap
                            proxy for faithfulness, NOT an LLM judge: it cannot see wrong
                            reasoning, only invented figures.
* Abstention              - when the context lacks the answer, does the model say so
                            (good) or answer anyway (hallucination)? And does it wrongly
                            abstain when the answer was there?
* Context compression     - tokens sent per question vs sending the whole document.

Usage:
    python benchmarks/contract_review/eval_answers.py RESULTS.json[:retriever] ... --models-dir models
        retriever = bm25 (default) | hybrid
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_pack import FACTS, build_contract_pack, build_nda  # noqa: E402
from run import NOT_FOUND  # noqa: E402

from longdoc import (  # noqa: E402
    BM25Retriever,
    DocumentQA,
    EmbeddingRetriever,
    HybridRetriever,
    QAConfig,
    load_document,
)
from longdoc.local import BGE_QUERY_PREFIX, LlamaCppEmbedder  # noqa: E402
from longdoc.prompts import clean_answer, extract_citations  # noqa: E402

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
SECTION_ID = re.compile(r"p\d{4}(?:-c\d{3})?")
WHOLE_DOC_TOKENS = 45_646  # measured with the local model's tokenizer (see benchmark README)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; better than normal-approx at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def numbers(text: str) -> set[str]:
    text = SECTION_ID.sub(" ", text)
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="results.json[:bm25|hybrid]")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=HERE / "eval" / "answer_metrics.json")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="longdoc-eval-"))
    build_contract_pack(work / "msa.pdf")
    build_nda(work / "nda.pdf")
    docs = {k: load_document(work / f"{k}.pdf") for k in ("msa", "nda")}
    questions = {q["id"]: q for q in json.loads((HERE / "questions.json").read_text())["questions"]}
    bge = LlamaCppEmbedder(args.models_dir / "bge-small-en-v1.5-q8_0.gguf")
    norm = lambda s: " ".join(s.split())  # noqa: E731

    report: dict[str, Any] = {}
    for spec in args.runs:
        path, _, kind = spec.partition(":")
        kind = kind or "bm25"
        data = json.loads(Path(path).read_text())
        config = QAConfig(max_parents=3, max_context_chars=12_000, fallbacks=None)

        def make(kind: str = kind) -> Any:
            if kind == "hybrid":
                return HybridRetriever(
                    BM25Retriever(), EmbeddingRetriever(bge, query_prefix=BGE_QUERY_PREFIX)
                )
            return BM25Retriever()

        qa = {
            k: DocumentQA(d, config=config, retriever=make(), client=object())
            for k, d in docs.items()
        }
        needed: dict[str, dict[str, set[str]]] = {
            k: {
                key: {p.id for p in q.chunks.parents if norm(t)[:60] in norm(p.text)}
                for key, t in FACTS.items()
            }
            for k, q in qa.items()
        }

        rows = []
        for r in data["results"]:
            q = questions[r["question_id"]]
            ctx = qa[q["doc"]].retrieve(q["question"])
            ctx_ids = {p.id for p in ctx}
            # What the model actually saw: section text plus the heading and page labels in
            # each <section> tag (answers legitimately quote page numbers from those labels).
            ctx_text = " ".join(f"{p.heading or ''} {p.page_label or ''} {p.text}" for p in ctx)
            need = (
                set().union(*(needed[q["doc"]][f] for f in q.get("facts", [])))
                if q.get("facts")
                else set()
            )
            cited_stored = set(r["citations"])  # parsed by the SDK version that ran the benchmark
            # Re-parse with the current SDK (accepts bare IDs and echoed <section> tags too).
            cited = {
                c.id for c in extract_citations(clean_answer(r["answer"]), qa[q["doc"]].chunks)
            }
            ctx_has_answer = bool(need & ctx_ids) if need else None
            stored = r.get("retrieved_correctly")
            ungrounded = numbers(r["answer"]) - numbers(ctx_text) - numbers(q["question"])
            rows.append(
                {
                    "id": r["question_id"],
                    "correct": r["correct"],
                    "ctx_has_answer": ctx_has_answer,
                    "consistent_with_run": (stored is None) or (stored == ctx_has_answer),
                    "abstained": bool(NOT_FOUND.search(r["answer"])),
                    "unanswerable": bool(q.get("expect_not_found")),
                    "cite_recall": len(need & cited) / len(need) if need else None,
                    "cite_precision": len(need & cited) / len(cited) if need and cited else None,
                    "cited_any_needed": bool(need & cited) if need else None,
                    "cited_any_needed_as_stored": bool(need & cited_stored) if need else None,
                    "ungrounded": sorted(ungrounded),
                    "tokens_in": r["usage"]["input_tokens"],
                }
            )

        n = len(rows)
        answerable = [x for x in rows if not x["unanswerable"] and x["ctx_has_answer"] is not None]
        in_ctx = [x for x in answerable if x["ctx_has_answer"]]
        not_in_ctx = [x for x in answerable if not x["ctx_has_answer"]]
        unans = [x for x in rows if x["unanswerable"]]
        with_cites = [x for x in answerable if x["cite_recall"] is not None]
        prec = [x["cite_precision"] for x in answerable if x["cite_precision"] is not None]
        avg_tokens = sum(x["tokens_in"] for x in rows) / n
        report[f"{Path(path).parent.name} [{kind}]"] = {
            "questions": n,
            "context_reconstruction_matches_run": all(x["consistent_with_run"] for x in rows),
            "answer_accuracy": rate(sum(x["correct"] for x in rows), n),
            "conditional_accuracy_when_context_has_answer": rate(
                sum(x["correct"] for x in in_ctx), len(in_ctx)
            ),
            "retrieval_hit_rate": rate(len(in_ctx), len(answerable)),
            "citation_any_needed_section": rate(
                sum(x["cited_any_needed"] for x in with_cites), len(with_cites)
            ),
            "citation_any_needed_section_as_stored": rate(
                sum(x["cited_any_needed_as_stored"] for x in with_cites), len(with_cites)
            ),
            "citation_recall_strict_mean": round(
                sum(x["cite_recall"] for x in with_cites) / len(with_cites), 3
            ),
            "citation_precision_mean": round(sum(prec) / len(prec), 3) if prec else None,
            "answers_with_ungrounded_numbers": rate(sum(bool(x["ungrounded"]) for x in rows), n),
            "ungrounded_examples": {x["id"]: x["ungrounded"] for x in rows if x["ungrounded"]},
            "abstains_correctly_when_unanswerable": rate(
                sum(x["abstained"] for x in unans), len(unans)
            ),
            "hallucinated_when_context_lacks_answer": rate(
                sum(not x["abstained"] for x in not_in_ctx), len(not_in_ctx)
            ),
            "wrongly_abstained_when_context_has_answer": rate(
                sum(x["abstained"] for x in in_ctx), len(in_ctx)
            ),
            "avg_prompt_tokens": round(avg_tokens),
            "context_compression_vs_whole_document": round(WHOLE_DOC_TOKENS / avg_tokens, 1),
            "per_question": rows,
        }
        s = report[f"{Path(path).parent.name} [{kind}]"]
        print(f"\n== {Path(path).parent.name} [{kind}] ==")
        for key in (
            "answer_accuracy",
            "conditional_accuracy_when_context_has_answer",
            "retrieval_hit_rate",
            "citation_any_needed_section",
            "citation_any_needed_section_as_stored",
            "answers_with_ungrounded_numbers",
            "abstains_correctly_when_unanswerable",
            "hallucinated_when_context_lacks_answer",
            "wrongly_abstained_when_context_has_answer",
        ):
            v = s[key]
            print(f"  {key:46s} {v['k']}/{v['n']} = {v['rate']}  CI95 {v['ci95']}")
        print(
            f"  citation recall (strict) {s['citation_recall_strict_mean']}  "
            f"precision {s['citation_precision_mean']}  "
            f"compression {s['context_compression_vs_whole_document']}x  "
            f"reconstruction ok: {s['context_reconstruction_matches_run']}"
        )
        print("  ungrounded numbers:", s["ungrounded_examples"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
