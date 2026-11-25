"""Does context-rot cleaning / recursive retrieval help on the contract benchmark?

Runs the 16 contract questions through the built-in engine under each setting and reports answer
accuracy (every expected fact present) and retrieval hit (the planted fact's section was
retrieved). Unlike the needle test, the baseline here is not at 100%, so a gain or loss shows.

    python benchmarks/contract_review/compare_features.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from generate_pack import FACTS, build_contract_pack, build_nda  # noqa: E402

from longdoc import DocumentQA, QAConfig, load_document  # noqa: E402
from longdoc.extractive import NOT_FOUND_TEXT, ExtractiveClient  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="longdoc-cmp-"))
    build_contract_pack(work / "msa.pdf")
    build_nda(work / "nda.pdf")
    docs = {k: load_document(work / f"{k}.pdf") for k in ("msa", "nda")}
    questions = json.loads((HERE / "questions.json").read_text())["questions"]
    settings = {
        "baseline": {},
        "context_rot": {"context_rot": True},
        "recursive": {"recursive": True},
        "both": {"context_rot": True, "recursive": True},
    }
    print(f"{'setting':<12} {'answers':>8} {'retrieval':>10}  failed")
    for name, extra in settings.items():
        qas = {
            k: DocumentQA(d, client=ExtractiveClient(), config=QAConfig(fallbacks=None, **extra))
            for k, d in docs.items()
        }
        ok = hit = hit_n = 0
        failed = []
        for q in questions:
            qa = qas[q["doc"]]
            text = norm(qa.ask(q["question"]).text)
            if q.get("expect_not_found"):
                good = NOT_FOUND_TEXT.lower() in text
            else:
                good = all(any(norm(alt) in text for alt in group) for group in q["expect"])
            ok += good
            if not good:
                failed.append(q["id"])
            if q.get("facts"):
                need = {
                    p.id
                    for p in qa.chunks.parents
                    for f in q["facts"]
                    if norm(FACTS[f])[:60] in norm(p.text)
                }
                got = {p.id for p in qa.retrieve(q["question"])}
                hit_n += 1
                hit += bool(need & got)
        print(f"{name:<12} {ok:>5}/{len(questions)} {hit:>7}/{hit_n}  {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
