"""Context clean-up and recursive retrieval helpers (opt-in; see their module docs)."""

from __future__ import annotations

from longdoc import Document, DocumentQA, QAConfig
from longdoc.context import clean_hits, order_by_strength
from longdoc.extractive import ExtractiveClient
from longdoc.models import ChildChunk, ParentChunk, RetrievalHit
from longdoc.recursive import decompose_heuristic, merge_hits


def hit(cid: str, parent: str, text: str, score: float) -> RetrievalHit:
    child = ChildChunk(id=cid, parent_id=parent, text=text, order=0)
    return RetrievalHit(child=child, score=score)


def test_clean_hits_applies_floor_and_removes_duplicates() -> None:
    hits = [
        hit("c1", "p1", "alpha beta gamma delta", 10.0),
        hit("c2", "p2", "alpha beta gamma delta", 9.0),  # repeats c1
        hit("c3", "p3", "completely different words here", 8.0),
        hit("c4", "p4", "weak match only", 1.0),  # below 25% of best
    ]
    assert [h.child.id for h in clean_hits(hits)] == ["c1", "c3"]
    assert clean_hits([]) == [] and clean_hits([hit("c", "p", "x", 0.0)]) == []


def test_order_by_strength_puts_the_best_at_the_edges() -> None:
    parents = [ParentChunk(id=f"p{i}", text="t", order=i) for i in range(1, 6)]
    hits = [hit(f"c{i}", f"p{i}", "t", 6 - i) for i in range(1, 6)]
    ordered = [p.id for p in order_by_strength(parents, hits)]
    assert ordered == ["p1", "p3", "p5", "p4", "p2"]  # ranks 1,3,5,4,2: best first, 2nd best last
    assert order_by_strength(parents[:1], hits) == (parents[0],)


def test_decompose_and_merge() -> None:
    parts = decompose_heuristic("What is the notice period and what is the liability cap?")
    assert len(parts) == 2
    assert len(decompose_heuristic("When is payment due?")) == 1
    merged = merge_hits(
        [hit("c1", "p", "a", 1.0), hit("c1", "p", "a", 3.0), hit("c2", "p", "b", 2.0)], max_total=5
    )
    assert [(h.child.id, h.score) for h in merged] == [("c1", 3.0), ("c2", 2.0)]


def test_options_run_end_to_end_and_default_to_off() -> None:
    assert QAConfig().context_rot is False and QAConfig().recursive is False
    text = "# 1. Notice\n\nNotice is 90 days.\n\n# 2. Fees\n\nFees are due in 30 days.\n"
    qa = DocumentQA(
        Document.from_text(text),
        client=ExtractiveClient(),
        config=QAConfig(fallbacks=None, context_rot=True, recursive=True),
    )
    answer = qa.ask("What is the notice period and when are fees due?")
    assert "90 days" in answer.text
