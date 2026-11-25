"""Optional clean-up of retrieved context (``QAConfig(context_rot=True)``).

Three small, deterministic steps applied to the retrieved hits and sections:

1. **Relevance floor**: drop hits scoring below ``min_relevance`` of the best hit.
2. **Near-duplicate removal**: drop a hit whose text mostly repeats a higher-ranked hit
   (repeated boilerplate wastes the context budget).
3. **Edge ordering**: arrange the final sections so the strongest evidence sits at the start and
   end and the weakest in the middle, because language models tend to use the middle of a long
   prompt least reliably ("lost in the middle", Liu et al. 2023).

Status: this is **not measured to improve answers**. On the built-in engine, which reads every
section equally, it changes nothing (see ``benchmarks/contract_review/compare_features.py``). Any
benefit would show only with a language model, and has not been tested. Off by default.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import ParentChunk, RetrievalHit


def _words(text: str) -> set[str]:
    return set(text.lower().split())


def clean_hits(
    hits: Sequence[RetrievalHit],
    *,
    min_relevance: float = 0.25,
    dedup_threshold: float = 0.85,
) -> list[RetrievalHit]:
    """Apply the relevance floor and near-duplicate removal. Order (best first) is kept."""
    ranked = sorted(hits, key=lambda h: -h.score)
    if not ranked or ranked[0].score <= 0:
        return []
    floor = ranked[0].score * min_relevance
    kept: list[RetrievalHit] = []
    kept_words: list[set[str]] = []
    for hit in ranked:
        if hit.score < floor:
            break
        words = _words(hit.child.text)
        duplicate = any(
            len(words & other) / max(len(words), len(other), 1) >= dedup_threshold
            for other in kept_words
        )
        if not duplicate:
            kept.append(hit)
            kept_words.append(words)
    return kept


def order_by_strength(
    parents: Sequence[ParentChunk], hits: Sequence[RetrievalHit]
) -> tuple[ParentChunk, ...]:
    """Arrange sections strongest-first-and-last, weakest in the middle.

    A section's strength is its best hit's score. With ranks 1 (best) to n, the result is
    ``1, 3, 5, ..., 6, 4, 2``.
    """
    best: dict[str, float] = {}
    for hit in hits:
        pid = hit.child.parent_id
        best[pid] = max(best.get(pid, 0.0), hit.score)
    by_strength = sorted(parents, key=lambda p: (-best.get(p.id, 0.0), p.order))
    front, back = by_strength[0::2], by_strength[1::2]
    return tuple(front + back[::-1])
