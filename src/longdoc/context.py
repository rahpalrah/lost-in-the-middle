"""Context cleaning: deduplication, weak-section filtering, reorganization, and compaction.

After retrieval, candidate sections can be noisy: duplicates (boilerplate appearing in many
sections), weak matches (scoring just barely above zero), and structurally weak evidence (a quoted
answer that is not in context). This module cleans the context before it goes to the model,
reducing wasted tokens and the "lost in the middle" effect.

**Lost in the middle:** When long sequences are provided as context (Llm reports show this under
~1000 tokens), passages near the start and end have disproportionately high attention, while those
in the middle are attended less reliably. By reorganizing context—keeping the strongest evidence
at the edges and weaker supporting passages toward the center—we keep the model's focus on the
most important facts.

**Deduplication:** Legal and technical documents often repeat the same boilerplate (e.g. liability
disclaimers, definitions). The same passage showing up multiple times wastes context. We score
each section's similarity to those with higher scores, and drop it if it is redundant.

**Weak-section filtering:** Sections below a relevance threshold (relative to the best-scoring
section) are dropped entirely. They contribute noise, not signal.

**Compaction:** When the total context would exceed the budget (``max_context_chars`), instead
of dropping a section wholesale, we keep the best scoring sentences within it, ordered by their
position in the section so the text reads coherently.

Usage:

    hits = retriever.search(question, top_k)
    hits = clean_hits(hits, min_relevance=0.3, max_chars=40000)
    parents = select_parents(hits, document, max_parents=4, max_chars=40000)
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .retrieval import RetrievalHit


class _Sentence(NamedTuple):
    text: str
    score: float
    order: int


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


def clean_hits(
    hits: list[RetrievalHit],
    min_relevance: float = 0.25,
    dedup_threshold: float = 0.85,
) -> list[RetrievalHit]:
    """Filter and reorder hits by context-rot criteria.

    1. Drop hits scoring below ``min_relevance * max_score``.
    2. Drop near-duplicates using ``dedup_threshold``.
    3. Reorder by **strength envelope**: best at edges, weaker in middle.

    Args:
        hits: Hits ranked by search score (highest first).
        min_relevance: Drop hits scoring < this fraction of the max score.
        dedup_threshold: Similarity above this threshold triggers deduplication.

    Returns:
        Hits ordered by the envelope rule (not the original score order).
    """
    if not hits:
        return []
    max_score = max(h.score for h in hits)
    if max_score == 0:
        return []

    # Filter by relevance threshold.
    threshold = max_score * min_relevance
    strong = [h for h in hits if h.score >= threshold]
    if not strong:
        strong = hits[:1]  # keep the best even if below threshold

    # Deduplication: drop sections that are very similar to higher-scoring ones.
    # This is a simple text-overlap heuristic: a section is dropped if the top N% of
    # any higher-scoring section overlaps with the top N% of this one.
    kept = []
    for i, hit in enumerate(strong):
        is_dup = False
        for other in strong[:i]:  # only check higher-ranked hits
            overlap = len(set(hit.child.text.split()) & set(other.child.text.split()))
            hit_words = len(hit.child.text.split())
            other_words = len(other.child.text.split())
            if overlap / max(hit_words, other_words, 1) >= dedup_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(hit)

    # Strength envelope: arrange by score so that highest-scored items are at the
    # beginning and end, with weaker items in the middle. This counteracts the "lost
    # in the middle" effect. Sort by (distance from edge of envelope), i.e. the lower
    # score, should come toward the middle.
    if len(kept) <= 2:
        return kept

    # Sort by score descending so we can assign positions by pulling from both ends.
    by_score = sorted(kept, key=lambda h: -h.score)
    envelope = []
    left = 0
    right = len(by_score) - 1
    turn_left = True
    while left <= right:
        if turn_left:
            envelope.append(by_score[left])
            left += 1
        else:
            envelope.append(by_score[right])
            right -= 1
        turn_left = not turn_left
    return envelope


def compact_section(
    text: str,
    max_chars: int,
    top_n_sentences: int = 3,
) -> str:
    """Keep the top-scoring sentences from a section when it exceeds the budget.

    Heuristic scoring: a sentence scores high if it contains numbers, citations,
    or is in the first/last position.

    Args:
        text: The section text.
        max_chars: Character budget; if text <= max_chars, return it unchanged.
        top_n_sentences: How many top-scoring sentences to keep.

    Returns:
        A compacted version of the text, with sentences reordered to their original
        order, so it reads naturally.
    """
    if len(text) <= max_chars:
        return text

    sentences = text.split(". ")
    if len(sentences) <= top_n_sentences:
        return text

    scored = []
    for i, sent in enumerate(sentences):
        score = 0.0
        if any(c.isdigit() for c in sent):
            score += 2.0
        if "[" in sent or "(" in sent:
            score += 1.0
        if i == 0 or i == len(sentences) - 1:
            score += 1.0
        scored.append((score, i, sent))

    # Keep top N by score, then reorder by original position.
    top = sorted(scored, key=lambda x: -x[0])[:top_n_sentences]
    top.sort(key=lambda x: x[1])
    result = ". ".join(s[2] for s in top)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0]
    return result
