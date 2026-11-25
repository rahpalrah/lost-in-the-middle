"""Recursive section tree and sub-question decomposition for multi-part retrieval.

When a question asks about multiple topics spread across a document (e.g., "What are the notice
period for Service A and the liability cap for Service B?"), naive retrieval may crowd out one
topic when the top-k results all address the other. This module decomposes the question into
sub-questions, searches each independently, and merges the results.

The simplest decomposition:
1. Use the built-in engine (or a cheap LLM) to split the question into independent sub-questions.
2. Retrieve top-k results for each.
3. Merge by ranking (highest score first) and deduplication.

A more advanced strategy (not yet implemented) builds a tree of sections (recursively chunking
coarse sections into finer ones) and searches from top down, pruning branches that don't match
the sub-question keywords.

Usage:

    decomposer = QuestionDecomposer()  # built-in heuristic
    sub_questions = decomposer.decompose(question)  # ["notice period?", "liability cap?"]

    all_hits = []
    for sq in sub_questions:
        hits = retriever.search(sq, top_k=12)
        all_hits.extend(hits)

    merged = merge_hits(all_hits, max_total=20)
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .retrieval import RetrievalHit


class SubQuestion(NamedTuple):
    """One independent part of a decomposed question."""

    text: str
    keywords: list[str]
    importance: float = 1.0


def decompose_heuristic(question: str) -> list[SubQuestion]:
    """Heuristically split a question into independent sub-questions.

    Looks for patterns like:
    - "X and Y" / "X, Y, and Z"
    - "between X and Y"
    - Multiple questions separated by "?"

    Returns a list of sub-questions, each with extracted keywords. If no pattern is found,
    returns the original question as a single sub-question.
    """
    q = question.strip()

    # Pattern: "What are ... and ..."
    if " and " in q and "what" in q.lower():
        parts = q.split(" and ")
        if len(parts) == 2:
            return [
                SubQuestion(text=parts[0].replace("what are ", "what is ", 1),
                           keywords=_extract_keywords(parts[0])),
                SubQuestion(text=parts[1], keywords=_extract_keywords(parts[1])),
            ]

    # Pattern: Multiple questions in one (separated by ?)
    if q.count("?") > 1:
        parts = [p.strip() for p in q.split("?") if p.strip()]
        return [SubQuestion(text=p + "?", keywords=_extract_keywords(p)) for p in parts[:3]]

    # Default: treat as a single question
    return [SubQuestion(text=q, keywords=_extract_keywords(q))]


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a question or phrase."""
    # Remove common question words and punctuation.
    words = re.findall(r'\b[a-z_-]{3,}\b', text.lower())
    stop = {
        "what", "is", "are", "the", "for", "and", "or", "in", "on", "at", "to", "of",
        "how", "which", "can", "must", "does", "do", "did", "has", "have", "been",
        "will", "would", "should", "could", "may", "might", "shall",
    }
    return [w for w in words if w not in stop][:5]  # limit to top 5


def merge_hits(
    all_hits: list[RetrievalHit],
    max_total: int = 20,
    dedup_by_id: bool = True,
) -> list[RetrievalHit]:
    """Merge hits from multiple sub-question searches.

    Deduplicates by child ID (keeps the highest-scoring instance) and re-ranks by score.

    Args:
        all_hits: Combined results from all sub-question searches.
        max_total: Return at most this many hits (after deduplication).
        dedup_by_id: If True, keep only the highest score for each unique child ID.

    Returns:
        Merged and ranked hits.
    """
    if not all_hits:
        return []

    if dedup_by_id:
        best_by_id: dict[str, RetrievalHit] = {}
        for hit in all_hits:
            cid = hit.child.id
            if cid not in best_by_id or hit.score > best_by_id[cid].score:
                best_by_id[cid] = hit
        all_hits = list(best_by_id.values())

    # Sort by score (descending) and return top N.
    all_hits.sort(key=lambda h: -h.score)
    return all_hits[:max_total]
