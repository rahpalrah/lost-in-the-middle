"""Retrieval over child chunks.

:class:`Retriever` is a :class:`typing.Protocol`, so you can plug in any search
backend (a vector database, an embedding model, a hybrid search service) by
implementing two methods. :class:`BM25Retriever` is the dependency-free default.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from .models import ChildChunk, ChunkedDocument, ParentChunk, RetrievalHit

_TOKEN = re.compile(r"\w+", re.UNICODE)

# A short English stop-word list; kept small so domain terms are never dropped.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
        "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
        "their", "there", "these", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "why", "will", "with", "does", "do", "did", "how", "can", "could",
        "should", "would", "may", "might", "shall",
    }
)  # fmt: skip


def stem(token: str) -> str:
    """Conservative English suffix stripping so word forms match.

    "governs", "governed" and "governing" all become "govern"; "increases" and
    "increase" both become "increas". Short words and numbers are left alone.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("al", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)] + keep
            break
    else:
        if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            token = token[:-1]
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case, stemmed word tokens with stop words removed."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@runtime_checkable
class Retriever(Protocol):
    """Anything that can index child chunks and search them."""

    def index(self, children: Sequence[ChildChunk]) -> None:
        """(Re)build the index over ``children``."""
        ...

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """Return up to ``k`` hits, best first."""
        ...


class BM25Retriever:
    """Okapi BM25 keyword search, implemented in pure Python.

    Good at exact terms (clause numbers, defined terms, names). For paraphrased
    questions, plug in an embedding-based :class:`Retriever` instead.

    Args:
        k1: Term-frequency saturation.
        b: Length normalisation.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._children: list[ChildChunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0

    def index(self, children: Sequence[ChildChunk]) -> None:
        """Build the index."""
        self._children = list(children)
        self._tfs = [
            Counter(tokenize(f"{c.heading} {c.text}" if c.heading else c.text))
            for c in self._children
        ]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        n = len(self._children)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """Score every child against ``query``; return the top ``k`` with score > 0."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms or k <= 0:
            return []
        scored: list[RetrievalHit] = []
        for child, tf, length in zip(self._children, self._tfs, self._lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1.0))
            score = sum(
                self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append(RetrievalHit(child=child, score=score))
        scored.sort(key=lambda h: (-h.score, h.child.id))
        return scored[:k]


def select_parents(
    hits: Sequence[RetrievalHit],
    document: ChunkedDocument,
    *,
    max_parents: int,
    max_chars: int,
) -> tuple[ParentChunk, ...]:
    """Map child hits to their parent sections.

    Parents are ranked by their best child's score, deduplicated, and added until
    ``max_parents`` or ``max_chars`` is reached (the top-ranked parent is always
    included). The result is returned in document order, which reads more naturally
    and keeps prompts deterministic.
    """
    ranked: list[ParentChunk] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.child.parent_id in seen:
            continue
        parent = document.get_parent(hit.child.parent_id)
        if parent is None:
            continue
        seen.add(parent.id)
        ranked.append(parent)

    chosen: list[ParentChunk] = []
    used = 0
    for parent in ranked:
        if len(chosen) >= max_parents:
            break
        if chosen and used + len(parent.text) > max_chars:
            continue
        chosen.append(parent)
        used += len(parent.text)
    return tuple(sorted(chosen, key=lambda p: p.order))


# -- semantic and hybrid retrieval ----------------------------------------------------

Embedder = Callable[[list[str]], Sequence[Sequence[float]]]
"""Turns a batch of texts into vectors, one per text, in order."""


def _normalise_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class EmbeddingRetriever:
    """Semantic search: finds passages that *mean* the same as the question.

    Catches paraphrases keyword search misses ("the vendor is acquired" vs "change of
    control"). Works with any embedding model through a plain function, for example
    :class:`longdoc.local.LlamaCppEmbedder` (local, free) or a hosted embeddings API.
    Vectors are kept in memory; plug in a vector database for very large collections.

    Args:
        embed: Function from a list of texts to a list of vectors.
        query_prefix: Text prepended to queries. Some models need one; for BGE models
            use ``"Represent this sentence for searching relevant passages: "``.
        document_prefix: Text prepended to passages before embedding.
        batch_size: Passages embedded per call.
    """

    def __init__(
        self,
        embed: Embedder,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self.embed = embed
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = max(1, batch_size)
        self._children: list[ChildChunk] = []
        self._vectors: list[list[float]] = []

    def index(self, children: Sequence[ChildChunk]) -> None:
        """Embed every passage (with its section heading)."""
        self._children = list(children)
        texts = [
            f"{self.document_prefix}{c.heading + '. ' if c.heading else ''}{c.text}"
            for c in self._children
        ]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out = self.embed(batch)
            if len(out) != len(batch):
                raise ValueError(f"embed() returned {len(out)} vectors for {len(batch)} texts")
            vectors.extend(_normalise_vector(v) for v in out)
        self._vectors = vectors

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """Return the ``k`` passages most similar to ``query`` (cosine similarity)."""
        if k <= 0 or not self._children:
            return []
        q = _normalise_vector(self.embed([self.query_prefix + query])[0])
        scored = [
            RetrievalHit(child=c, score=sum(a * b for a, b in zip(q, v, strict=True)))
            for c, v in zip(self._children, self._vectors, strict=True)
        ]
        scored.sort(key=lambda h: (-h.score, h.child.id))
        return scored[:k]


class HybridRetriever:
    """Combine several retrievers with reciprocal rank fusion (RRF).

    Keyword search is precise on exact terms (clause numbers, names, defined terms);
    semantic search catches paraphrases. Fusing their rankings is more robust than
    either alone and needs no score calibration.

    Args:
        retrievers: Two or more retrievers, for example ``BM25Retriever()`` and an
            :class:`EmbeddingRetriever`.
        weights: Optional weight per retriever (default 1.0 each).
        rrf_k: RRF damping constant; 60 is the value from the original RRF paper.
        candidates: How many results to take from each retriever before fusing.
        keep_top: Each retriever's top ``keep_top`` hits are always included, ahead of
            the fused ranking. Plain RRF favours passages that are mediocre in every
            list over one that a single retriever ranks first; this keeps such strong
            single-retriever matches (an exact paraphrase, an exact clause number).
    """

    def __init__(
        self,
        *retrievers: Retriever,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        candidates: int = 50,
        keep_top: int = 2,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever needs at least two retrievers.")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("Give one weight per retriever.")
        self.retrievers = retrievers
        self.weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.keep_top = max(0, keep_top)

    def index(self, children: Sequence[ChildChunk]) -> None:
        """Index ``children`` in every retriever."""
        for retriever in self.retrievers:
            retriever.index(children)

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """Fuse each retriever's ranking; return the top ``k``."""
        if k <= 0:
            return []
        scores: dict[str, float] = {}
        children: dict[str, ChildChunk] = {}
        per_retriever: list[list[str]] = []
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            hits = retriever.search(query, self.candidates)
            per_retriever.append([h.child.id for h in hits])
            for rank, hit in enumerate(hits, start=1):
                children[hit.child.id] = hit.child
                scores[hit.child.id] = scores.get(hit.child.id, 0.0) + weight / (self.rrf_k + rank)

        ranked: list[str] = []
        # Round-robin over each retriever's best hits first...
        for position in range(self.keep_top):
            for ids in per_retriever:
                if position < len(ids) and ids[position] not in ranked:
                    ranked.append(ids[position])
        # ...then everything else by fused score.
        ranked += [c for c in sorted(scores, key=lambda c: (-scores[c], c)) if c not in ranked]
        return [RetrievalHit(child=children[c], score=scores[c]) for c in ranked[:k]]
