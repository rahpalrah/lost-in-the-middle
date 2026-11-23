from __future__ import annotations

import pytest

from longdoc import BM25Retriever, Document, ParentChildChunker, Retriever
from longdoc.retrieval import select_parents, tokenize


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("What is the Termination fee?") == ["termination", "fee"]


def test_bm25_finds_the_buried_clause(contract: Document) -> None:
    chunked = ParentChildChunker().chunk(contract)
    retriever = BM25Retriever()
    retriever.index(chunked.children)
    hits = retriever.search("termination notice early fee", k=5)
    assert hits
    best_parent = chunked.parent(hits[0].child.parent_id)
    assert best_parent.page_start == 54
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_bm25_no_match_returns_empty(contract: Document) -> None:
    retriever = BM25Retriever()
    retriever.index(ParentChildChunker().chunk(contract).children)
    assert retriever.search("zebra quantum", k=5) == []
    assert retriever.search("termination", k=0) == []


def test_bm25_satisfies_protocol() -> None:
    assert isinstance(BM25Retriever(), Retriever)


def test_select_parents_dedupes_limits_and_orders(contract: Document) -> None:
    chunked = ParentChildChunker().chunk(contract)
    retriever = BM25Retriever()
    retriever.index(chunked.children)
    hits = retriever.search("parties cooperate termination payment", k=50)

    chosen = select_parents(hits, chunked, max_parents=3, max_chars=1_000_000)
    assert len(chosen) == 3
    assert len({p.id for p in chosen}) == 3
    assert [p.order for p in chosen] == sorted(p.order for p in chosen)

    tight = select_parents(hits, chunked, max_parents=10, max_chars=1)
    assert len(tight) == 1  # the best parent is always kept


@pytest.mark.parametrize(
    "words",
    [
        ["governs", "governed", "governing", "govern"],
        ["increase", "increases", "increased"],
        ["renewal", "renew", "renews"],
        ["liabilities", "liability"],
    ],
)
def test_stem_groups_word_forms(words: list[str]) -> None:
    from longdoc.retrieval import stem

    assert len({stem(w) for w in words}) == 1


def test_stem_leaves_short_words_and_numbers() -> None:
    from longdoc.retrieval import stem

    assert [stem(w) for w in ["fee", "law", "2026", "gas"]] == ["fee", "law", "2026", "gas"]
    assert [stem(w) for w in ["laws", "fees", "uses", "class", "status"]] == [
        "law",
        "fee",
        "use",
        "class",
        "status",
    ]


def test_headings_are_searchable() -> None:
    from longdoc import Document, ParentChildChunker

    doc = Document.from_text(
        "21.2 Governing Law\n\nThis Agreement is subject to the courts of New York County.\n\n"
        "5.1 Invoicing\n\nInvoices are payable in thirty days."
    )
    chunked = ParentChildChunker().chunk(doc)
    retriever = BM25Retriever()
    retriever.index(chunked.children)
    hit = retriever.search("which law governs the agreement", k=1)[0]
    assert hit.child.heading == "21.2 Governing Law"


# -- semantic and hybrid retrieval -----------------------------------------------------

from longdoc import ChildChunk, EmbeddingRetriever, HybridRetriever  # noqa: E402


def _toy_embed(texts: list[str]) -> list[list[float]]:
    """2-D 'semantic' space: axis 0 = ownership/acquisition, axis 1 = money."""
    vocab = {
        "acquired": (1, 0),
        "control": (1, 0),
        "assign": (0.8, 0),
        "paying": (0, 1),
        "fee": (0, 1),
        "price": (0, 1),
    }
    out = []
    for t in texts:
        x = y = 0.01
        for word, (a, b) in vocab.items():
            if word in t.lower():
                x, y = x + a, y + b
        out.append([x, y])
    return out


CHILDREN = [
    ChildChunk("p0000-c000", "p0000", "Vendor may assign on a change of control.", 0, "Assignment"),
    ChildChunk("p0001-c000", "p0001", "The annual fee is 480,000.", 0, "Fees"),
    ChildChunk("p0002-c000", "p0002", "Notices must be in writing to the vendor.", 0, "Notices"),
]


def test_embedding_retriever_finds_paraphrase() -> None:
    r = EmbeddingRetriever(_toy_embed, query_prefix="q: ", batch_size=2)
    r.index(CHILDREN)
    assert r.search("If the vendor is acquired?", 1)[0].child.parent_id == "p0000"
    assert r.search("What are we paying?", 1)[0].child.parent_id == "p0001"
    assert r.search("anything", 0) == []


def test_embedding_retriever_rejects_bad_embedder() -> None:
    r = EmbeddingRetriever(lambda texts: [[1.0, 0.0]])
    with pytest.raises(ValueError, match="returned 1 vectors"):
        r.index(CHILDREN)


def test_hybrid_keeps_strong_single_retriever_hits() -> None:
    # BM25 matches "vendor" everywhere; only embeddings understand "acquired".
    hybrid = HybridRetriever(BM25Retriever(), EmbeddingRetriever(_toy_embed), keep_top=1)
    hybrid.index(CHILDREN)
    hits = hybrid.search("If the vendor is acquired, what happens?", 2)
    assert "p0000" in {h.child.parent_id for h in hits}
    assert len({h.child.id for h in hits}) == len(hits)  # no duplicates


def test_hybrid_validates_arguments() -> None:
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever(BM25Retriever())
    with pytest.raises(ValueError, match="one weight"):
        HybridRetriever(BM25Retriever(), BM25Retriever(), weights=[1.0])
    assert isinstance(HybridRetriever(BM25Retriever(), BM25Retriever()), Retriever)
