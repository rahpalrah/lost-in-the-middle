from __future__ import annotations

import pytest

from longdoc import ChunkingConfig, ConfigurationError, Document, ParentChildChunker
from longdoc.chunking import is_heading
from longdoc.exceptions import EmptyDocumentError


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("# Introduction", True),
        ("19. Termination", True),
        ("3.2 Limitation of Liability", True),
        ("Section 9 Confidentiality", True),
        ("SCHEDULE C", True),
        ("This is an ordinary sentence that ends with a period.", False),
        ("", False),
        ("x" * 200, False),
    ],
)
def test_is_heading(line: str, expected: bool) -> None:
    assert is_heading(line) is expected


def test_headings_start_new_parents(contract: Document) -> None:
    chunked = ParentChildChunker().chunk(contract)
    headings = [p.heading for p in chunked.parents if p.heading]
    assert "# 7. Payment" in headings
    assert "# 19. Termination" in headings


def test_parent_pages_are_tracked(contract: Document) -> None:
    chunked = ParentChildChunker().chunk(contract)
    termination = next(p for p in chunked.parents if p.heading == "# 19. Termination")
    assert termination.page_start == 54
    assert all(p.page_start is not None for p in chunked.parents)
    assert chunked.parents[-1].page_end == 60


def test_size_limits_are_respected(contract: Document) -> None:
    cfg = ChunkingConfig(parent_max_chars=3000, child_max_chars=500, child_overlap_chars=100)
    chunked = ParentChildChunker(cfg).chunk(contract)
    assert all(len(p.text) <= cfg.parent_max_chars + 2 for p in chunked.parents)
    assert all(len(c.text) <= cfg.child_max_chars for c in chunked.children)


def test_every_child_belongs_to_a_parent_and_is_inside_it(contract: Document) -> None:
    chunked = ParentChildChunker().chunk(contract)
    for child in chunked.children:
        parent = chunked.parent(child.parent_id)
        assert child.text.split(" ")[0] in parent.text


def test_children_overlap() -> None:
    sentences = " ".join(f"Sentence number {i} is here." for i in range(60))
    cfg = ChunkingConfig(parent_max_chars=10_000, child_max_chars=200, child_overlap_chars=60)
    children = ParentChildChunker(cfg).chunk(Document.from_text(sentences)).children
    assert len(children) > 2
    first_last = children[0].text.split(". ")[-1]
    assert first_last in children[1].text


def test_chunking_is_deterministic(contract: Document) -> None:
    a = ParentChildChunker().chunk(contract)
    b = ParentChildChunker().chunk(contract)
    assert a.parents == b.parents
    assert a.children == b.children


def test_long_unbroken_text_is_split() -> None:
    cfg = ChunkingConfig(parent_max_chars=1000, child_max_chars=100, child_overlap_chars=10)
    chunked = ParentChildChunker(cfg).chunk(Document.from_text("word " * 2000))
    assert len(chunked.parents) > 5
    assert all(len(c.text) <= 100 for c in chunked.children)


def test_empty_document_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        ParentChildChunker().chunk(Document.from_text("  \n\n  "))


def test_invalid_config() -> None:
    with pytest.raises(ConfigurationError):
        ChunkingConfig(parent_max_chars=100, child_max_chars=200)
    with pytest.raises(ConfigurationError):
        ChunkingConfig(child_max_chars=100, child_overlap_chars=100)


# -- regressions found by the contract-review benchmark (PDF-extracted text) ----------


def test_headings_detected_inside_pdf_style_paragraphs() -> None:
    # pypdf output: no blank lines between headings, wrapped body lines, sub-headings.
    page = (
        "ARTICLE 19. TERM AND TERMINATION\n"
        "19.1 Termination for Cause\n"
        "Either party may terminate this Agreement if the other party materially\n"
        "breaches it and fails to cure within thirty (30) days.\n"
        "19.2 Termination for Convenience\n"
        "Customer may terminate on ninety (90) days notice, subject to\n"
        "Article 12 and the fee in Schedule C.\n"
        "A.3 Personal Data Breach\n"
        "The Processor shall notify the Controller within 48 hours."
    )
    chunked = ParentChildChunker().chunk(Document.from_pages([page]))
    assert [p.heading for p in chunked.parents] == [
        "ARTICLE 19. TERM AND TERMINATION > 19.1 Termination for Cause",
        "19.2 Termination for Convenience",
        "A.3 Personal Data Breach",
    ]
    assert "Article 12 and the fee" in chunked.parents[1].text  # wrapped line, not a heading


@pytest.mark.parametrize(
    "line",
    [
        "12 months of the Initial Term by giving Vendor not less than",
        "Article 21. Such obligations apply equally to any Affiliate performing",
        "30 days of the invoice date.",
    ],
)
def test_wrapped_body_lines_are_not_headings(line: str) -> None:
    assert is_heading(line) is False
