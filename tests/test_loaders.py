from __future__ import annotations

from pathlib import Path

import pytest

from longdoc import DocumentLoadError, UnsupportedFormatError, load_document
from longdoc.exceptions import EmptyDocumentError


def test_load_markdown(tmp_path: Path) -> None:
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\nBody text.", encoding="utf-8")
    doc = load_document(f)
    assert doc.source == str(f)
    assert "Body text." in doc.text
    assert doc.pages[0].number is None


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_document(tmp_path / "nope.txt")


def test_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "sheet.xlsx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFormatError):
        load_document(f)


def test_empty_text_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("   ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(f)


def test_corrupt_pdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"not a pdf")
    with pytest.raises(DocumentLoadError):
        load_document(f)


def test_blank_pdf_is_empty(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    f = tmp_path / "blank.pdf"
    with f.open("wb") as fh:
        writer.write(fh)
    with pytest.raises(EmptyDocumentError):
        load_document(f)


def test_strip_running_headers_and_page_numbers() -> None:
    from longdoc.loaders import strip_running_headers

    pages = [
        f"ACME MSA - CONFIDENTIAL\nClause {i}.1 applies.\nFee is {i * 10} dollars.\nPage {i} of 5"
        for i in range(1, 6)
    ]
    cleaned = strip_running_headers(pages)
    assert all("CONFIDENTIAL" not in p and "Page" not in p for p in cleaned)
    # Lines that differ only by a number are real content and must be kept.
    assert cleaned[2] == "Clause 3.1 applies.\nFee is 30 dollars."
    assert strip_running_headers(pages[:2]) == pages[:2]  # too few pages to judge


def test_pdf_round_trip_page_numbers(tmp_path: Path) -> None:
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.set_font("Helvetica", size=11)
    for n in range(1, 4):
        pdf.add_page()
        pdf.multi_cell(0, 6, f"Running header\n{n}. Section {n}\nClause text for page {n}.")
    f = tmp_path / "three.pdf"
    pdf.output(str(f))
    doc = load_document(f)
    assert [p.number for p in doc.pages] == [1, 2, 3]
    assert "Running header" not in doc.text
    assert "Clause text for page 2." in doc.pages[1].text
