"""The built-in PDF reader. Most tests hand-build tiny PDFs, so no PDF library is needed."""

from __future__ import annotations

# ruff: noqa: RUF001  (these tests use real typographic and Greek characters on purpose)
import base64
import difflib
import re
import zlib
from pathlib import Path

import pytest

from longdoc import DocumentLoadError, EmptyDocumentError, UnsupportedFormatError, load_bytes
from longdoc.pdf import extract_pages


def stream(data: bytes, extra: str = "", *, compress: bool = False) -> bytes:
    if compress:
        data = zlib.compress(data)
        extra += " /Filter /FlateDecode"
    return f"<< /Length {len(data)} {extra} >>\nstream\n".encode() + data + b"\nendstream"


def build(objects: dict[int, bytes | str], header: bytes = b"%PDF-1.5\n") -> bytes:
    """Concatenate objects into a PDF. No xref table: the reader finds objects by scanning."""
    out = header
    for num, body in objects.items():
        raw = body.encode() if isinstance(body, str) else body
        out += f"{num} 0 obj\n".encode() + raw + b"\nendobj\n"
    return out + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"


def simple_pdf(
    content: str,
    font: str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    extra_objects: dict[int, bytes | str] | None = None,
    compress: bool = False,
) -> bytes:
    objects: dict[int, bytes | str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [3 0 R] /Count 1 /Resources << /Font << /F1 4 0 R >> >> >>",
        3: "<< /Type /Page /Parent 2 0 R /Contents 5 0 R >>",
        4: font,
        5: stream(content.encode("latin-1"), compress=compress),
    }
    objects.update(extra_objects or {})
    return build(objects)


def text_of(pdf: bytes) -> str:
    return "\n".join(extract_pages(pdf))


def test_lines_words_and_escapes() -> None:
    pdf = simple_pdf(
        "BT /F1 12 Tf 72 700 Td (Hello \\(World\\) \\101\\102) Tj 0 -14 Td (Second line) Tj ET"
    )
    assert text_of(pdf) == "Hello (World) AB\nSecond line"


def test_flate_compressed_content() -> None:
    pdf = simple_pdf("BT /F1 12 Tf 72 700 Td (Compressed text) Tj ET", compress=True)
    assert text_of(pdf) == "Compressed text"


def test_ascii85_and_hex_filters() -> None:
    body = b"BT /F1 12 Tf 72 700 Td (From ASCII85) Tj ET"
    a85 = base64.a85encode(body) + b"~>"
    pdf = simple_pdf("", extra_objects={5: stream(a85, "/Filter /ASCII85Decode")})
    assert text_of(pdf) == "From ASCII85"
    hexed = body.hex().encode() + b">"
    pdf = simple_pdf("", extra_objects={5: stream(hexed, "/Filter /ASCIIHexDecode")})
    assert text_of(pdf) == "From ASCII85"
    both = base64.a85encode(zlib.compress(body)) + b"~>"
    pdf = simple_pdf("", extra_objects={5: stream(both, "/Filter [/ASCII85Decode /FlateDecode]")})
    assert text_of(pdf) == "From ASCII85"


def test_tj_arrays_and_word_gaps() -> None:
    pdf = simple_pdf("BT /F1 12 Tf 72 700 Td [(Hel) 20 (lo) -400 (World)] TJ ET")
    assert text_of(pdf) == "Hello World"  # a big negative kern is a space; a small one is not


def test_same_line_runs_are_separated_but_hyphens_and_punctuation_are_not() -> None:
    content = (
        "BT /F1 12 Tf 72 700 Td (non) Tj 20 0 Td (-) Tj 5 0 Td (renewal) Tj "
        "40 0 Td (and) Tj 30 0 Td (more) Tj 20 0 Td (.) Tj ET"
    )
    assert text_of(simple_pdf(content)) == "non-renewal and more."


def test_moving_down_starts_a_new_line_and_t_star_uses_leading() -> None:
    content = "BT /F1 10 Tf 14 TL 72 700 Td (one) Tj T* (two) Tj (three) ' ET"
    assert text_of(simple_pdf(content)) == "one\ntwo\nthree"


def test_tm_positions_lines() -> None:
    content = "BT /F1 10 Tf 1 0 0 1 72 700 Tm (top) Tj 1 0 0 1 72 680 Tm (bottom) Tj ET"
    assert text_of(simple_pdf(content)) == "top\nbottom"


def test_to_unicode_cmap_with_chars_and_ranges() -> None:
    cmap = (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        b"2 beginbfchar <0001> <0048> <0002> <0069> endbfchar\n"
        b"1 beginbfrange <0003> <0005> <0041> endbfrange\n"
        b"1 beginbfrange <0010> <0011> [<00E9> <20AC>] endbfrange\n"
        b"endcmap end end"
    )
    pdf = simple_pdf(
        "BT /F1 12 Tf 72 700 Td <00010002000300040005> Tj <00100011> Tj ET",
        font=(
            "<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 6 0 R >>"
        ),
        extra_objects={6: stream(cmap)},
    )
    assert text_of(pdf) == "HiABCé€".replace("Cé", "Cé")


def test_differences_encoding_and_glyph_names() -> None:
    font = (
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /Type /Encoding "
        "/Differences [65 /bullet /approxequal /eacute /uni20AC /Alpha] >> >>"
    )
    pdf = simple_pdf("BT /F1 12 Tf 72 700 Td (ABCDE) Tj ET", font=font)
    assert text_of(pdf) == "•≈é€Α"


def test_winansi_and_symbol_fonts() -> None:
    assert (
        text_of(
            simple_pdf(
                "BT /F1 12 Tf 72 700 Td (caf\\351 \\223q\\224) Tj ET",
                font="<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                "/Encoding /WinAnsiEncoding >>",
            )
        )
        == "café “q”"
    )
    symbol = "<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>"
    assert text_of(simple_pdf("BT /F1 12 Tf 72 700 Td (\\273 \\267 a) Tj ET", font=symbol)) == (
        "≈ • α"
    )


def test_object_streams_hold_the_page_tree() -> None:
    inner = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 /Resources << /Font << /F1 4 0 R >> >> >>"),
        (3, "<< /Type /Page /Parent 2 0 R /Contents 5 0 R >>"),
        (4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    header, body, offset = "", "", 0
    for num, text in inner:
        header += f"{num} {offset} "
        body += text + "\n"
        offset += len(text) + 1
    pdf = build(
        {
            6: stream(
                (header + body).encode(),
                f"/Type /ObjStm /N {len(inner)} /First {len(header)}",
                compress=True,
            ),
            5: stream(b"BT /F1 12 Tf 72 700 Td (Inside an object stream) Tj ET"),
        },
        header=b"%PDF-1.7\n",
    )
    assert text_of(pdf) == "Inside an object stream"


def test_page_tree_with_nesting_inheritance_and_multiple_content_streams() -> None:
    pdf = build(
        {
            1: "<< /Type /Catalog /Pages 2 0 R >>",
            2: (
                "<< /Type /Pages /Kids [6 0 R 3 0 R] /Count 3 "
                "/Resources << /Font << /F1 4 0 R >> >> >>"
            ),
            6: "<< /Type /Pages /Parent 2 0 R /Kids [7 0 R 8 0 R] /Count 2 >>",
            7: "<< /Type /Page /Parent 6 0 R /Contents 9 0 R >>",
            8: "<< /Type /Page /Parent 6 0 R /Contents [10 0 R 11 0 R] >>",
            3: "<< /Type /Page /Parent 2 0 R /Contents 12 0 R >>",
            4: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            9: stream(b"BT /F1 12 Tf 72 700 Td (page one) Tj ET"),
            10: stream(b"BT /F1 12 Tf 72 700 Td (page two,) Tj"),
            11: stream(b"0 -14 Td (second stream) Tj ET"),
            12: stream(b"BT /F1 12 Tf 72 700 Td (page three) Tj ET"),
        }
    )
    assert extract_pages(pdf) == ["page one", "page two,\nsecond stream", "page three"]


def test_cyclic_page_tree_terminates() -> None:
    pdf = build(
        {
            1: "<< /Type /Catalog /Pages 2 0 R >>",
            2: "<< /Type /Pages /Kids [2 0 R 3 0 R] /Count 1 >>",
            3: "<< /Type /Page /Parent 2 0 R /Contents 5 0 R /Resources << >> >>",
            5: stream(b"BT (still works) Tj ET"),
        }
    )
    assert extract_pages(pdf) == ["still works"]


def test_inline_images_do_not_corrupt_the_text() -> None:
    content = (
        b"BT /F1 12 Tf 72 700 Td (before) Tj ET q BI /W 2 /H 2 ID \x00(\x01) Tj\xff EI Q "
        b"BT 72 650 Td (after) Tj ET"
    )
    pdf = simple_pdf("", extra_objects={5: stream(content)})
    assert text_of(pdf) == "before\nafter"


def test_unknown_filters_and_missing_pieces_do_not_crash() -> None:
    pdf = simple_pdf("", extra_objects={5: stream(b"xxx", "/Filter /LZWDecode")})
    assert extract_pages(pdf) == [""]
    no_font = simple_pdf("BT /Nope 12 Tf 72 700 Td (plain) Tj ET")
    assert text_of(no_font) == "plain"


def test_errors() -> None:
    with pytest.raises(DocumentLoadError, match="Not a PDF"):
        extract_pages(b"<html>not a pdf</html>")
    encrypted = simple_pdf("BT (x) Tj ET").replace(b"trailer\n<<", b"trailer\n<< /Encrypt 9 0 R")
    with pytest.raises(UnsupportedFormatError, match="encrypted"):
        extract_pages(encrypted)
    assert extract_pages(b"%PDF-1.4\nthis is junk with no objects at all\n") == []


def test_loader_uses_the_builtin_reader_without_pypdf() -> None:
    pdf = simple_pdf("BT /F1 12 Tf 72 700 Td (Loaded without any library) Tj ET")
    doc = load_bytes(pdf, filename="x.pdf", pdf_backend="builtin")
    assert "Loaded without any library" in doc.text
    assert [p.number for p in doc.pages] == [1]
    with pytest.raises(EmptyDocumentError):
        load_bytes(b"%PDF-1.4\njunk\n", filename="junk.pdf", pdf_backend="builtin")
    with pytest.raises(UnsupportedFormatError, match="pdf_backend"):
        load_bytes(pdf, filename="x.pdf", pdf_backend="nope")  # type: ignore[arg-type]


# -- agreement with a reference reader on real producers ------------------------------------


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def agreement(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def test_agrees_with_pypdf_on_fpdf2_output(tmp_path: Path) -> None:
    fpdf = pytest.importorskip("fpdf")
    pypdf = pytest.importorskip("pypdf")
    pdf = fpdf.FPDF()
    pdf.set_font("Helvetica", size=11)
    for n in range(1, 4):
        pdf.add_page()
        pdf.multi_cell(0, 6, f"Page {n}. " + "The parties shall cooperate in good faith. " * 40)
    path = tmp_path / "f.pdf"
    pdf.output(str(path))
    mine = extract_pages(path.read_bytes())
    ref = [(p.extract_text() or "") for p in pypdf.PdfReader(str(path)).pages]
    assert len(mine) == len(ref) == 3
    for a, b in zip(mine, ref, strict=True):
        assert agreement(a, b) > 0.99


def test_agrees_with_pypdf_on_reportlab_output(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "r.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)  # ASCII85 + Flate compressed
    for page in range(2):
        t = c.beginText(72, 720)
        t.setFont("Helvetica", 11)
        for line in range(20):
            t.textLine(f"Page {page + 1} line {line + 1}: the Agreement renews for 12 months.")
        c.drawText(t)
        c.showPage()
    c.save()
    mine = extract_pages(path.read_bytes())
    ref = [(p.extract_text() or "") for p in pypdf.PdfReader(str(path)).pages]
    assert [agreement(a, b) for a, b in zip(mine, ref, strict=True)] == [1.0, 1.0]


def test_unicode_truetype_font_through_tounicode(tmp_path: Path) -> None:
    fpdf = pytest.importorskip("fpdf")
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("DejaVu font not installed")
    pdf = fpdf.FPDF()
    pdf.add_font("DejaVu", fname=str(font))
    pdf.set_font("DejaVu", size=11)
    pdf.add_page()
    pdf.multi_cell(0, 6, "Café – naïve “quoted” ≈ €440,000 • résumé")
    out = extract_pages(bytes(pdf.output()))
    assert "Café – naïve “quoted” ≈ €440,000 • résumé" in out[0]


def test_each_pdf_reader_covers_what_the_other_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """In auto mode a failure of one reader falls through to the other."""
    from longdoc import loaders

    pdf = simple_pdf("BT /F1 12 Tf 72 700 Td (Found by the fallback reader) Tj ET")

    def broken(data: bytes, filename: str) -> list[str]:
        raise DocumentLoadError("pypdf failed")

    monkeypatch.setattr(loaders, "_has_pypdf", lambda: True)
    monkeypatch.setattr(loaders, "_pages_with_pypdf", broken)
    assert "Found by the fallback reader" in load_bytes(pdf, filename="x.pdf").text

    # ...and the other way round: pypdf finds text when the built-in reader finds none
    monkeypatch.setattr(loaders, "_pages_with_pypdf", lambda d, f: ["text from pypdf"])
    monkeypatch.setattr(loaders, "_pages_with_builtin", lambda d, f: [""])
    assert load_bytes(pdf, filename="x.pdf", pdf_backend="auto").text == "text from pypdf"

    # an explicit choice never falls back
    monkeypatch.setattr(loaders, "_pages_with_pypdf", broken)
    with pytest.raises(DocumentLoadError, match="pypdf failed"):
        load_bytes(pdf, filename="x.pdf", pdf_backend="pypdf")

    # when every reader fails, the first error is reported
    monkeypatch.setattr(loaders, "_pages_with_builtin", broken)
    with pytest.raises(DocumentLoadError, match="pypdf failed"):
        load_bytes(pdf, filename="x.pdf")
