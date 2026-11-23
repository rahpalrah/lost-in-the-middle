"""Prompt construction.

Everything here is deterministic: the same inputs always produce byte-identical
output. Prompt caching is a prefix match, so a timestamp, a random ID or an unstable
ordering anywhere in the prefix would silently disable it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from .models import ChunkedDocument, ParentChunk

SYSTEM_PROMPT = """\
You answer questions about a document using only the sections provided to you.

Each section is wrapped in a <section> tag with an id attribute such as "p0007".
- Base every statement on the sections. Do not use outside knowledge about the \
document's subject.
- After each claim, cite the section it came from in square brackets, e.g. [p0007]. \
Cite every section you rely on.
- Pay equal attention to every section, including ones in the middle of the document; \
conditions, exceptions and deadlines often appear far from the main clause.
- If the sections do not contain the answer, say so plainly and say what is missing. \
Do not guess.
- Quote exact figures, dates and defined terms as written."""

CITATION_PATTERN = re.compile(r"\[(p\d{4})(?:-c\d{3})?\]")
# Looser forms models also produce: bare "p0007" and echoed <section id="p0007"> tags.
_BARE_ID = re.compile(r"(?<![\w-])(p\d{4})(?:-c\d{3})?(?![\w-])")
_LEAKED_OPEN_TAG = re.compile(r"<section\b[^>]*\bid=[\"'](p\d{4})[\"'][^>]*>")
_LEAKED_TAG = re.compile(r"</?(?:section|sections|document|question)\b[^>]*>")


def clean_answer(text: str) -> str:
    """Remove prompt markup a model echoed into its answer.

    Small models often copy ``<section id="p0007" heading="...">`` into the answer
    instead of writing ``[p0007]``. The opening tag becomes a ``[p0007]`` citation and
    any other prompt tags are dropped.
    """
    text = _LEAKED_OPEN_TAG.sub(lambda m: f"[{m.group(1)}] ", text)
    text = _LEAKED_TAG.sub("", text)
    text = re.sub(r"\[(p\d{4})\](?:\s*\[\1\])+", r"[\1]", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def render_section(parent: ParentChunk) -> str:
    """Render one parent section as an XML-tagged block."""
    attrs = [f"id={quoteattr(parent.id)}"]
    if parent.heading:
        attrs.append(f"heading={quoteattr(parent.heading)}")
    if parent.page_label:
        attrs.append(f"pages={quoteattr(parent.page_label)}")
    return f"<section {' '.join(attrs)}>\n{escape(parent.text)}\n</section>"


def render_sections(parents: Iterable[ParentChunk]) -> str:
    """Render sections, in the order given, separated by blank lines."""
    return "\n\n".join(render_section(p) for p in parents)


def system_blocks(
    *,
    instructions: str,
    document: ChunkedDocument | None = None,
    cache_ttl: str = "5m",
) -> list[dict[str, Any]]:
    """Build the ``system`` content blocks.

    With ``document``, the full sectioned document is appended after the instructions
    and marked with ``cache_control`` so it becomes a reusable cached prefix.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
    if document is not None:
        title = f" source={quoteattr(document.source)}" if document.source else ""
        blocks.append(
            {
                "type": "text",
                "text": f"<document{title}>\n{render_sections(document.parents)}\n</document>",
                "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
            }
        )
    return blocks


def user_message(question: str, context: Sequence[ParentChunk] | None = None) -> dict[str, Any]:
    """Build the user turn: retrieved sections (if any) followed by the question."""
    parts: list[str] = []
    if context is not None:
        if context:
            parts.append(f"<sections>\n{render_sections(context)}\n</sections>")
        else:
            parts.append("<sections>\n(no matching sections were found)\n</sections>")
    parts.append(f"<question>\n{question.strip()}\n</question>")
    return {"role": "user", "content": "\n\n".join(parts)}


def extract_citations(text: str, document: ChunkedDocument) -> tuple[ParentChunk, ...]:
    """Return the sections cited in ``text``, in order of first citation.

    Accepts ``[p0007]``, ``[p0007-c002]`` and bare ``p0007``. IDs that don't exist in
    the document are ignored, so an invented ID never becomes a citation.
    """
    ids = dict.fromkeys(m.group(1) for m in _BARE_ID.finditer(text))
    return tuple(p for pid in ids if (p := document.get_parent(pid)) is not None)
