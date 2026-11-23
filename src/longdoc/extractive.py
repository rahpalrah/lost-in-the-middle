"""A built-in answer engine that needs no model, no API key and no network.

:class:`ExtractiveClient` answers a question by **quoting** the most relevant sentences from the
sections it is given, each followed by its section citation. Nothing is generated, so:

* every quote is verbatim, and every citation points at the section the quote came from;
* it cannot invent a figure, a clause or a name;
* when the key terms of the question do not appear in the document it says so instead of
  guessing.

The trade-off is that it cannot reason, paraphrase, combine facts or do arithmetic: it returns
the passages and leaves the reading to you. Use it as the zero-setup default, as a fast first
pass, or as a baseline against which to measure a language model.

It plugs in like any other client, so ``DocumentQA(doc)`` works out of the box::

    from longdoc import DocumentQA
    answer = DocumentQA.from_file("contract.pdf").ask("What is the notice period?")
"""

from __future__ import annotations

import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .aio import AsyncAdapter
from .local import LocalMessage, LocalStopDetails, LocalTextBlock, LocalUsage, _Stream
from .retrieval import stem, tokenize

MODEL_NAME = "longdoc-extractive-1"
NOT_FOUND_TEXT = "The provided sections do not contain information that answers this question."

_SECTION = re.compile(
    r'<section id="(?P<id>p\d{4})"(?: heading="(?P<heading>[^"]*)")?'
    r'(?: pages="(?P<pages>[^"]*)")?>\n(?P<body>.*?)\n</section>',
    re.DOTALL,
)
_QUESTION = re.compile(r"<question>\n(?P<q>.*?)\n</question>", re.DOTALL)
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"(\[]?[A-Z0-9])")
_ABBREVIATIONS = (
    "inc.", "ltd.", "co.", "corp.", "no.", "nos.", "sec.", "art.", "vs.", "e.g.", "i.e.",
    "etc.", "u.s.", "dr.", "mr.", "ms.", "st.", "approx.",
)  # fmt: skip
_CLAUSE_LABEL = re.compile(r"^(?:[A-Z]\.)?\d+(?:\.\d+)*\.?$")
_CONTINUATION = re.compile(
    r"^(provided|however|except|unless|subject to|notwithstanding|in the event|if |where |"
    r"otherwise|but )",
    re.IGNORECASE,
)
_BACKWARD = re.compile(r"^(such|this|these|the foregoing|that|it |they |the same)", re.IGNORECASE)
_NUMBER_INTENT = re.compile(
    r"\b(how (?:much|many|long|often|soon|far)|when|what (?:percent|percentage|amount|fee|"
    r"cap|limit|notice|period|deadline|date|number)|within|days?|months?|years?|hours?)\b",
    re.IGNORECASE,
)
_HAS_NUMBER = re.compile(r"\d")
_MARKER_ID = re.compile(r"(?<![\w-])p(\d{4}(?:-c\d{3})?)(?![\w-])")
_MARKER_TAG = re.compile(r"<(/?(?:section|sections|document|question)\b)", re.IGNORECASE)
_MAX_QUOTE_CHARS = 900

# Everyday question words that say nothing about the topic. A contract never contains "tell" or
# "far", so they must not count as evidence that the document lacks the answer.
_GENERIC_STEMS = frozenset(
    stem(w)
    for w in (  # noqa: SIM905
        "tell far advance decide decides deciding give gives giving given need needs want wants "
        "get gets make makes know much many long start starts happen happens provide provides "
        "explain describe list say state mean means apply applies regarding about per same "
        "also please can could should would may might whether kind type sort way thing things "
        "case example info information detail details exactly specifically general overall "
        "us our we you your i me my they their it its must them him her his he she been had having "
        "done go going then than so some other each every both all only just very more most not no "
        "yes up out over under again once own"
    ).split()
)


@dataclass(frozen=True, slots=True)
class _Sentence:
    section_id: str
    heading: str
    text: str
    order: int
    tokens: tuple[str, ...]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping clause numbers and common abbreviations intact."""
    out: list[str] = []
    # Paragraphs are separated by blank lines. Single line breaks inside a paragraph are just
    # PDF line wrapping, so they must not end a sentence.
    for paragraph in re.split(r"\n\s*\n", text):
        line = " ".join(paragraph.split())
        if not line:
            continue
        parts: list[str] = []
        for piece in _BOUNDARY.split(line):
            piece = piece.strip()
            if not piece:
                continue
            if parts and (
                parts[-1].lower().endswith(_ABBREVIATIONS) or _CLAUSE_LABEL.match(parts[-1])
            ):
                parts[-1] = f"{parts[-1]} {piece}"
            else:
                parts.append(piece)
        out.extend(parts)
    return out


def parse_prompt(prompt: str) -> tuple[str, list[tuple[str, str, str, str]]]:
    """Extract ``(question, [(section_id, heading, pages, text), ...])`` from a longdoc prompt."""
    q = _QUESTION.search(prompt)
    question = html.unescape(q.group("q")).strip() if q else ""
    sections = [
        (
            m.group("id"),
            html.unescape(m.group("heading") or ""),
            html.unescape(m.group("pages") or ""),
            html.unescape(m.group("body")),
        )
        for m in _SECTION.finditer(prompt)
    ]
    return question, sections


def _flatten(system: Any, messages: list[dict[str, Any]]) -> str:
    def text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in content
        )

    parts = [text_of(system)] if system else []
    parts += [text_of(m["content"]) for m in messages]
    return "\n".join(parts)


class ExtractiveClient:
    """Answer by quoting the most relevant sentences. See the module docstring.

    Args:
        max_sentences: Most sentences to quote.
        max_unseen_share: Decline when at least this share (0 to 1) of the question's topical
            words (everyday words like "tell" or "far" are ignored) never occurs in the sections:
            a question about "carbon emissions" is declined by a contract that never mentions them.
        relative_cutoff: Quote further sentences only if they score at least this share of the
            best one.
    """

    model_name = MODEL_NAME

    def __init__(
        self,
        *,
        max_sentences: int = 3,
        max_unseen_share: float = 0.5,
        relative_cutoff: float = 0.45,
    ) -> None:
        self.max_sentences = max_sentences
        self.max_unseen_share = max_unseen_share
        self.relative_cutoff = relative_cutoff
        self.beta = SimpleNamespace(messages=_Messages(self))
        self.messages = self.beta.messages

    def __repr__(self) -> str:
        return f"ExtractiveClient(max_sentences={self.max_sentences})"

    # -- the engine ---------------------------------------------------------------------

    def answer(self, question: str, sections: list[tuple[str, str, str, str]]) -> str:
        """Return the answer text for ``question`` given ``(id, heading, pages, text)`` sections."""
        q_terms = list(dict.fromkeys(tokenize(question)))
        sentences = self._sentences(sections)
        if not q_terms or not sentences:
            return NOT_FOUND_TEXT

        n = len(sentences)
        df: Counter[str] = Counter()
        for s in sentences:
            df.update(set(s.tokens))
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
        topical = [t for t in q_terms if t not in _GENERIC_STEMS] or q_terms
        unseen = [t for t in topical if t not in df]
        if len(unseen) == len(topical) or len(unseen) / len(topical) >= self.max_unseen_share:
            return NOT_FOUND_TEXT

        numeric_question = bool(_NUMBER_INTENT.search(question))
        avg_len = sum(len(s.tokens) for s in sentences) / n
        scored: list[tuple[float, int]] = []  # (score, index)
        for i, s in enumerate(sentences):
            tf = Counter(s.tokens)
            norm = 1.5 * (0.25 + 0.75 * len(s.tokens) / (avg_len or 1.0))
            score = sum(idf[t] * tf[t] * 2.5 / (tf[t] + norm) for t in q_terms if t in tf)
            heading_terms = set(tokenize(s.heading))
            score += 0.5 * sum(idf.get(t, 0.0) for t in q_terms if t in heading_terms)
            if numeric_question and _HAS_NUMBER.search(s.text):
                score *= 1.15
            scored.append((score, i))

        best_score, _ = max(scored)
        if best_score <= 0:
            return NOT_FOUND_TEXT

        ranked = sorted(scored, reverse=True)
        chosen: list[int] = []
        for score, i in ranked:
            if len(chosen) >= self.max_sentences:
                break
            if not chosen or score >= self.relative_cutoff * best_score:
                chosen.append(i)
        chosen = self._with_neighbours(chosen, sentences)

        lines = [
            f'- "{self._clip(sentences[i].text)}" [{sentences[i].section_id}]'
            for i in sorted(chosen)
        ]
        return "The most relevant passages are:\n" + "\n".join(lines)

    def _sentences(self, sections: list[tuple[str, str, str, str]]) -> list[_Sentence]:
        out: list[_Sentence] = []
        seen: set[str] = set()
        for sid, heading, _pages, text in sections:
            for sent in split_sentences(text):
                words = sent.split()
                if len(words) < 4 or sent.strip() == heading.strip():
                    continue  # headings and fragments are not answers
                if len(words) < 15 and not sent.rstrip().endswith(
                    (".", "!", "?", ";", ":", '"', ")")
                ):
                    continue  # a short line with no closing punctuation is a heading
                if sent in seen:
                    continue  # the same boilerplate sentence recurs in many sections
                seen.add(sent)
                out.append(_Sentence(sid, heading, sent, len(out), tuple(tokenize(sent))))
        return out

    def _with_neighbours(self, chosen: list[int], sentences: list[_Sentence]) -> list[int]:
        """Keep a clause's conditions with it: 'provided that...' and 'Such notice...'."""
        result = list(chosen)
        for i in list(chosen):
            nxt = i + 1
            if (
                nxt < len(sentences)
                and sentences[nxt].section_id == sentences[i].section_id
                and _CONTINUATION.match(sentences[nxt].text)
                and nxt not in result
            ):
                result.append(nxt)
            prv = i - 1
            if (
                prv >= 0
                and sentences[prv].section_id == sentences[i].section_id
                and _BACKWARD.match(sentences[i].text)
                and prv not in result
            ):
                result.append(prv)
        return result

    @staticmethod
    def _clip(text: str) -> str:
        text = text.replace('"', "'")  # keep the quoting unambiguous
        # Text inside a quote must never read as one of OUR markers: a document containing
        # "[p0003]" or "<section id=...>" would otherwise look like a citation or a forged
        # section. An invisible zero-width space inside the ID (p0001 -> p<ZWSP>0001) breaks
        # the pattern; the visible text is unchanged.
        text = _MARKER_ID.sub("p\u200b\\1", text)
        text = _MARKER_TAG.sub("<\u200b\\1", text)
        if len(text) <= _MAX_QUOTE_CHARS:
            return text
        cut = text.rfind(" ", 0, _MAX_QUOTE_CHARS)
        return text[: cut if cut > 0 else _MAX_QUOTE_CHARS] + " ..."

    # -- client interface ---------------------------------------------------------------

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 0,
        system: Any = None,
        model: str | None = None,
        **_ignored: Any,
    ) -> LocalMessage:
        """Answer from the sections and question found in the prompt."""
        prompt = _flatten(system, messages)
        if max_tokens == 0:  # cache warm-up request: nothing to do
            return LocalMessage([], MODEL_NAME, "max_tokens", LocalUsage(0, 0))
        question, sections = parse_prompt(prompt)
        text = self.answer(question, sections)
        details = None
        stop_reason = "end_turn"
        if not question:
            stop_reason, details = (
                "refusal",
                LocalStopDetails("invalid_prompt", "No question found."),
            )
        return LocalMessage(
            content=[LocalTextBlock(text)],
            model=MODEL_NAME,
            stop_reason=stop_reason,
            usage=LocalUsage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4),
            stop_details=details,
        )

    def to_async(self) -> AsyncAdapter:
        """An async view of this client."""
        return AsyncAdapter(self)


class _Messages:
    def __init__(self, owner: ExtractiveClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> LocalMessage:
        """Answer one request."""
        return self._owner.complete(**kwargs)

    def stream(self, **kwargs: Any) -> _Stream:
        """Same as :meth:`create`, wrapped like the SDK's streaming helper."""
        return _Stream(self._owner.complete(**kwargs))
