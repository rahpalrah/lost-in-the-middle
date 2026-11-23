from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from longdoc import (
    AsyncDocumentQA,
    ConfigurationError,
    Document,
    DocumentQA,
    QAConfig,
    RefusalError,
    Strategy,
    TokenUsage,
)
from longdoc.client import FALLBACK_BETA

from .conftest import FakeClient, fake_message


def make_qa(contract: Document, client: FakeClient, **config: Any) -> DocumentQA:
    return DocumentQA(contract, config=QAConfig(**config), client=client)  # type: ignore[arg-type]


def test_retrieval_request_shape(contract: Document) -> None:
    client = FakeClient()
    qa = make_qa(contract, client)
    answer = qa.ask("What is the termination notice period and fee?")

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert "thinking" not in call  # adaptive is the default on Opus 5.5
    assert len(call["system"]) == 1  # no document in the prefix
    content = call["messages"][0]["content"]
    assert "<question>" in content
    assert "90 days" in content  # the buried clause was retrieved

    assert any(p.page_start == 54 for p in answer.context)
    assert len(answer.context) <= qa.config.max_parents


def test_cached_document_request_shape(contract: Document) -> None:
    client = FakeClient()
    qa = make_qa(contract, client, strategy=Strategy.CACHED_DOCUMENT, cache_ttl="1h")
    answer = qa.ask("Summarise payment terms.")

    call = client.messages.calls[0]
    doc_block = call["system"][1]
    assert doc_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "<document" in doc_block["text"]
    assert "<sections>" not in call["messages"][0]["content"]
    assert answer.context == qa.chunks.parents


def test_cached_prefix_is_byte_identical_across_questions(contract: Document) -> None:
    client = FakeClient()
    qa = make_qa(contract, client, strategy=Strategy.CACHED_DOCUMENT)
    qa.ask("First question?")
    qa.ask("A completely different second question?")
    first, second = client.messages.calls
    assert first["system"] == second["system"]


def test_per_call_strategy_override(contract: Document) -> None:
    client = FakeClient()
    qa = make_qa(contract, client)
    qa.ask("Payment?", strategy=Strategy.CACHED_DOCUMENT)
    assert len(client.messages.calls[0]["system"]) == 2


def test_answer_parsing_and_citations(contract: Document) -> None:
    qa = make_qa(contract, FakeClient(fake_message(cache_read=900, cache_write=0)))
    termination_id = next(p.id for p in qa.chunks.parents if p.page_start == 54)
    qa.client.messages.responses.append(  # type: ignore[union-attr]
        fake_message(f"90 days [{termination_id}]; see also [{termination_id}] and [p9999].")
    )
    qa.ask("warm")
    answer = qa.ask("Notice period?")

    assert [c.id for c in answer.citations] == [termination_id]  # deduped, unknown dropped
    assert answer.request_id == "req_test"
    assert not answer.truncated
    assert str(answer) == answer.text


def test_usage_accumulates(contract: Document) -> None:
    client = FakeClient(fake_message(cache_write=5000), fake_message(cache_read=5000))
    qa = make_qa(contract, client, strategy=Strategy.CACHED_DOCUMENT)
    qa.ask("one?")
    qa.ask("two?")
    assert qa.usage == TokenUsage(
        input_tokens=200,
        output_tokens=40,
        cache_creation_input_tokens=5000,
        cache_read_input_tokens=5000,
    )
    assert 0 < qa.usage.cache_hit_ratio < 1
    qa.reset_usage()
    assert qa.usage == TokenUsage()


def test_refusal_raises(contract: Document) -> None:
    details = SimpleNamespace(category="cyber", explanation="Declined.")
    qa = make_qa(contract, FakeClient(fake_message(stop_reason="refusal", stop_details=details)))
    with pytest.raises(RefusalError) as info:
        qa.ask("anything?")
    assert info.value.category == "cyber"
    assert info.value.request_id == "req_test"


def test_truncation_and_fallback_flags(contract: Document) -> None:
    fallback = SimpleNamespace(type="fallback")
    msg = fake_message(stop_reason="max_tokens", model="claude-opus-4-8", extra_blocks=(fallback,))
    answer = make_qa(contract, FakeClient(msg)).ask("long?")
    assert answer.truncated
    assert answer.fallback_used
    assert answer.model == "claude-opus-4-8"


def test_fallback_options(contract: Document) -> None:
    client = FakeClient()
    make_qa(contract, client, fallbacks=None).ask("q?")
    make_qa(contract, client, fallbacks=["claude-opus-4-8"]).ask("q?")
    off, named = client.messages.calls
    assert "betas" not in off
    assert "fallbacks" not in off
    assert named["fallbacks"] == [{"model": "claude-opus-4-8"}]


def test_warm_cache(contract: Document) -> None:
    client = FakeClient(fake_message(text="", cache_write=4000, stop_reason="max_tokens"))
    qa = make_qa(contract, client)
    usage = qa.warm_cache()
    call = client.messages.calls[0]
    assert call["max_tokens"] == 0
    assert "fallbacks" not in call
    assert usage.cache_creation_input_tokens == 4000

    qa.ask("real question?", strategy=Strategy.CACHED_DOCUMENT)
    assert client.messages.calls[1]["system"] == call["system"]  # warm-up matches real prefix


def test_empty_question_rejected(contract: Document) -> None:
    with pytest.raises(ConfigurationError):
        make_qa(contract, FakeClient()).ask("   ")


@pytest.mark.parametrize(
    "bad",
    [{"effort": "ultra"}, {"max_tokens": 0}, {"cache_ttl": "2h"}, {"fallbacks": "auto"}],
)
def test_invalid_config(bad: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        QAConfig(**bad)


def test_from_text(fake_client: FakeClient) -> None:
    qa = DocumentQA.from_text("# A\n\nAlpha.\n\n# B\n\nBeta.", client=fake_client)
    assert len(qa.chunks.parents) == 2


async def test_async_ask_and_warm(contract: Document) -> None:
    client = FakeClient(fake_message(cache_write=10), fake_message(), is_async=True)
    qa = AsyncDocumentQA(contract, client=client)  # type: ignore[arg-type]
    assert (await qa.warm_cache()).cache_creation_input_tokens == 10
    answer = await qa.ask("Termination notice?")
    assert answer.text
    assert len(client.messages.calls) == 2


# -- answer cleaning and citation parsing (found by the local-model benchmark) ---------


def test_clean_answer_turns_leaked_tags_into_citations() -> None:
    from longdoc.prompts import clean_answer

    raw = (
        'According to section <section id="p0045" heading="19.2 Termination">19.2 '
        "Termination</section> [p0045], notice is 90 days. <question>x</question>"
    )
    assert clean_answer(raw) == (
        "According to section [p0045] 19.2 Termination [p0045], notice is 90 days. x"
    )
    assert clean_answer("Fee is due [p0003] [p0003].") == "Fee is due [p0003]."


def test_citations_accept_bare_and_bracketed_ids(contract: Document) -> None:
    from longdoc import ParentChildChunker
    from longdoc.prompts import extract_citations

    chunked = ParentChildChunker().chunk(contract)
    text = "See section p0002 and [p0001-c003]; also p9999 (invented) and xp0003."
    assert [p.id for p in extract_citations(text, chunked)] == ["p0002", "p0001"]
