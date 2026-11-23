"""Tests for the local-model adapter, using a stand-in for llama_cpp.Llama."""

from __future__ import annotations

from typing import Any

import pytest

from longdoc import Document, DocumentQA, DocumentService, QAConfig, Strategy
from longdoc.local import (
    AsyncLlamaCppClient,
    ContextWindowExceededError,
    LlamaCppClient,
    to_chat_messages,
)


class FakeLlama:
    def __init__(self, reply: str = "Ninety days [p0000].", finish: str = "stop") -> None:
        self.reply = reply
        self.finish = finish
        self.calls: list[dict[str, Any]] = []

    def tokenize(self, data: bytes, add_bos: bool = True) -> list[int]:
        return list(range(len(data.split())))

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": self.reply}, "finish_reason": self.finish}],
            "usage": {"prompt_tokens": 321, "completion_tokens": 12},
        }


def make_client(**kw: Any) -> tuple[LlamaCppClient, FakeLlama]:
    fake = FakeLlama(**kw)
    return LlamaCppClient("models/tiny-model.gguf", llama=fake, n_ctx=4096), fake


def test_to_chat_messages_flattens_blocks() -> None:
    chat = to_chat_messages(
        [{"type": "text", "text": "rules"}, {"type": "text", "text": "doc", "cache_control": {}}],
        [{"role": "user", "content": "question"}],
    )
    assert chat == [
        {"role": "system", "content": "rules\n\ndoc"},
        {"role": "user", "content": "question"},
    ]


def test_document_qa_runs_on_local_model() -> None:
    client, fake = make_client()
    doc = Document.from_text(
        "19.2 Termination for Convenience\n\nCustomer may terminate on 90 days."
    )
    qa = DocumentQA(doc, client=client, config=QAConfig(model=client.model_name, fallbacks=None))
    answer = qa.ask("What notice is needed to terminate for convenience?")

    assert answer.text == "Ninety days [p0000]."
    assert [c.id for c in answer.citations] == ["p0000"]
    assert answer.model == "local:tiny-model.gguf"
    assert answer.usage.input_tokens == 321
    assert answer.usage.cache_read_input_tokens == 0
    call = fake.calls[0]
    assert call["temperature"] == 0.0
    assert call["messages"][0]["role"] == "system"
    assert "<question>" in call["messages"][1]["content"]


def test_truncation_maps_to_max_tokens() -> None:
    client, _ = make_client(finish="length")
    qa = DocumentQA(Document.from_text("Some text."), client=client)
    assert qa.ask("anything?").truncated


def test_context_window_exceeded() -> None:
    client, _ = make_client()
    huge = Document.from_text("word " * 20_000)
    qa = DocumentQA(huge, client=client, config=QAConfig(strategy=Strategy.CACHED_DOCUMENT))
    with pytest.raises(ContextWindowExceededError) as info:
        qa.ask("q?")
    assert info.value.n_ctx == 4096


def test_warm_cache_is_a_no_op() -> None:
    client, fake = make_client()
    qa = DocumentQA(Document.from_text("Some text."), client=client)
    assert qa.warm_cache().output_tokens == 0
    assert fake.calls == []


async def test_service_shares_one_model_for_sync_and_async() -> None:
    client, fake = make_client()
    service = DocumentService(client=client)
    assert isinstance(service.async_client, AsyncLlamaCppClient)
    assert service.async_client.sync_client is client
    service.add_text("5.1 Invoicing\n\nInvoices are payable in 30 days.", document_id="d")
    answer = await service.aask("d", "When are invoices payable?")
    assert answer.text
    assert len(fake.calls) == 1


def test_missing_runtime_has_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_llama(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_llama)
    with pytest.raises(ImportError, match=r"longdoc-qa\[local\]"):
        LlamaCppClient("model.gguf")


def test_llama_cpp_embedder_orders_by_index() -> None:
    from longdoc.local import LlamaCppEmbedder

    class FakeEmbedModel:
        def create_embedding(self, texts: list[str]) -> dict[str, Any]:
            data = [{"index": i, "embedding": [float(i), 1]} for i in range(len(texts))]
            return {"data": list(reversed(data))}

    embed = LlamaCppEmbedder("bge.gguf", llama=FakeEmbedModel())
    assert embed(["a", "b", "c"]) == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
