"""OpenAI client tests: a fake transport stands in for the network (no key, no HTTP)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from longdoc import (
    ConfigurationError,
    Document,
    DocumentQA,
    DocumentService,
    QAConfig,
    RefusalError,
    Strategy,
)
from longdoc.http import ProviderQuotaError, Response
from longdoc.openai_client import OpenAIClient

# Key-shaped strings are built at run time so no key-looking text exists in the source.
FAKE_KEY = "sk-" + "f" * 12 + "-notreal-" + "9" * 12
CFG = QAConfig(model="test-chat-model", fallbacks=None)
DOC = Document.from_text("19.2 Termination for Convenience\n\nCustomer may terminate on 90 days.")


def completion(
    text: str = "Ninety days [p0000].",
    *,
    finish: str = "stop",
    refusal: str | None = None,
    prompt: int = 1000,
    cached: int = 800,
    out: int = 20,
) -> Response:
    return Response(
        200,
        {
            "choices": [
                {"message": {"content": text, "refusal": refusal}, "finish_reason": finish}
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": out,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
            "model": "test-chat-model-2026-01-01",
        },
        {"x-request-id": "req_openai_1"},
    )


class FakeTransport:
    def __init__(self, *responses: Response | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], Mapping[str, Any]]] = []

    def __call__(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> Response:
        self.calls.append((url, headers, body))
        item = self.responses.pop(0) if self.responses else completion()
        if isinstance(item, Exception):
            raise item
        return item


def make_qa(transport: FakeTransport, **client_kwargs: Any) -> DocumentQA:
    client = OpenAIClient(api_key=FAKE_KEY, transport=transport, **client_kwargs)
    return DocumentQA(DOC, client=client, config=CFG)


def test_request_and_answer_mapping() -> None:
    t = FakeTransport()
    answer = make_qa(t).ask("What notice is needed?")

    url, headers, body = t.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions" or url.endswith("/chat/completions")
    assert headers == {"authorization": f"Bearer {FAKE_KEY}"}
    assert body["model"] == "test-chat-model"
    assert body["max_completion_tokens"] == CFG.max_tokens
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "<question>" in body["messages"][1]["content"]
    assert not {"betas", "fallbacks", "output_config", "thinking"} & set(body)  # Claude-only

    assert answer.text == "Ninety days [p0000]."
    assert [c.id for c in answer.citations] == ["p0000"]
    assert answer.model == "test-chat-model-2026-01-01"
    assert answer.request_id == "req_openai_1"
    assert answer.usage.input_tokens == 200  # 1000 prompt - 800 cached
    assert answer.usage.cache_read_input_tokens == 800
    assert answer.usage.total_input_tokens == 1000
    assert answer.usage.cache_hit_ratio == pytest.approx(0.8)


def test_optional_parameters_are_passed_through() -> None:
    t = FakeTransport()
    make_qa(t, token_param="max_tokens", temperature=0.0, reasoning_effort="low").ask("q?")
    body = t.calls[0][2]
    assert body["max_tokens"] == CFG.max_tokens
    assert "max_completion_tokens" not in body
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    ("kwargs", "category", "text"),
    [
        ({"text": "", "refusal": "I can't help with that."}, "refusal", "I can't help"),
        ({"text": "", "finish": "content_filter"}, "content_filter", ""),
    ],
)
def test_refusals_raise(kwargs: dict[str, Any], category: str, text: str) -> None:
    with pytest.raises(RefusalError) as info:
        make_qa(FakeTransport(completion(**kwargs))).ask("q?")
    assert info.value.category == category
    assert text in str(info.value)
    assert info.value.request_id == "req_openai_1"


def test_length_finish_is_truncation() -> None:
    answer = make_qa(FakeTransport(completion(finish="length"))).ask("q?")
    assert answer.truncated
    assert answer.stop_reason == "max_tokens"


def test_claude_model_name_is_rejected_with_a_clear_message() -> None:
    qa = DocumentQA(DOC, client=OpenAIClient(api_key=FAKE_KEY, transport=FakeTransport()))
    with pytest.raises(ConfigurationError, match="Claude model"):
        qa.ask("q?")


def test_custom_server_may_use_any_model_name() -> None:
    t = FakeTransport()
    client = OpenAIClient(base_url="http://127.0.0.1:8080/v1", transport=t)
    DocumentQA(DOC, client=client).ask("q?")  # default model name is allowed on custom servers
    assert t.calls[0][0] == "http://127.0.0.1:8080/v1/chat/completions"
    assert t.calls[0][1] == {}  # no key, no Authorization header


def test_warm_cache_makes_no_request() -> None:
    t = FakeTransport()
    qa = DocumentQA(
        DOC,
        client=OpenAIClient(api_key=FAKE_KEY, transport=t),
        config=QAConfig(model="test-chat-model", strategy=Strategy.CACHED_DOCUMENT, fallbacks=None),
    )
    assert qa.warm_cache().output_tokens == 0
    assert t.calls == []


def test_provider_errors_propagate_unchanged() -> None:
    quota = ProviderQuotaError("no credit", status=429, code="credit_balance_exhausted")
    with pytest.raises(ProviderQuotaError):
        make_qa(FakeTransport(quota)).ask("q?")


# -- credentials -----------------------------------------------------------------------


def test_missing_key_gives_a_helpful_error_without_any_secret() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_key_from_environment_is_used_but_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    t = FakeTransport()
    client = OpenAIClient(transport=t)
    DocumentQA(DOC, client=client, config=CFG).ask("q?")
    assert t.calls[0][1] == {"authorization": f"Bearer {FAKE_KEY}"}
    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in repr(vars(client))
    assert FAKE_KEY not in repr(client.to_async())


# -- async and service -----------------------------------------------------------------


async def test_service_async_path_and_usage() -> None:
    t = FakeTransport(completion(cached=0, prompt=300))
    service = DocumentService(client=OpenAIClient(api_key=FAKE_KEY, transport=t), config=CFG)
    service.add_document(DOC, document_id="d")
    answer = await service.aask("d", "Notice?")
    assert answer.usage.input_tokens == 300
    assert len(t.calls) == 1
    assert service.usage.input_tokens == 300
