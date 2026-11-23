"""Claude client tests with a fake transport. No key, no network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from longdoc import ConfigurationError, Document, DocumentQA, QAConfig, RefusalError
from longdoc.anthropic_client import AnthropicClient
from longdoc.http import ProviderAuthError, Response

FAKE_KEY = "sk-ant-" + "f" * 10 + "-notreal-" + "9" * 10
DOC = Document.from_text("19.2 Termination for Convenience\n\nCustomer may terminate on 90 days.")


def message(**overrides: Any) -> Response:
    data = {
        "id": "msg_1",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Ninety days [p0000]."}],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 15,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 0,
        },
    }
    data.update(overrides)
    return Response(200, data, {"request-id": "req_claude_1"})


class FakeTransport:
    def __init__(self, response: Response | Exception | None = None) -> None:
        self.response = response or message()
        self.calls: list[tuple[str, Mapping[str, str], Mapping[str, Any]]] = []

    def __call__(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> Response:
        self.calls.append((url, headers, body))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def qa(t: FakeTransport, **config: Any) -> DocumentQA:
    return DocumentQA(
        DOC, client=AnthropicClient(api_key=FAKE_KEY, transport=t), config=QAConfig(**config)
    )


def test_request_shape_matches_the_messages_api() -> None:
    t = FakeTransport()
    answer = qa(t).ask("Notice period?")

    url, headers, body = t.calls[0]
    assert url.endswith("/v1/messages")
    assert headers["x-api-key"] == FAKE_KEY
    assert headers["anthropic-version"] == "2023-06-01"
    assert (
        headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
    )  # from the fallback option
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert isinstance(body["system"], list)
    assert body["messages"][0]["role"] == "user"
    assert "betas" not in body  # betas travel as a header, not in the body
    assert "thinking" not in body

    assert answer.text == "Ninety days [p0000]."
    assert answer.request_id == "req_claude_1"
    assert answer.usage.cache_creation_input_tokens == 500
    assert answer.usage.input_tokens == 120


def test_fallbacks_can_be_disabled_and_named() -> None:
    t = FakeTransport()
    qa(t, fallbacks=None).ask("q?")
    qa(t, fallbacks=["claude-opus-4-8"]).ask("q?")
    off, named = t.calls
    assert "anthropic-beta" not in off[1]
    assert "fallbacks" not in off[2]
    assert named[2]["fallbacks"] == [{"model": "claude-opus-4-8"}]


def test_cached_document_marks_the_cache_prefix_and_warmup_sends_zero_tokens() -> None:
    t = FakeTransport(message(content=[], stop_reason="max_tokens"))
    from longdoc import Strategy

    q = qa(t, strategy=Strategy.CACHED_DOCUMENT, cache_ttl="1h", fallbacks=None)
    q.warm_cache()
    body = t.calls[0][2]
    assert body["max_tokens"] == 0
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_refusal_and_fallback_markers_are_understood() -> None:
    refused = message(
        stop_reason="refusal",
        content=[],
        stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
    )
    with pytest.raises(RefusalError) as info:
        qa(FakeTransport(refused)).ask("q?")
    assert info.value.category == "cyber"
    assert info.value.request_id == "req_claude_1"

    rescued = message(
        content=[{"type": "fallback"}, {"type": "text", "text": "ok"}],
        model="claude-opus-4-8",
        usage={"input_tokens": 1, "output_tokens": 1, "iterations": [{"type": "fallback_message"}]},
    )
    answer = qa(FakeTransport(rescued)).ask("q?")
    assert answer.fallback_used
    assert answer.model == "claude-opus-4-8"


def test_truncation_and_thinking_blocks() -> None:
    resp = message(
        stop_reason="max_tokens",
        content=[{"type": "thinking", "thinking": ""}, {"type": "text", "text": "cut off"}],
    )
    answer = qa(FakeTransport(resp)).ask("q?")
    assert answer.truncated
    assert answer.text == "cut off"


def test_provider_errors_propagate() -> None:
    with pytest.raises(ProviderAuthError):
        qa(FakeTransport(ProviderAuthError("bad key", status=401))).ask("q?")


def test_key_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = AnthropicClient(transport=FakeTransport())
    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in repr(vars(client))


def test_base_url_comes_from_the_environment_and_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    assert AnthropicClient(api_key=FAKE_KEY).base_url == "http://127.0.0.1:9"
    assert AnthropicClient(api_key=FAKE_KEY, base_url="https://proxy.example/").base_url == (
        "https://proxy.example"
    )


async def test_async_view_runs_the_same_client() -> None:
    from longdoc import AsyncDocumentQA

    t = FakeTransport()
    client = AnthropicClient(api_key=FAKE_KEY, transport=t)
    aqa = AsyncDocumentQA(DOC, client=client.to_async(), config=QAConfig(fallbacks=None))
    answer = await aqa.ask("q?")
    assert answer.text == "Ninety days [p0000]."
    assert len(t.calls) == 1
