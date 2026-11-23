"""OpenAI adapter tests. Fake clients only: no network, no real keys."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from longdoc import (
    ConfigurationError,
    Document,
    DocumentQA,
    DocumentService,
    QAConfig,
    RefusalError,
    Strategy,
)
from longdoc.openai_client import AsyncOpenAIClient, OpenAIClient
from longdoc.server import create_app
from longdoc.server.asgi import app_from_env

# Key-shaped strings are built at run time so no key-looking text exists in the source.
FAKE_KEY = "sk-" + "f" * 12 + "-notreal-" + "9" * 12
CFG = QAConfig(model="test-chat-model", fallbacks=None)


def completion(
    text: str = "Ninety days [p0000].",
    *,
    finish: str = "stop",
    refusal: str | None = None,
    prompt: int = 1000,
    cached: int = 800,
    out: int = 20,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, refusal=refusal), finish_reason=finish
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=out,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
        model="test-chat-model-2026-01-01",
        _request_id="req_openai_1",
    )


class FakeOpenAI:
    def __init__(self, *responses: Any, raises: Exception | None = None) -> None:
        self.responses = list(responses)
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return self.responses.pop(0) if self.responses else completion()


class FakeAsyncOpenAI(FakeOpenAI):
    def __init__(self, *responses: Any, raises: Exception | None = None) -> None:
        super().__init__(*responses, raises=raises)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._acreate))

    async def _acreate(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)


DOC = Document.from_text("19.2 Termination for Convenience\n\nCustomer may terminate on 90 days.")


def make_qa(fake: FakeOpenAI, **client_kwargs: Any) -> DocumentQA:
    return DocumentQA(DOC, client=OpenAIClient(client=fake, **client_kwargs), config=CFG)


def test_request_and_answer_mapping() -> None:
    fake = FakeOpenAI()
    answer = make_qa(fake).ask("What notice is needed?")

    call = fake.calls[0]
    assert call["model"] == "test-chat-model"
    assert call["max_completion_tokens"] == CFG.max_tokens
    assert "max_tokens" not in call
    assert "temperature" not in call
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "<question>" in call["messages"][1]["content"]
    assert not {"betas", "fallbacks", "output_config", "thinking"} & set(call)  # Claude-only

    assert answer.text == "Ninety days [p0000]."
    assert [c.id for c in answer.citations] == ["p0000"]
    assert answer.model == "test-chat-model-2026-01-01"
    assert answer.request_id == "req_openai_1"
    assert answer.usage.input_tokens == 200  # 1000 prompt - 800 cached
    assert answer.usage.cache_read_input_tokens == 800
    assert answer.usage.total_input_tokens == 1000
    assert answer.usage.cache_hit_ratio == pytest.approx(0.8)


def test_optional_parameters_are_passed_through() -> None:
    fake = FakeOpenAI()
    make_qa(fake, token_param="max_tokens", temperature=0.0, reasoning_effort="low").ask("q?")
    call = fake.calls[0]
    assert call["max_tokens"] == CFG.max_tokens
    assert "max_completion_tokens" not in call
    assert call["temperature"] == 0.0
    assert call["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    ("kwargs", "category", "text"),
    [
        ({"text": "", "refusal": "I can't help with that."}, "refusal", "I can't help"),
        ({"text": "", "finish": "content_filter"}, "content_filter", ""),
    ],
)
def test_refusals_raise(kwargs: dict[str, Any], category: str, text: str) -> None:
    with pytest.raises(RefusalError) as info:
        make_qa(FakeOpenAI(completion(**kwargs))).ask("q?")
    assert info.value.category == category
    assert text in str(info.value)
    assert info.value.request_id == "req_openai_1"


def test_length_finish_is_truncation() -> None:
    answer = make_qa(FakeOpenAI(completion(finish="length"))).ask("q?")
    assert answer.truncated
    assert answer.stop_reason == "max_tokens"


def test_claude_model_name_is_rejected_with_a_clear_message() -> None:
    qa = DocumentQA(DOC, client=OpenAIClient(client=FakeOpenAI()))  # default model is Claude
    with pytest.raises(ConfigurationError, match="Claude model"):
        qa.ask("q?")


def test_warm_cache_makes_no_request() -> None:
    fake = FakeOpenAI()
    qa = DocumentQA(
        DOC,
        client=OpenAIClient(client=fake),
        config=QAConfig(model="test-chat-model", strategy=Strategy.CACHED_DOCUMENT, fallbacks=None),
    )
    assert qa.warm_cache().output_tokens == 0
    assert fake.calls == []


# -- credentials -----------------------------------------------------------------------


def test_missing_key_gives_a_helpful_error_without_any_secret() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_key_from_environment_is_used_but_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    client = OpenAIClient()
    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in repr(client.to_async())
    assert isinstance(client.to_async(), AsyncOpenAIClient)


def test_explicit_key_is_not_stored_on_our_objects() -> None:
    client = OpenAIClient(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in json.dumps({k: str(v) for k, v in vars(client).items() if k != "_sdk"})


def test_keyless_compatible_server_is_allowed() -> None:
    client = OpenAIClient(base_url="http://localhost:11434/v1")
    assert client.base_url == "http://localhost:11434/v1"


# -- service and async -----------------------------------------------------------------


async def test_service_async_path_and_usage() -> None:
    sync, aio = FakeOpenAI(), FakeAsyncOpenAI(completion(cached=0, prompt=300))
    service = DocumentService(client=OpenAIClient(client=sync, async_client=aio), config=CFG)
    service.add_document(DOC, document_id="d")
    answer = await service.aask("d", "Notice?")
    assert answer.usage.input_tokens == 300
    assert len(aio.calls) == 1
    assert sync.calls == []
    assert service.usage.input_tokens == 300


# -- server error mapping ---------------------------------------------------------------


def _status_error(cls: type[openai.APIStatusError], status: int, message: str) -> Exception:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls(message, response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (_status_error(openai.RateLimitError, 429, "slow down"), 429, "rate_limited"),
        (
            _status_error(openai.AuthenticationError, 401, f"Incorrect key {FAKE_KEY}"),
            502,
            "upstream_error",
        ),
        (_status_error(openai.InternalServerError, 500, "boom"), 502, "upstream_error"),
    ],
)
def test_server_maps_openai_errors_without_leaking(exc: Exception, status: int, code: str) -> None:
    service = DocumentService(
        client=OpenAIClient(
            client=FakeOpenAI(raises=exc), async_client=FakeAsyncOpenAI(raises=exc)
        ),
        config=CFG,
    )
    service.add_document(DOC, document_id="d")
    api = TestClient(create_app(service, api_key="k"))
    r = api.post(
        "/v1/documents/d/ask", json={"question": "q?"}, headers={"Authorization": "Bearer k"}
    )
    assert r.status_code == status
    assert r.json()["error"] == code
    assert FAKE_KEY not in r.text
    assert "boom" not in r.text


# -- ASGI configuration ------------------------------------------------------------------


def test_asgi_selects_openai_from_environment_and_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={FAKE_KEY}\nLONGDOC_MODEL=test-chat-model\n")
    env_file.chmod(0o600)
    app = app_from_env({"LONGDOC_PROVIDER": "openai", "LONGDOC_ENV_FILE": str(env_file)})
    service = app.state.service
    assert isinstance(service.client, OpenAIClient)
    assert service.config.model == "test-chat-model"
    assert service.config.fallbacks is None
    assert FAKE_KEY not in repr(service.client)


def test_asgi_openai_requires_a_model_and_known_provider() -> None:
    with pytest.raises(ConfigurationError, match="LONGDOC_MODEL"):
        app_from_env({"LONGDOC_PROVIDER": "openai", "OPENAI_API_KEY": FAKE_KEY})
    with pytest.raises(ConfigurationError, match="Unknown LONGDOC_PROVIDER"):
        app_from_env({"LONGDOC_PROVIDER": "acme"})


# -- command line ------------------------------------------------------------------------


def test_cli_openai_flow_and_key_safety(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from longdoc import cli

    doc = tmp_path / "terms.md"
    doc.write_text("# Fees\n\nThe late fee is 2 percent.", encoding="utf-8")

    assert cli.main(["ask", str(doc), "fee?", "--provider", "openai"]) == 1
    assert "--model" in capsys.readouterr().err

    monkeypatch.setattr(
        "longdoc.openai_client.OpenAIClient",
        lambda **kw: OpenAIClient(client=FakeOpenAI(completion("Two percent [p0000].", cached=0))),
    )
    assert cli.main(["ask", str(doc), "fee?", "--provider", "openai", "--model", "m"]) == 0
    assert "Two percent" in capsys.readouterr().out

    bad = _status_error(openai.AuthenticationError, 401, f"Incorrect API key: {FAKE_KEY}")
    monkeypatch.setattr(
        "longdoc.openai_client.OpenAIClient",
        lambda **kw: OpenAIClient(client=FakeOpenAI(raises=bad)),
    )
    assert cli.main(["ask", str(doc), "fee?", "--provider", "openai", "--model", "m"]) == 2
    err = capsys.readouterr().err
    assert FAKE_KEY not in err
    assert "OPENAI_API_KEY" in err


def test_cli_env_file_loads_names_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    from longdoc import cli

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")  # so teardown restores the real state
    monkeypatch.delenv("OPENAI_API_KEY")
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={FAKE_KEY}\n")
    env_file.chmod(0o600)
    doc = tmp_path / "t.md"
    doc.write_text("# A\n\nText.", encoding="utf-8")
    monkeypatch.setattr(
        "longdoc.openai_client.OpenAIClient",
        lambda **kw: OpenAIClient(client=FakeOpenAI(completion("ok", cached=0))),
    )
    code = cli.main(
        ["--env-file", str(env_file), "ask", str(doc), "q?", "--provider", "openai", "--model", "m"]
    )
    assert code == 0
    assert os.environ["OPENAI_API_KEY"] == FAKE_KEY
    out = capsys.readouterr()
    assert FAKE_KEY not in out.out + out.err
    assert cli.main(["--env-file", str(tmp_path / "missing.env"), "ask", str(doc), "q"]) == 1
