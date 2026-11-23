"""Shared fixtures: a sample contract and a fake Anthropic client (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from longdoc import Document

FILLER = (
    "The parties shall cooperate in good faith on all operational matters. "
    "Each party remains responsible for its own staff, systems and subcontractors. "
)


def make_contract() -> Document:
    """A 60-page contract where the critical condition sits on page 54."""
    pages: list[str] = []
    for n in range(1, 61):
        body = FILLER * 6
        if n == 1:
            body = "# 1. Definitions\n\n" + body
        if n == 20:
            body = "# 7. Payment\n\nInvoices are payable within 30 days of receipt.\n\n" + body
        if n == 54:
            body = (
                "# 19. Termination\n\nEither party may terminate this agreement on 90 days "
                "written notice, provided that the early termination fee in Schedule C is "
                "paid in full before the notice period ends.\n\n" + body
            )
        pages.append(body)
    return Document.from_pages(pages, source="contract.pdf")


@pytest.fixture
def contract() -> Document:
    return make_contract()


def fake_message(
    text: str = "Ninety days' notice, and the Schedule C fee must be paid first [p0003].",
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    cache_read: int = 0,
    cache_write: int = 0,
    extra_blocks: tuple[Any, ...] = (),
    stop_details: Any = None,
) -> Any:
    return SimpleNamespace(
        content=[*extra_blocks, SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        _request_id="req_test",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            iterations=None,
        ),
    )


@dataclass
class _Stream:
    message: Any

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


@dataclass
class _AsyncStream(_Stream):
    async def get_final_message(self) -> Any:
        return self.message


@dataclass
class FakeMessages:
    responses: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    is_async: bool = False

    def _next(self) -> Any:
        return self.responses.pop(0) if self.responses else fake_message()

    def stream(self, **kwargs: Any) -> _Stream:
        self.calls.append(kwargs)
        cls = _AsyncStream if self.is_async else _Stream
        return cls(self._next())

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        msg = self._next()
        if not self.is_async:
            return msg

        async def _coro() -> Any:
            return msg

        return _coro()


class FakeClient:
    def __init__(self, *responses: Any, is_async: bool = False) -> None:
        self.messages = FakeMessages(list(responses), is_async=is_async)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
