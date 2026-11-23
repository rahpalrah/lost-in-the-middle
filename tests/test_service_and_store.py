from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from longdoc import (
    ConfigurationError,
    Document,
    DocumentLoadError,
    DocumentNotFoundError,
    DocumentService,
    DocumentStore,
    InMemoryStore,
    JSONFileStore,
    ParentChildChunker,
    TokenUsage,
    answer_to_dict,
    chunked_from_dict,
    chunked_to_dict,
)

from .conftest import FakeClient, fake_message

# -- serialization --------------------------------------------------------------------


def test_chunked_round_trip(contract: Document) -> None:
    chunked = ParentChildChunker().chunk(contract)
    data = json.loads(json.dumps(chunked_to_dict(chunked)))
    restored = chunked_from_dict(data)
    assert restored.parents == chunked.parents
    assert restored.children == chunked.children
    assert restored.source == chunked.source


@pytest.mark.parametrize(
    "bad", [{"schema_version": 99}, {"schema_version": 1}, {"schema_version": 1, "parents": [{}]}]
)
def test_chunked_from_dict_rejects_bad_data(bad: dict[str, Any]) -> None:
    with pytest.raises(DocumentLoadError):
        chunked_from_dict(bad)


# -- stores ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "json"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> DocumentStore:
    return InMemoryStore() if request.param == "memory" else JSONFileStore(tmp_path / "docs")


def test_store_crud(store: DocumentStore, contract: Document) -> None:
    assert isinstance(store, DocumentStore)
    chunked = ParentChildChunker().chunk(contract)
    assert store.get("abc") is None
    store.put("abc", chunked)
    store.put("xyz", chunked)
    got = store.get("abc")
    assert got is not None
    assert got.parents == chunked.parents
    assert list(store.list_ids()) == ["abc", "xyz"]
    assert store.delete("abc") is True
    assert store.delete("abc") is False
    assert list(store.list_ids()) == ["xyz"]


def test_json_store_rejects_path_traversal(tmp_path: Path, contract: Document) -> None:
    store = JSONFileStore(tmp_path)
    with pytest.raises(ConfigurationError):
        store.put("../evil", ParentChildChunker().chunk(contract))
    with pytest.raises(ConfigurationError):
        store.get("a/b")


def test_json_store_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        JSONFileStore(tmp_path).get("bad")


def test_json_store_leaves_no_temp_files(tmp_path: Path, contract: Document) -> None:
    store = JSONFileStore(tmp_path)
    store.put("doc1", ParentChildChunker().chunk(contract))
    assert [p.name for p in tmp_path.iterdir()] == ["doc1.json"]


# -- service --------------------------------------------------------------------------


def make_service(*responses: Any, store: DocumentStore | None = None) -> DocumentService:
    return DocumentService(
        store=store,
        client=FakeClient(*responses),
        async_client=FakeClient(*responses, is_async=True),
        max_cached_documents=2,
    )


def test_service_lifecycle(contract: Document) -> None:
    service = make_service()
    info = service.add_document(contract, document_id="contract-1")
    assert info.id == "contract-1"
    assert info.pages == 60
    assert info.sections > 1
    assert service.get_info("contract-1") == info
    assert [i.id for i in service.list_documents()] == ["contract-1"]

    sections = service.retrieve("contract-1", "termination fee")
    assert any(s.page_start == 54 for s in sections)

    answer = service.ask("contract-1", "Notice period?")
    assert answer.text
    assert service.usage == answer.usage

    assert service.delete("contract-1") is True
    with pytest.raises(DocumentNotFoundError):
        service.ask("contract-1", "again?")


def test_service_generates_ids_and_validates_them() -> None:
    service = make_service()
    info = service.add_text("# A\n\nAlpha text.")
    assert len(info.id) == 32
    with pytest.raises(ConfigurationError):
        service.add_text("x", document_id="has spaces")


def test_service_replacing_a_document_refreshes_the_index() -> None:
    service = make_service()
    service.add_text("Payment is due in 30 days.", document_id="d")
    assert service.retrieve("d", "payment")
    service.add_text("Delivery happens on Mondays.", document_id="d")
    assert service.retrieve("d", "payment") == ()
    assert service.retrieve("d", "delivery")


def test_service_shares_persistent_store(tmp_path: Path, contract: Document) -> None:
    store = JSONFileStore(tmp_path)
    make_service(store=store).add_document(contract, document_id="shared")
    other = make_service(store=JSONFileStore(tmp_path))  # e.g. another worker process
    assert other.get_info("shared").pages == 60
    assert other.ask("shared", "fee?").text


def test_service_lru_cache_is_bounded() -> None:
    service = make_service()
    for i in range(4):
        service.add_text(f"Document number {i} about topic{i}.", document_id=f"d{i}")
        service.retrieve(f"d{i}", f"topic{i}")
    assert len(service._sync_qa) == 2


async def test_service_async_ask(contract: Document) -> None:
    service = make_service(fake_message(cache_read=50))
    service.add_document(contract, document_id="c")
    answer = await service.aask("c", "Termination?")
    assert service.usage == TokenUsage(
        input_tokens=100, output_tokens=20, cache_read_input_tokens=50
    )
    data = answer_to_dict(answer)
    assert data["usage"]["cache_read_input_tokens"] == 50
    assert json.dumps(data)  # JSON-serializable
