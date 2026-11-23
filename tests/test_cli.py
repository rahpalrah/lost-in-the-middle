from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from longdoc import cli
from longdoc.client import DocumentQA

from .conftest import FakeClient


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "terms.md"
    path.write_text("# Fees\n\nThe late fee is 2 percent [p0000].", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DocumentQA, "__init__", _patched_init)


_original_init = DocumentQA.__init__


def _patched_init(self: Any, document: Any, **kwargs: Any) -> None:
    kwargs["client"] = FakeClient()
    _original_init(self, document, **kwargs)


def test_ask_text_output(doc: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", str(doc), "What is the late fee?"]) == 0
    out = capsys.readouterr().out
    assert "Q: What is the late fee?" in out
    assert "A: " in out


def test_ask_json_output(doc: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", str(doc), "Fee?", "--json", "--strategy", "cached_document"]) == 0
    line = json.loads(capsys.readouterr().out.strip())
    assert line["question"] == "Fee?"
    assert "usage" in line


def test_missing_file_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", str(tmp_path / "missing.pdf"), "q?"]) == 1
    assert "error:" in capsys.readouterr().err


def test_builtin_server_is_one_process(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["serve", "--workers", "2"]) == 1
    assert "--framework fastapi" in capsys.readouterr().err


def test_fastapi_workers_need_a_shared_store(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("uvicorn")
    assert cli.main(["serve", "--framework", "fastapi", "--workers", "2"]) == 1
    assert "--store" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "longdoc" in capsys.readouterr().out
