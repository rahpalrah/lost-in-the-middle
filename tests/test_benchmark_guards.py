"""The benchmark must never make billed API calls without explicit consent."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

RUN = Path(__file__).resolve().parent.parent / "benchmarks" / "contract_review" / "run.py"


@pytest.fixture(scope="module")
def run_module() -> ModuleType:
    sys.path.insert(0, str(RUN.parent))
    spec = importlib.util.spec_from_file_location("benchmark_run", RUN)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flags", [["--live"], ["--openai", "some-model"]])
def test_billed_modes_need_explicit_consent(
    run_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    monkeypatch.delenv("LONGDOC_ALLOW_LIVE_API", raising=False)
    assert run_module.main(flags) == 1
    assert "LONGDOC_ALLOW_LIVE_API=1" in capsys.readouterr().err


def test_openai_mode_needs_a_key_and_prices(
    run_module: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LONGDOC_ALLOW_LIVE_API", "1")
    assert run_module.main(["--openai", "some-model"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-a-key")
    assert run_module.main(["--openai", "some-model"]) == 1
    assert "--price-in" in capsys.readouterr().err
