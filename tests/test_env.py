from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

from longdoc import ConfigurationError
from longdoc.env import load_env_file, read_env_file

SECRET = "s3cr3t-" + "v" * 20  # not key-shaped; used to prove values never leak


def write(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_parses_comments_quotes_export_and_inline_comments(tmp_path: Path) -> None:
    f = write(
        tmp_path / ".env",
        "# comment\n\nA=plain\nexport B=\"quoted value\"\nC='single'\nD=value # trailing\n"
        "E=has#hash\nF=\n",
    )
    assert read_env_file(f) == {
        "A": "plain",
        "B": "quoted value",
        "C": "single",
        "D": "value",
        "E": "has#hash",
        "F": "",
    }


def test_malformed_line_error_never_contains_the_value(tmp_path: Path) -> None:
    f = write(tmp_path / ".env", f"OK=1\nthis line has {SECRET} and no equals\n")
    with pytest.raises(ConfigurationError) as info:
        read_env_file(f)
    assert "line 2" in str(info.value)
    assert SECRET not in str(info.value)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        read_env_file(tmp_path / "nope.env")


def test_real_environment_wins_unless_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_TEST_A", "from-shell")
    monkeypatch.setenv("LD_TEST_B", "x")
    monkeypatch.delenv("LD_TEST_B")
    f = write(tmp_path / ".env", "LD_TEST_A=from-file\nLD_TEST_B=from-file\n")
    assert load_env_file(f) == ["LD_TEST_B"]
    assert os.environ["LD_TEST_A"] == "from-shell"
    assert load_env_file(f, override=True) == ["LD_TEST_A", "LD_TEST_B"]
    assert os.environ["LD_TEST_A"] == "from-file"
    monkeypatch.delenv("LD_TEST_B")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_warns_when_file_is_readable_by_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="longdoc.env"):
        read_env_file(write(tmp_path / "open.env", f"K={SECRET}\n", mode=0o644))
        read_env_file(write(tmp_path / "closed.env", f"K={SECRET}\n", mode=0o600))
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "chmod 600" in warnings[0]
    assert SECRET not in " ".join(warnings)


# -- repository hygiene: secrets must never be committed ------------------------------


def _tracked_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    return [root / n for n in out.decode().split("\0") if n]


def test_no_key_shaped_strings_in_tracked_files() -> None:
    import re

    pattern = re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}")
    offenders = []
    for path in _tracked_files():
        if path.suffix in {".pdf", ".gguf", ".png", ".db"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            offenders.append(path.name)
    assert offenders == [], f"key-shaped text found in: {offenders}"


def test_env_files_are_ignored_by_git() -> None:
    root = Path(__file__).resolve().parent.parent
    ignore = (root / ".gitignore").read_text()
    assert ".env" in ignore.splitlines()
    assert (root / ".env.example").exists()
    example = (root / ".env.example").read_text()
    for line in example.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            assert line == "OPENAI_API_KEY=", "the example file must not contain a key"
