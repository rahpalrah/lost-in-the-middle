"""Read API keys and settings from a ``.env`` file, without extra dependencies.

A ``.env`` file keeps a secret out of your code and out of your shell history. Rules this
module follows, so a key is not leaked by accident:

* Values are never logged, printed or put in error messages; only variable *names* are.
* Real environment variables win over the file, unless you pass ``override=True``.
* Nothing is loaded implicitly. You must pass a path (``--env-file`` or ``LONGDOC_ENV_FILE``).
* On Linux and macOS a warning is logged if the file can be read by other users
  (fix with ``chmod 600 .env``).

Never commit the file. ``.env`` is already in this repository's ``.gitignore``.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Supports comments, ``export KEY=...`` and quoted values.

    Raises:
        ConfigurationError: The file is missing or a line is malformed (the message names
            the line number, never the content).
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"Env file not found: {p}")
    _warn_if_world_readable(p)
    values: dict[str, str] = {}
    for number, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not _NAME.match(name):
            raise ConfigurationError(f"{p}: line {number} is not KEY=VALUE.")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[name] = value
    return values


def load_env_file(path: str | os.PathLike[str], *, override: bool = False) -> list[str]:
    """Load a ``.env`` file into ``os.environ``. Returns the names that were set."""
    loaded = []
    for name, value in read_env_file(path).items():
        if override or name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    logger.debug("Loaded %d variable(s) from %s: %s", len(loaded), path, ", ".join(loaded))
    return loaded


def _warn_if_world_readable(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        logger.warning("%s can be read by other users. Restrict it with: chmod 600 %s", path, path)
