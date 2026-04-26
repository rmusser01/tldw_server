from __future__ import annotations

import shutil

# Formatter commands are fixed executable paths and run with shell=False.
import subprocess  # nosec B404
from collections.abc import Iterable
from pathlib import Path


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


def maybe_format(paths: Iterable[str]) -> None:
    """Run Black and Ruff on the specified paths if available."""
    paths = [path for path in paths if Path(path).suffix in {".py", ".pyi"}]
    if not paths:
        return
    black = _tool_path("black")
    if black:
        subprocess.run([black, *paths], check=False)  # nosec B603
    ruff = _tool_path("ruff")
    if ruff:
        subprocess.run([ruff, "check", "--fix", *paths], check=False)  # nosec B603
