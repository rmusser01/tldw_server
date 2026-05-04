from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    "coverage",
    "target",
    "site",
)


@dataclass(frozen=True)
class CodeGraphSettings:
    """Configuration for the native CodeGraph foundation."""

    index_base_dir: Path = Path("Databases/codegraph")
    max_file_size_bytes: int = 1_048_576
    foreground_max_files: int = 500
    foreground_max_bytes: int = 50_000_000
    max_index_seconds: float = 20.0
    max_context_chars: int = 35_000
    max_search_results: int = 100
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> CodeGraphSettings:
        raw = dict(values or {})
        defaults = cls()

        return cls(
            index_base_dir=_path_or_default(raw.get("index_base_dir"), defaults.index_base_dir),
            max_file_size_bytes=_positive_int(
                raw.get("max_file_size_bytes"),
                defaults.max_file_size_bytes,
            ),
            foreground_max_files=_positive_int(
                raw.get("foreground_max_files"),
                defaults.foreground_max_files,
            ),
            foreground_max_bytes=_positive_int(
                raw.get("foreground_max_bytes"),
                defaults.foreground_max_bytes,
            ),
            max_index_seconds=_positive_float(
                raw.get("max_index_seconds"),
                defaults.max_index_seconds,
            ),
            max_context_chars=_positive_int(
                raw.get("max_context_chars"),
                defaults.max_context_chars,
            ),
            max_search_results=_positive_int(
                raw.get("max_search_results"),
                defaults.max_search_results,
            ),
            exclude_dirs=_coerce_exclude_dirs(raw.get("exclude_dirs"), defaults.exclude_dirs),
        )


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _positive_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.001, parsed)


def _path_or_default(value: Any, default: Path) -> Path:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return Path(text).expanduser()


def _coerce_exclude_dirs(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple, set)):
        return default

    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return tuple(items) or default
