"""Shared path-scope candidate contracts for MCP policy enforcement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

PathScopeAction = Literal["read", "edit", "write"]

_VALID_ACTIONS = {"read", "edit", "write"}


@dataclass(frozen=True, slots=True)
class PathScopeCandidate:
    """One path/action candidate derived before path-scope enforcement."""

    path: str
    action: PathScopeAction
    source: str
    display_path: str | None = None
    requires_existing_file: bool = False
    creates_file: bool = False
    workspace_id: str | None = None


def _clean_optional_string(value: Any) -> str | None:
    """Normalize optional candidate strings by trimming blanks to None."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_path_scope_candidate(raw: PathScopeCandidate | Mapping[str, Any]) -> PathScopeCandidate:
    """Normalize a raw candidate object from a module into the shared contract."""

    if isinstance(raw, PathScopeCandidate):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("path scope candidate must be an object")

    path = _clean_optional_string(raw.get("path"))
    if path is None:
        raise ValueError("path scope candidate path is required")

    action = _clean_optional_string(raw.get("action"))
    if action not in _VALID_ACTIONS:
        raise ValueError("path scope candidate action is invalid")

    source = _clean_optional_string(raw.get("source")) or "module"
    return PathScopeCandidate(
        path=path,
        action=action,  # type: ignore[arg-type]
        source=source,
        display_path=_clean_optional_string(raw.get("display_path")),
        requires_existing_file=bool(raw.get("requires_existing_file", False)),
        creates_file=bool(raw.get("creates_file", False)),
        workspace_id=_clean_optional_string(raw.get("workspace_id")),
    )


def normalize_path_scope_candidates(
    raw_items: Iterable[PathScopeCandidate | Mapping[str, Any]] | None,
) -> list[PathScopeCandidate]:
    """Normalize module-produced path-scope candidates into a list."""

    if raw_items is None:
        return []
    return [normalize_path_scope_candidate(item) for item in raw_items]
