"""Deterministic parsing for local note wikilink projections."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

WIKILINK_PARSER_VERSION = 1
MAX_WIKILINK_TARGETS = 1_024

_WIKILINK_RE = re.compile(r"\[\[id:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]\]")


@dataclass(frozen=True, slots=True)
class WikilinkProjection:
    """One bounded, first-occurrence-ordered parser result."""

    target_note_ids: tuple[str, ...]
    truncated: bool
    parser_version: int = WIKILINK_PARSER_VERSION


def parse_wikilinks(
    content: str,
    *,
    source_note_id: str | None = None,
    max_targets: int = MAX_WIKILINK_TARGETS,
) -> WikilinkProjection:
    """Parse canonical UUID targets without resolving them against product rows."""

    if max_targets < 1:
        raise ValueError("max_targets must be at least one")
    normalized_source = _normalized_uuid(source_note_id) if source_note_id else None
    seen: set[str] = set()
    targets: list[str] = []
    truncated = False
    for match in _WIKILINK_RE.finditer(content or ""):
        target = _normalized_uuid(match.group(1))
        if target is None or target == normalized_source or target in seen:
            continue
        seen.add(target)
        if len(targets) == max_targets:
            truncated = True
            continue
        targets.append(target)
    return WikilinkProjection(tuple(targets), truncated)


def _normalized_uuid(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


__all__ = [
    "MAX_WIKILINK_TARGETS",
    "WIKILINK_PARSER_VERSION",
    "WikilinkProjection",
    "parse_wikilinks",
]
