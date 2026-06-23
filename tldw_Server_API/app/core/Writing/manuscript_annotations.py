"""Pure helpers for manuscript annotation anchoring."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping


VALID_ANNOTATION_STATUSES = ("open", "resolved")
VALID_ANNOTATION_SOURCES = ("user", "ai_selected_text", "ai_scene_review")
VALID_ANNOTATION_CATEGORIES = (
    "style",
    "clarity",
    "pacing",
    "continuity",
    "character",
    "worldbuilding",
    "structure",
    "research",
    "other",
)
VALID_TARGET_TYPES = ("scene", "chapter", "project")
ANCHOR_CONTEXT_CHARS = 240


def create_document_fingerprint(text: str) -> str:
    """Create a stable fingerprint for scene text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_scene_anchor(text: str, *, start: int, end: int, scene_version: int) -> dict[str, object]:
    """Build persisted range-anchor metadata from saved scene text."""
    normalized_start, normalized_end = _validate_range(text, start, end)
    selected_text = text[normalized_start:normalized_end]
    return {
        "scene_version": int(scene_version),
        "anchor_start": normalized_start,
        "anchor_end": normalized_end,
        "selected_text": selected_text,
        "document_fingerprint": create_document_fingerprint(text),
        "anchor_prefix": text[max(0, normalized_start - ANCHOR_CONTEXT_CHARS):normalized_start],
        "anchor_suffix": text[normalized_end:normalized_end + ANCHOR_CONTEXT_CHARS],
        "anchor_status": "attached",
    }


def derive_scene_anchor_status(
    anchor: Mapping[str, object],
    text: str,
    *,
    current_scene_version: int | None = None,
) -> dict[str, object]:
    """Derive the current scene-anchor status without mutating persisted metadata."""
    if anchor.get("target_type") in {"chapter", "project"}:
        return _status("scene_level")

    selected_text = anchor.get("selected_text")
    start = anchor.get("anchor_start")
    end = anchor.get("anchor_end")
    if not isinstance(selected_text, str) or not selected_text:
        return _status("scene_level")
    if not isinstance(start, int) or not isinstance(end, int):
        return _status("scene_level")

    if (
        _scene_version_matches(anchor.get("scene_version"), current_scene_version)
        and _is_valid_existing_range(text, start, end)
        and text[start:end] == selected_text
    ):
        return _status("attached", start, end)

    if (
        _scene_version_is_malformed(anchor.get("scene_version"), current_scene_version)
        and _is_valid_existing_range(text, start, end)
        and text[start:end] == selected_text
    ):
        return _status("attached", start, end)

    selected_matches = _find_all(text, selected_text)
    if len(selected_matches) == 1:
        match_start = selected_matches[0]
        return _status("reattached", match_start, match_start + len(selected_text))

    context_matches = _find_context_ranges(anchor, text, selected_text, selected_matches)
    if len(context_matches) == 1:
        match_start, match_end = context_matches[0]
        return _status("reattached", match_start, match_end)

    return _status("needs_review")


def _validate_range(text: str, start: int, end: int) -> tuple[int, int]:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("Anchor offsets must be integers.")
    if start < 0 or end > len(text) or start >= end:
        raise ValueError("Anchor range must select non-empty text within the scene.")
    return start, end


def _is_valid_existing_range(text: str, start: int, end: int) -> bool:
    return 0 <= start < end <= len(text)


def _scene_version_matches(
    scene_version: object,
    current_scene_version: int | None,
) -> bool:
    if current_scene_version is None or scene_version is None:
        return True
    try:
        return int(scene_version) == current_scene_version
    except (TypeError, ValueError):
        return False


def _scene_version_is_malformed(
    scene_version: object,
    current_scene_version: int | None,
) -> bool:
    if current_scene_version is None or scene_version is None:
        return False
    try:
        int(scene_version)
    except (TypeError, ValueError):
        return True
    return False


def _find_all(text: str, needle: str) -> list[int]:
    matches: list[int] = []
    position = text.find(needle)
    while position != -1:
        matches.append(position)
        position = text.find(needle, position + 1)
    return matches


def _find_context_ranges(
    anchor: Mapping[str, object],
    text: str,
    selected_text: str,
    selected_matches: list[int],
) -> list[tuple[int, int]]:
    prefix = anchor.get("anchor_prefix")
    suffix = anchor.get("anchor_suffix")
    if not isinstance(prefix, str):
        prefix = ""
    if not isinstance(suffix, str):
        suffix = ""
    if not prefix and not suffix:
        return []

    if selected_matches:
        return [
            (match_start, match_start + len(selected_text))
            for match_start in selected_matches
            if _context_matches(text, match_start, match_start + len(selected_text), prefix, suffix)
        ]

    return _find_replacement_ranges(text, prefix, suffix)


def _find_replacement_ranges(text: str, prefix: str, suffix: str) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()

    if prefix:
        prefix_ends = [start + len(prefix) for start in _find_all(text, prefix)]
    else:
        prefix_ends = [0]

    if suffix:
        suffix_starts = _find_all(text, suffix)
    else:
        suffix_starts = [len(text)]

    for start in prefix_ends:
        for end in suffix_starts:
            if start < end:
                candidates.add((start, end))

    return sorted(candidates)


def _context_matches(text: str, start: int, end: int, prefix: str, suffix: str) -> bool:
    prefix_matches = not prefix or text[:start].endswith(prefix)
    suffix_matches = not suffix or text[end:].startswith(suffix)
    return prefix_matches and suffix_matches


def _status(
    anchor_status: str,
    derived_start: int | None = None,
    derived_end: int | None = None,
) -> dict[str, object]:
    return {
        "anchor_status": anchor_status,
        "derived_start": derived_start,
        "derived_end": derived_end,
    }
