"""Pure helpers for manuscript annotation anchoring."""

from __future__ import annotations

import hashlib
import json
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
_SELECTED_TEXT_REVIEW_SYSTEM_MESSAGE = (
    "You are a focused manuscript editor. Respond only with valid JSON for one annotation."
)


def create_document_fingerprint(text: str) -> str:
    """Create a stable fingerprint for scene text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_selected_text_review_prompt(
    *,
    scene_text: str,
    selected_text: str,
    category_hints: list[str],
    instruction: str | None,
) -> list[dict[str, str]]:
    """Build a prompt for reviewing one selected manuscript range."""
    allowed_categories = ", ".join(VALID_ANNOTATION_CATEGORIES)
    hints = [hint for hint in category_hints if hint in VALID_ANNOTATION_CATEGORIES]
    hint_text = ", ".join(hints) if hints else "Choose the best allowed category."
    instruction_text = instruction.strip() if instruction and instruction.strip() else "No additional instruction."
    user_prompt = (
        "Review only the selected text from the scene and return exactly one annotation.\n\n"
        f"Allowed categories: {allowed_categories}\n"
        f"Category hints: {hint_text}\n"
        f"Instruction: {instruction_text[:2000]}\n\n"
        "Return a JSON object with these fields:\n"
        '- "category": one allowed category\n'
        '- "body": concise reviewer note under 2000 characters\n'
        '- "suggested_fix": optional concrete rewrite under 8000 characters\n\n'
        f"Scene text:\n{scene_text[:12000]}\n\n"
        f"Selected text:\n{selected_text[:12000]}"
    )
    return [
        {"role": "system", "content": _SELECTED_TEXT_REVIEW_SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def parse_annotation_review_response(raw_text: str) -> list[dict[str, str]]:
    """Parse and validate one selected-text annotation from model output."""
    content = _strip_markdown_fences(raw_text)
    if not content:
        raise ValueError("Model response must contain valid JSON.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response must contain valid JSON.") from exc

    annotation_payload = _coerce_annotation_payload(payload)
    if len(annotation_payload) != 1:
        raise ValueError("Selected-text review must return exactly one annotation.")

    return [_validate_review_annotation(annotation_payload[0])]


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


def _coerce_annotation_payload(payload: object) -> list[object]:
    if isinstance(payload, dict):
        annotations = payload.get("annotations")
        if isinstance(annotations, list):
            return annotations
        annotation = payload.get("annotation")
        if isinstance(annotation, dict):
            return [annotation]
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("Model response must be a JSON object or array.")


def _validate_review_annotation(annotation: object) -> dict[str, str]:
    if not isinstance(annotation, dict):
        raise ValueError("Annotation entry must be a JSON object.")

    category = annotation.get("category")
    if not isinstance(category, str) or category not in VALID_ANNOTATION_CATEGORIES:
        raise ValueError("Annotation category must be one of the allowed values.")

    body = annotation.get("body")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("Annotation body must be a non-empty string.")
    normalized_body = body.strip()
    if len(normalized_body) >= 2000:
        raise ValueError("Annotation body must be under 2000 characters.")

    parsed = {"category": category, "body": normalized_body}
    suggested_fix = annotation.get("suggested_fix")
    if suggested_fix is not None:
        if not isinstance(suggested_fix, str):
            raise ValueError("Annotation suggested_fix must be a string when provided.")
        normalized_fix = suggested_fix.strip()
        if len(normalized_fix) >= 8000:
            raise ValueError("Annotation suggested_fix must be under 8000 characters.")
        if normalized_fix:
            parsed["suggested_fix"] = normalized_fix
    return parsed


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return text


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
