"""Expression slot normalization for visual identity packs."""

from __future__ import annotations

import re

CANONICAL_EXPRESSION_SLOTS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "thinking",
    "confused",
    "surprised",
)
CUSTOM_EXPRESSION_PREFIX = "custom:"

EXPRESSION_ALIASES = {
    "default": "neutral",
    "normal": "neutral",
    "calm": "neutral",
    "joy": "happy",
    "joyful": "happy",
    "cheerful": "happy",
    "hype": "excited",
    "thrilled": "excited",
    "upset": "sad",
    "sorrowful": "sad",
    "mad": "angry",
    "annoyed": "angry",
    "furious": "angry",
    "anger": "angry",
    "thoughtful": "thinking",
    "pondering": "thinking",
    "unsure": "confused",
    "puzzled": "confused",
    "shocked": "surprised",
    "astonished": "surprised",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_expression_key(value: str) -> str | None:
    """Normalize a user-facing expression label into a canonical or custom key."""
    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    if not raw_value:
        return None

    if raw_value.lower().startswith(CUSTOM_EXPRESSION_PREFIX):
        custom_part = raw_value[len(CUSTOM_EXPRESSION_PREFIX) :]
        custom_key = _sanitize_expression_token(custom_part)
        return f"{CUSTOM_EXPRESSION_PREFIX}{custom_key}" if custom_key else None

    normalized = _sanitize_expression_token(raw_value)
    if not normalized:
        return None
    if normalized in CANONICAL_EXPRESSION_SLOTS:
        return normalized
    alias = EXPRESSION_ALIASES.get(normalized)
    if alias is not None:
        return alias
    return f"{CUSTOM_EXPRESSION_PREFIX}{normalized}"


def normalize_expression_filename(filename: str) -> str | None:
    """Normalize a source filename stem into a canonical or custom expression key."""
    if not isinstance(filename, str):
        return None

    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if "." in basename:
        basename = basename.rsplit(".", 1)[0]
    normalized = _sanitize_expression_token(basename)
    if not normalized:
        return None
    if normalized in CANONICAL_EXPRESSION_SLOTS:
        return normalized
    alias = EXPRESSION_ALIASES.get(normalized)
    if alias is not None:
        return alias
    return f"{CUSTOM_EXPRESSION_PREFIX}{normalized}"


def is_custom_expression_key(value: str) -> bool:
    """Return whether a value normalizes to a custom expression key."""
    normalized = normalize_expression_key(value)
    return normalized is not None and normalized.startswith(CUSTOM_EXPRESSION_PREFIX)


def display_label_for_expression_key(value: str) -> str:
    """Build a human-readable label for a canonical, alias, or custom expression key."""
    normalized = normalize_expression_key(value)
    if normalized is None:
        return ""
    if normalized.startswith(CUSTOM_EXPRESSION_PREFIX):
        normalized = normalized[len(CUSTOM_EXPRESSION_PREFIX) :]
    return normalized.replace("_", " ").title()


def _sanitize_expression_token(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub("_", value.strip().lower())
    return normalized.strip("_")
