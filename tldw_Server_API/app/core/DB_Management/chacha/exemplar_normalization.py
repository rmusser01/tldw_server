"""Shared normalization helpers for persona and character exemplars."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError


def normalize_exemplar_enum(
    value: Any,
    *,
    allowed: tuple[str, ...],
    field_name: str,
    default: str,
) -> str:
    """Normalize and validate enum-like exemplar fields."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise InputError(f"Field '{field_name}' must be a string.")  # noqa: TRY003
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized not in allowed:
        raise InputError(  # noqa: TRY003
            f"Invalid value '{value}' for field '{field_name}'. Allowed: {', '.join(allowed)}"
        )
    return normalized


def normalize_exemplar_string_list(value: Any, field_name: str) -> list[str]:
    """Normalize list-like exemplar metadata to a string list."""
    if value is None:
        return []

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [stripped]

    if isinstance(parsed, set):
        parsed = list(parsed)

    if not isinstance(parsed, list):
        raise InputError(f"Field '{field_name}' must be a list of strings.")  # noqa: TRY003

    normalized: list[str] = []
    for item in parsed:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized
