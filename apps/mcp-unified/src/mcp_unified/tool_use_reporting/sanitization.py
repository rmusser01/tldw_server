"""Sanitizers for metadata-only MCP tool-use reporting fields."""

from __future__ import annotations

import re
from typing import Any

MAX_SAFE_ID_LENGTH = 128
MAX_REASON_CODE_LENGTH = 64

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def sanitize_safe_id(
    value: Any,
    *,
    field: str,
    max_length: int = MAX_SAFE_ID_LENGTH,
) -> str | None:
    """Return a bounded allowlisted identifier or None when unsafe."""

    del field
    if value is None or not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > max_length:
        return None
    if "/" in text or "\\" in text or "@" in text:
        return None
    if not _SAFE_ID_RE.fullmatch(text):
        return None
    return text


def sanitize_reason_code(value: Any) -> str | None:
    """Return a safe reason code, unknown for non-blank unsafe values."""

    if value is None:
        return None
    safe = sanitize_safe_id(
        value,
        field="reason_code",
        max_length=MAX_REASON_CODE_LENGTH,
    )
    if safe is not None:
        return safe
    if isinstance(value, str) and value.strip():
        return "unknown"
    return None
