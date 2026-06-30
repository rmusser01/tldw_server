from __future__ import annotations

from typing import Any


def coerce_optional_nonempty_string(value: Any) -> str | None:
    """Normalize optional metadata strings and drop empty placeholders."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
