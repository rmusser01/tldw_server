from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.testing import is_truthy


def _coerce_bool_option(value: Any, default: bool = False) -> bool:
    """Normalize loose option values into stable booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return is_truthy(value.strip().lower())
    return bool(value)
