"""Pure audit storage selection shared by HTTP dependencies and backup admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy

_FALSEY = {"0", "false", "no", "n", "off"}
_VALID_STORAGE_MODES = {"per_user", "shared"}


def audit_setting_bool(settings: Mapping[str, Any], key: str, default: bool) -> bool:
    """Coerce a settings value to bool, accepting common string representations."""
    raw = settings.get(key, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if is_truthy(s):
        return True
    if s in _FALSEY:
        return False
    logger.warning(f"Invalid {key}={raw!r}; using default {default}")
    return default


def resolve_audit_storage_mode(settings: Mapping[str, Any]) -> str:
    """Resolve audit storage mode with rollback precedence."""
    if audit_setting_bool(settings, "AUDIT_STORAGE_ROLLBACK", False):
        return "per_user"
    raw = settings.get("AUDIT_STORAGE_MODE", "per_user")
    try:
        mode = str(raw).strip().lower()
    except (TypeError, ValueError):
        mode = "per_user"
    if mode not in _VALID_STORAGE_MODES:
        logger.warning(f"Invalid AUDIT_STORAGE_MODE={raw!r}; using per_user")
        return "per_user"
    return mode
