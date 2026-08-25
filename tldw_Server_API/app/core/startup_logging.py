"""Helpers for safe startup logging."""

from __future__ import annotations

import os

_SHOW_KEY_TRUE_VALUES = {"true", "1", "yes"}
_LOGURU_LEVELS = {
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


def normalize_startup_log_level(value: str | None) -> str:
    """Return a valid Loguru startup level, defaulting to INFO."""
    normalized = (value or "INFO").strip().upper()
    return normalized if normalized in _LOGURU_LEVELS else "INFO"


def mask_api_key_for_startup_logs(api_key: str) -> str:
    """Return a short masked representation suitable for startup logs."""
    if not api_key or len(api_key) < 8:
        return "********"
    return f"{api_key[:4]}...{api_key[-4:]}"


def startup_api_key_log_value(api_key: str) -> str:
    """Return full key only when explicitly requested via env flag."""
    show_key = os.getenv("SHOW_API_KEY_ON_STARTUP", "false").lower() in _SHOW_KEY_TRUE_VALUES
    return api_key if show_key else mask_api_key_for_startup_logs(api_key)
