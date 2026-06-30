"""Dependency-free environment helpers for MCP Unified.

These helpers mirror the small truthy/test-mode surface MCP needs without
depending on host-level testing modules, so the code can move with the
standalone package boundary.
"""

from __future__ import annotations

import os
from typing import Any

_TRUTHY = {"1", "true", "yes", "y", "on"}
_ENV_HELPER_NONCRITICAL_EXCEPTIONS = (AttributeError, OSError, TypeError, ValueError)


def is_truthy(value: Any) -> bool:
    """Return True when a value uses one of the accepted truthy spellings."""
    try:
        return str(value or "").strip().lower() in _TRUTHY
    except _ENV_HELPER_NONCRITICAL_EXCEPTIONS:
        return False


def env_flag_enabled(name: str) -> bool:
    """Return True when the named environment variable is explicitly truthy."""
    try:
        return is_truthy(os.getenv(name))
    except _ENV_HELPER_NONCRITICAL_EXCEPTIONS:
        return False


def is_test_mode() -> bool:
    """Return True when MCP test mode is explicitly enabled by environment."""
    return env_flag_enabled("TEST_MODE") or env_flag_enabled("TLDW_TEST_MODE")


def is_explicit_pytest_runtime() -> bool:
    """Return True while pytest exposes its active test runtime signal."""
    try:
        return bool(str(os.getenv("PYTEST_CURRENT_TEST") or "").strip())
    except _ENV_HELPER_NONCRITICAL_EXCEPTIONS:
        return False


__all__ = [
    "env_flag_enabled",
    "is_explicit_pytest_runtime",
    "is_test_mode",
    "is_truthy",
]
