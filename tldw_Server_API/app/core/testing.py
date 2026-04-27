"""
Lightweight helpers for test-mode detection and truthy env parsing.

Kept dependency-free (stdlib only) to avoid import-time side effects.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "y", "on"}
_PRODUCTION_VALUES = {"production", "prod", "live"}
_PRODUCTION_ENV_KEYS = (
    "ENVIRONMENT",
    "APP_ENV",
    "DEPLOYMENT_ENV",
    "FASTAPI_ENV",
    "TLDW_ENV",
)
_TEST_FLAG_KEYS = ("TEST_MODE", "TESTING", "TLDW_TEST_MODE")


def _env_truthy(val: str | None) -> bool:
    try:
        return str(val or "").strip().lower() in _TRUTHY
    except Exception:
        return False


def is_truthy(value: str | None) -> bool:
    """Public truthy parser used for consistent environment-flag semantics."""
    return _env_truthy(value)


def env_flag_enabled(name: str) -> bool:
    """Return True when the named environment variable is truthy."""
    try:
        return _env_truthy(os.getenv(name))
    except Exception:
        return False


def is_test_mode() -> bool:
    """Return True when server-side test mode is enabled.

    Checks both TEST_MODE and TLDW_TEST_MODE, using a consistent truthy set.
    Never reads client data; only server environment variables.
    """
    return env_flag_enabled("TEST_MODE") or env_flag_enabled("TLDW_TEST_MODE")


def is_explicit_pytest_runtime() -> bool:
    """Return True only when running under explicit pytest runtime context.

    The runtime signal is `PYTEST_CURRENT_TEST`, which pytest sets for active
    test execution phases. This intentionally avoids looser heuristics such as
    checking `sys.modules`.
    """
    try:
        return bool(str(os.getenv("PYTEST_CURRENT_TEST") or "").strip())
    except Exception:
        return False


def audio_imports_enabled_for_runtime() -> bool:
    """Allow heavy audio router imports outside pytest or with explicit opt-in."""
    return not is_explicit_pytest_runtime() or env_flag_enabled("MINIMAL_TEST_INCLUDE_AUDIO")


def is_production_like_env() -> bool:
    """Detect production-like runtime from common deployment environment variables."""
    try:
        if _env_truthy(os.getenv("tldw_production")):
            return True
    except Exception:
        return False

    for key in _PRODUCTION_ENV_KEYS:
        try:
            value = str(os.getenv(key, "")).strip().lower()
        except Exception:
            value = ""
        if value in _PRODUCTION_VALUES:
            return True
    return False


def _active_test_mode_flags() -> list[str]:
    active: list[str] = []
    for key in _TEST_FLAG_KEYS:
        if env_flag_enabled(key):
            active.append(key)
    return active


def validate_test_runtime_flags() -> None:
    """Fail fast when test-mode flags are enabled outside explicit pytest runtime.

    Guarded flags:
    - TEST_MODE
    - TESTING
    - TLDW_TEST_MODE
    """
    active_flags = _active_test_mode_flags()
    if not active_flags:
        return
    if is_explicit_pytest_runtime():
        return

    enabled = ", ".join(active_flags)
    raise RuntimeError(
        "Unsafe startup configuration: test-mode flags are enabled outside explicit pytest "
        f"runtime (missing PYTEST_CURRENT_TEST): {enabled}. "
        "Unset test flags or run under pytest."
    )
