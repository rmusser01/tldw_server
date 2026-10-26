"""
Lightweight helpers for test-mode detection and truthy env parsing.

Kept dependency-free (stdlib only) to avoid import-time side effects.
The boolean vocabulary lives in ``core/Utils/coercion.py`` and is re-exported
here for the existing importers of ``is_truthy``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.Utils.coercion import env_bool, parse_bool  # noqa: F401  (re-export)

# The one production detector (TASK-13950.6). Reading these keys anywhere else in
# app/ fails tests/lint/test_production_detector_single_source.py.
PRODUCTION_ENV_KEYS = (
    "tldw_production",
    "TLDW_PRODUCTION",
    "ENVIRONMENT",
    "APP_ENV",
    "DEPLOYMENT_ENV",
    "FASTAPI_ENV",
    "TLDW_ENV",
    "ENV",
)
_PRODUCTION_FLAG_KEYS = frozenset({"tldw_production", "TLDW_PRODUCTION"})  # also accept truthy values
_PRODUCTION_VALUES = frozenset({"prod", "production", "live"})
_TEST_FLAG_KEYS = ("TEST_MODE", "TESTING", "TLDW_TEST_MODE")
# Test seams refused at startup in a production-like runtime, PYTEST_CURRENT_TEST or
# not. tests/lint/test_test_seam_env_reads_guarded.py requires every other *TEST* env
# read in app/ to be allowlisted with a reason.
GUARDED_TEST_SEAM_KEYS = _TEST_FLAG_KEYS + (
    "TLDW_ALLOW_TEST_FALLBACK_KEYS",
    "SINGLE_USER_TEST_API_KEY",
    "E2E_TEST_BASE_URL",
    "MINIMAL_TEST_APP",
    "JWT_SECRET_TEST_KEY",  # signing-secret fallback (settings, crypto_utils, admin_data_ops_service)
    "ENABLE_ADMIN_E2E_TEST_MODE",  # mounts the AuthNZ seed/reset test-support router
    "EVALUATIONS_TEST_DB_PATH",  # one evaluations DB for every user
    "JOBS_TEST_NOW_EPOCH",  # freezes the Jobs lease clock
)
GUARDED_TEST_SEAM_PREFIXES = ("RG_TEST_",)


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Whether the runtime is production-like, and which key said so."""

    is_production: bool
    source_key: str | None = None


def _env_truthy(val: str | None) -> bool:
    # Two-way on the string form: unrecognised is False. str() keeps the
    # historical behaviour for non-string inputs (2 and 1.0 are not truthy).
    try:
        return parse_bool(str(val or ""), default=False)
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


def runtime_environment(environ: Mapping[str, str] | None = None) -> RuntimeEnvironment:
    """Classify the runtime from PRODUCTION_ENV_KEYS (default: ``os.environ``).

    Production-like when any key, case-insensitively, is ``prod``, ``production`` or
    ``live``; ``tldw_production`` and ``TLDW_PRODUCTION`` also accept a truthy value.
    Uncached, so a changed environment is seen on the next call.
    """
    env = os.environ if environ is None else environ
    for key in PRODUCTION_ENV_KEYS:
        value = str(env.get(key) or "").strip().lower()
        if value in _PRODUCTION_VALUES or (key in _PRODUCTION_FLAG_KEYS and _env_truthy(value)):
            return RuntimeEnvironment(is_production=True, source_key=key)
    return RuntimeEnvironment(is_production=False)


def is_production_like_env() -> bool:
    """Return True when :func:`runtime_environment` reports a production-like runtime."""
    return runtime_environment().is_production


def _active_test_mode_flags() -> list[str]:
    active: list[str] = []
    for key in _TEST_FLAG_KEYS:
        if env_flag_enabled(key):
            active.append(key)
    return active


def armed_test_seams(environ: Mapping[str, str] | None = None) -> list[str]:
    """Guarded test seams that are set and not explicitly switched off (fail closed)."""
    env = os.environ if environ is None else environ
    return sorted(
        key
        for key, value in env.items()
        if (key in GUARDED_TEST_SEAM_KEYS or key.startswith(GUARDED_TEST_SEAM_PREFIXES))
        and parse_bool(value, default=True)
    )


def validate_test_runtime_flags() -> None:
    """Fail fast on test seams the runtime must not honour.

    - In a production-like runtime (:func:`runtime_environment`), any armed seam in
      GUARDED_TEST_SEAM_KEYS / GUARDED_TEST_SEAM_PREFIXES raises, even when
      PYTEST_CURRENT_TEST is set.
    - Elsewhere, TEST_MODE, TESTING and TLDW_TEST_MODE still require an explicit
      pytest runtime; the other seams stay usable by e2e servers started outside pytest.
    """
    runtime = runtime_environment()
    if runtime.is_production:
        armed = armed_test_seams()
        if armed:
            raise RuntimeError(
                "Unsafe startup configuration: test seams are armed in a production-like runtime "
                f"({runtime.source_key}): {', '.join(armed)}. PYTEST_CURRENT_TEST does not exempt "
                "production; unset the test seams or the production marker."
            )

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
