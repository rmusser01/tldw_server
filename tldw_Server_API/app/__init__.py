"""
App package initializer.

Sets conservative parallelism defaults when running under tests to avoid
spawning subprocess pools (e.g., joblib/loky) that can leak semaphores in
constrained CI or sandboxed environments. This reduces flakiness in pytest
and keeps resource usage predictable.

These settings only apply in test environments (detected via TESTING env var
or the presence of PYTEST_CURRENT_TEST). They are no-ops for normal runtime.
"""

from __future__ import annotations

import os
import sys

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal tooling environments
    import logging

    logger = logging.getLogger(__name__)
try:
    from starlette import status as _starlette_status
except Exception:  # pragma: no cover - fallback for minimal tooling environments
    _starlette_status = None
from tldw_Server_API.app.core.testing import is_truthy

# Compatibility aliases across Starlette versions.
if _starlette_status is not None and not hasattr(_starlette_status, "HTTP_413_CONTENT_TOO_LARGE"):
    setattr(
        _starlette_status,
        "HTTP_413_CONTENT_TOO_LARGE",
        getattr(_starlette_status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413),
    )

if _starlette_status is not None and not hasattr(_starlette_status, "HTTP_422_UNPROCESSABLE_CONTENT"):
    setattr(
        _starlette_status,
        "HTTP_422_UNPROCESSABLE_CONTENT",
        getattr(_starlette_status, "HTTP_422_UNPROCESSABLE_ENTITY", 422),
    )


def _under_pytest() -> bool:
    try:
        if "PYTEST_CURRENT_TEST" in os.environ:
            return True
        # Fallback heuristic if PYTEST_CURRENT_TEST isn't set yet
        return any("pytest" in (arg or "") for arg in sys.argv)
    except Exception:
        logger.debug("app.__init__._under_pytest check failed")
        return False


def _env_flag_true(name: str) -> bool:
    return is_truthy(os.getenv(name))


if _env_flag_true("TESTING") or _under_pytest():
    # Prevent joblib from spawning loky process pools in tests
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("JOBLIB_BACKEND", "threading")
    # Constrain CPU-bound libraries to a single thread for determinism
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Reduce tokenizers parallel contention and warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
