"""
App package initializer.

Installs Starlette status-code compatibility aliases. The test-time
parallelism defaults (joblib/loky/BLAS threads) live in the root conftest.py.
"""

from __future__ import annotations

try:
    from starlette import status as _starlette_status
except Exception:  # pragma: no cover - fallback for minimal tooling environments
    _starlette_status = None

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
