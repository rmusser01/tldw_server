"""
Startup test-runtime guard extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


def validate_startup_test_runtime(
    *,
    logger: Any,
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _validate_test_runtime_flags()
    except RuntimeError as exc:
        logger.critical(f"Startup aborted due to unsafe test-mode flags: {exc}")
        raise
    except import_exceptions as exc:
        logger.debug(f"Test-mode runtime guard import skipped: {exc}")


def _validate_test_runtime_flags() -> None:
    from tldw_Server_API.app.core.testing import validate_test_runtime_flags

    validate_test_runtime_flags()
