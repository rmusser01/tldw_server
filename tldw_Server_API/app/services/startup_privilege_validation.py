"""
Startup privilege validation extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


def validate_startup_privilege_metadata(
    *,
    app: Any,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> Any:
    try:
        return _validate_privilege_metadata_on_startup(app)
    except startup_guard_exceptions as exc:
        logger.exception(f"App Startup: Privilege metadata validation failed: {exc}")
        raise


def _validate_privilege_metadata_on_startup(app: Any) -> Any:
    from tldw_Server_API.app.core.PrivilegeMaps.startup import (
        validate_privilege_metadata_on_startup,
    )

    return validate_privilege_metadata_on_startup(app)
