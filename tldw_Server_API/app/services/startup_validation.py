"""
Startup validation helpers extracted from the application lifespan startup.
"""

from __future__ import annotations

import os

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.startup_integrity import (
    verify_authnz_sqlite_startup_integrity,
)
from tldw_Server_API.app.core.Setup.setup_manager import needs_setup
from tldw_Server_API.app.core.testing import is_truthy as _shared_is_truthy

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def run_startup_validations() -> None:
    """Run pre-auth startup checks that should remain ahead of service init."""
    try:
        if needs_setup():
            logger.warning(
                "First-time setup is enabled. The setup API is local-only and blocks proxied requests. "
                "If running behind a reverse proxy, ensure /setup and /api/v1/setup are not publicly exposed, or "
                "set TLDW_SETUP_ALLOW_REMOTE=1 temporarily on trusted networks."
            )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Setup status check failed during startup: {exc}")

    try:
        auth_settings = get_settings()
        allow_corrupt_startup = _shared_is_truthy(
            os.getenv("TLDW_ALLOW_CORRUPT_AUTHNZ_STARTUP")
        )
        await verify_authnz_sqlite_startup_integrity(
            database_url=str(getattr(auth_settings, "DATABASE_URL", "")),
            auth_mode=str(getattr(auth_settings, "AUTH_MODE", "single_user")),
            dispatch_alerts=True,
            fail_on_error=not allow_corrupt_startup,
        )
        if allow_corrupt_startup:
            logger.warning(
                "App Startup: Corrupt AuthNZ DB fail-open mode enabled via "
                "TLDW_ALLOW_CORRUPT_AUTHNZ_STARTUP=true"
            )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.exception(
            f"App Startup: AuthNZ SQLite integrity preflight failed: {exc}"
        )
        raise
