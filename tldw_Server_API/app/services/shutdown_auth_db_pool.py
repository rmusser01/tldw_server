"""
Auth DB pool shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


async def shutdown_auth_db_pool(
    *,
    db_pool: Any | None,
    in_pytest_for_db_pool_shutdown: bool,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Close the auth DB pool unless shutdown is running in pytest."""
    try:
        if db_pool is None:
            return
        if in_pytest_for_db_pool_shutdown:
            logger.info("App Shutdown: Skipping DB pool close in test context")
            return

        await db_pool.close()
        logger.info("App Shutdown: Auth database pool closed")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error closing auth database pool: {exc}")
