"""
Evaluations shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from loguru import logger


async def shutdown_evaluations_resources(
    *,
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Shut down lazily initialized evaluations resources."""
    await _shutdown_evaluations_pool(
        import_exceptions=import_exceptions,
    )
    await _shutdown_evaluations_webhook_manager(
        import_exceptions=import_exceptions,
    )


async def _shutdown_evaluations_pool(
    *,
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _shutdown_evaluations_pool_service()
        logger.info("App Shutdown: Evaluations connection manager shutdown (lazy)")
    except import_exceptions as exc:
        logger.debug(f"App Shutdown: Evaluations pool shutdown skipped/failed: {exc}")


async def _shutdown_evaluations_webhook_manager(
    *,
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _shutdown_evaluations_webhook_manager_service()
        logger.info("App Shutdown: Evaluations webhook manager shutdown (lazy)")
    except import_exceptions as exc:
        logger.debug(f"App Shutdown: Evaluations webhook manager shutdown skipped/failed: {exc}")


def _shutdown_evaluations_pool_service() -> None:
    from tldw_Server_API.app.core.Evaluations.connection_pool import (
        shutdown_evaluations_pool_if_initialized,
    )

    shutdown_evaluations_pool_if_initialized()


def _shutdown_evaluations_webhook_manager_service() -> None:
    from tldw_Server_API.app.core.Evaluations.webhook_manager import (
        shutdown_webhook_manager_if_initialized,
    )

    shutdown_webhook_manager_if_initialized()
