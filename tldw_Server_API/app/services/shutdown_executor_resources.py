"""
Executor shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio

from loguru import logger


async def shutdown_executor_resources(
    *,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Shut down registered executors and the event loop default executor."""
    try:
        await _shutdown_registered_executors_service(
            wait=True,
            cancel_futures=True,
        )
        logger.info("App Shutdown: Registered executors shutdown")
    except import_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down executors: {exc}")

    await _shutdown_default_executor(
        guard_exceptions=startup_guard_exceptions,
    )


async def _shutdown_default_executor(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        loop = asyncio.get_running_loop()
        if hasattr(loop, "shutdown_default_executor"):
            await loop.shutdown_default_executor()
            logger.info("App Shutdown: Default executor shutdown")
    except guard_exceptions as exc:
        logger.debug(f"App Shutdown: Default executor shutdown skipped/failed: {exc}")


async def _shutdown_registered_executors_service(
    *,
    wait: bool,
    cancel_futures: bool,
) -> None:
    from tldw_Server_API.app.core.Utils.executor_registry import (
        shutdown_all_registered_executors,
    )

    await shutdown_all_registered_executors(
        wait=wait,
        cancel_futures=cancel_futures,
    )
