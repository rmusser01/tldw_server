"""
CPU pool shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from loguru import logger


async def shutdown_cpu_pools(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Clean up CPU-bound worker pools."""
    try:
        _cleanup_cpu_pools_service()
        logger.info("App Shutdown: CPU pools cleaned up")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error cleaning up CPU pools: {exc}")


def _cleanup_cpu_pools_service() -> None:
    from tldw_Server_API.app.core.Utils.cpu_bound_handler import cleanup_pools

    cleanup_pools()
