"""
Post-worker non-worker cleanup helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class PostWorkerNonWorkerCleanupHandles:
    """Updated handles produced by post-worker non-worker cleanup."""


async def run_shutdown_post_worker_non_worker_cleanup(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> PostWorkerNonWorkerCleanupHandles:
    """Run post-worker cleanup that is not owned by lifecycle worker handles."""
    try:
        await _shutdown_personalization_consolidation(
            guard_exceptions=guard_exceptions,
        )
    except guard_exceptions as exc:
        logger.debug(f"Post-worker non-worker cleanup skipped: {exc}")
    return PostWorkerNonWorkerCleanupHandles()


async def _shutdown_personalization_consolidation(**kwargs):
    from tldw_Server_API.app.services.shutdown_personalization_consolidation import (
        shutdown_personalization_consolidation,
    )

    await shutdown_personalization_consolidation(**kwargs)
