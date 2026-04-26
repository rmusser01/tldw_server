"""
Personalization consolidation shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


async def shutdown_personalization_consolidation(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Stop the personalization consolidation service with legacy guard semantics."""
    try:
        service = _get_consolidation_service()
        await service.stop()
        logger.info("Personalization consolidation service stopped")
    except guard_exceptions as exc:
        logger.warning(f"Personalization consolidation shutdown failed: {exc}")


def _get_consolidation_service() -> Any:
    from tldw_Server_API.app.services.personalization_consolidation import get_consolidation_service

    return get_consolidation_service()
