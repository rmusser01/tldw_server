"""
AuthNZ scheduler shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger


async def maybe_stop_authnz_scheduler(
    *,
    authnz_scheduler_started: bool,
    coordinated_legacy_component_names: set[str],
    guard_exceptions: tuple[type[BaseException], ...],
    debug_log: Callable[[str], None] | None = None,
) -> bool:
    """Stop the AuthNZ scheduler when it was started and is not coordinator-owned."""
    if not authnz_scheduler_started or "authnz_scheduler" in coordinated_legacy_component_names:
        return authnz_scheduler_started

    if debug_log is None:
        debug_log = logger.debug

    try:
        await _stop_authnz_scheduler_service()
        logger.info("AuthNZ scheduler stopped")
        return False
    except guard_exceptions as exc:
        debug_log(f"AuthNZ scheduler shutdown skipped: {exc}")
        return authnz_scheduler_started


async def _stop_authnz_scheduler_service() -> None:
    from tldw_Server_API.app.core.AuthNZ.scheduler import stop_authnz_scheduler

    await stop_authnz_scheduler()
