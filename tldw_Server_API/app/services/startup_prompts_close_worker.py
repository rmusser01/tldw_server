"""
Prompts DB close-worker startup extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


def start_prompts_close_worker(
    *,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _start_prompts_pending_close_worker()
    except startup_guard_exceptions as exc:
        logger.debug(f"App Startup: Prompts close worker startup skipped/failed: {exc}")


def _start_prompts_pending_close_worker() -> None:
    from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import (
        start_prompts_pending_close_worker,
    )

    start_prompts_pending_close_worker()
