"""
Startup preflight execution and summary logging extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


async def run_startup_preflight_checks(
    *,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        preflight = await _run_preflight_checks_in_thread()
        logger.info(
            "Preflight: {} checks, {} warnings, {} failures",
            len(preflight.checks),
            len(preflight.warnings),
            len(preflight.failures),
        )
    except RuntimeError:
        raise
    except startup_guard_exceptions as exc:
        logger.debug("Preflight checks skipped: {}", exc)


async def _run_preflight_checks_in_thread() -> Any:
    import asyncio

    from tldw_Server_API.app.core.startup_preflight import run_preflight_checks

    return await asyncio.to_thread(run_preflight_checks)
