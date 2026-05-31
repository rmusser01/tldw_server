"""
Cleanup-tail shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class CleanupTimedShutdownHandles:
    """Updated handles produced by the cleanup/timed-segment shutdown tail."""


async def shutdown_cleanup_timed_segments(
    *,
    app: Any,
    db_pool: Any | None,
    session_manager: Any | None,
    heavy_startup_handles: Any | None,
    in_pytest_for_db_pool_shutdown: bool,
    in_pytest_for_tts_shutdown: bool,
    import_exceptions: tuple[type[BaseException], ...],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    timed_shutdown_segment: Callable[[Any, str], AbstractContextManager[Any]],
) -> CleanupTimedShutdownHandles:
    """Run the late cleanup tail and timed shutdown segments in the legacy order."""
    logger.info("App Shutdown: Cleaning up resources...")
    logger.info("App Shutdown: Audit services cleanup handled by dependency injection")

    await _shutdown_auth_db_pool(
        db_pool=db_pool,
        in_pytest_for_db_pool_shutdown=in_pytest_for_db_pool_shutdown,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_resource_cleanup(
        app=app,
        session_manager=session_manager,
        heavy_startup_handles=heavy_startup_handles,
        in_pytest_for_tts_shutdown=in_pytest_for_tts_shutdown,
        import_exceptions=import_exceptions,
        startup_guard_exceptions=startup_guard_exceptions,
    )

    with timed_shutdown_segment(app, "evaluations_pool_shutdown"):
        await _shutdown_evaluations_resources(
            import_exceptions=import_exceptions,
        )

    with timed_shutdown_segment(app, "unified_audit_and_executor_shutdown"):
        await _shutdown_unified_audit_services(
            startup_guard_exceptions=startup_guard_exceptions,
            import_exceptions=import_exceptions,
        )
        await _shutdown_executor_resources(
            startup_guard_exceptions=startup_guard_exceptions,
            import_exceptions=import_exceptions,
        )
        await _shutdown_cpu_pools(
            guard_exceptions=startup_guard_exceptions,
        )

    with timed_shutdown_segment(app, "telemetry_shutdown"):
        await _shutdown_telemetry_services(
            import_exceptions=import_exceptions,
        )

    return CleanupTimedShutdownHandles()


async def _shutdown_auth_db_pool(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_auth_db_pool import shutdown_auth_db_pool

    await shutdown_auth_db_pool(**kwargs)


async def _shutdown_resource_cleanup(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_resource_cleanup import shutdown_resource_cleanup

    await shutdown_resource_cleanup(**kwargs)


async def _shutdown_evaluations_resources(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_evaluations_resources import (
        shutdown_evaluations_resources,
    )

    await shutdown_evaluations_resources(**kwargs)


async def _shutdown_unified_audit_services(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_unified_audit_services import (
        shutdown_unified_audit_services,
    )

    await shutdown_unified_audit_services(**kwargs)


async def _shutdown_executor_resources(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_executor_resources import (
        shutdown_executor_resources,
    )

    await shutdown_executor_resources(**kwargs)


async def _shutdown_cpu_pools(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_cpu_pools import shutdown_cpu_pools

    await shutdown_cpu_pools(**kwargs)


async def _shutdown_telemetry_services(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_telemetry_services import (
        shutdown_telemetry_services,
    )

    return await shutdown_telemetry_services(**kwargs)
