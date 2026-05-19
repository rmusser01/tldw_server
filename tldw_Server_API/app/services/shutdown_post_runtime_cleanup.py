"""
Post-runtime shutdown cleanup helpers extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


async def shutdown_post_runtime_cleanup(
    *,
    test_db_instance_ref: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Run the post-telemetry shutdown cleanup tail."""
    await _reset_media_db_cache(
        import_exceptions=import_exceptions,
    )
    await _shutdown_content_backend(
        guard_exceptions=startup_guard_exceptions,
    )
    await _close_managed_backend_registries(
        guard_exceptions=startup_guard_exceptions,
    )
    await _close_test_db_connections(
        test_db_instance_ref=test_db_instance_ref,
    )
    await _reset_jobs_acquire_gate(
        guard_exceptions=startup_guard_exceptions,
    )


async def _reset_media_db_cache(
    *,
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _reset_media_db_cache_service()
        logger.info("App Shutdown: Media DB cache cleared")
    except import_exceptions as exc:
        logger.debug("App Shutdown: Media DB cache cleanup skipped/failed: {}", exc)


async def _shutdown_content_backend(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _shutdown_content_backend_service()
        logger.info("App Shutdown: Content DB backend pool closed")
    except guard_exceptions as exc:
        logger.debug("App Shutdown: Content backend pool close skipped/failed: {}", exc)


async def _close_managed_backend_registries(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _close_all_backends_service()
        logger.info("App Shutdown: Database backend registries cleared")
    except guard_exceptions as exc:
        logger.debug("App Shutdown: Database backend registry cleanup skipped/failed: {}", exc)


async def _close_test_db_connections(
    *,
    test_db_instance_ref: Any,
) -> None:
    if test_db_instance_ref and hasattr(test_db_instance_ref, "close_all_connections"):
        logger.info("App Shutdown: Closing test DB connections")
        test_db_instance_ref.close_all_connections()
    else:
        logger.info("App Shutdown: No test DB instance found to close")


async def _reset_jobs_acquire_gate(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _set_jobs_acquire_gate_service(enabled=False)
    except guard_exceptions:
        pass


def _reset_media_db_cache_service() -> None:
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import reset_media_db_cache

    reset_media_db_cache()


def _shutdown_content_backend_service() -> None:
    from tldw_Server_API.app.core.DB_Management.DB_Manager import (
        shutdown_content_backend,
    )

    shutdown_content_backend()


def _close_all_backends_service() -> None:
    from tldw_Server_API.app.core.DB_Management.backends.factory import (
        close_all_backends,
    )

    close_all_backends()


def _set_jobs_acquire_gate_service(*, enabled: bool) -> None:
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    JobManager.set_acquire_gate(enabled)
