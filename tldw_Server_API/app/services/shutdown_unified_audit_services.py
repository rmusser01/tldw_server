"""
Unified audit shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from loguru import logger


async def shutdown_unified_audit_services(
    *,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Shut down unified audit services and their local adapter loops."""
    try:
        logger.info("App Shutdown: Shutting down unified audit services...")
        await _shutdown_cached_audit_services()
        logger.info("App Shutdown: Unified audit services stopped")
        await _shutdown_sharing_audit_service(
            guard_exceptions=startup_guard_exceptions + (ImportError, ModuleNotFoundError),
        )
        await _shutdown_embeddings_audit_adapter(
            guard_exceptions=startup_guard_exceptions + import_exceptions,
        )
        await _shutdown_evaluations_audit_adapter(
            guard_exceptions=startup_guard_exceptions + import_exceptions,
        )
    except import_exceptions as exc:
        logger.exception(f"App Shutdown: Error stopping unified audit services: {exc}")


async def _shutdown_cached_audit_services() -> None:
    await _shutdown_cached_audit_services_service()


async def _shutdown_sharing_audit_service(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        await _shutdown_sharing_audit_service_service()
        logger.info("App Shutdown: Sharing audit service stopped")
    except guard_exceptions as exc:
        logger.debug(f"Sharing audit service shutdown skipped: {exc}")


async def _shutdown_embeddings_audit_adapter(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _shutdown_embeddings_audit_adapter_service()
        logger.info("App Shutdown: Embeddings audit adapter loop stopped")
    except guard_exceptions as exc:
        logger.debug("Embeddings audit adapter loop shutdown skipped: {}", exc)


async def _shutdown_evaluations_audit_adapter(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _shutdown_evaluations_audit_adapter_service()
        logger.info("App Shutdown: Evaluations audit adapter loop stopped")
    except guard_exceptions as exc:
        logger.debug("Evaluations audit adapter loop shutdown skipped: {}", exc)


async def _shutdown_cached_audit_services_service() -> None:
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
        shutdown_all_audit_services,
    )

    await shutdown_all_audit_services()


async def _shutdown_sharing_audit_service_service() -> None:
    from tldw_Server_API.app.api.v1.endpoints.sharing import (
        shutdown_sharing_audit_service,
    )

    await shutdown_sharing_audit_service()


def _shutdown_embeddings_audit_adapter_service() -> None:
    from tldw_Server_API.app.core.Embeddings.audit_adapter import (
        shutdown_local_audit_adapter_loop,
    )

    shutdown_local_audit_adapter_loop()


def _shutdown_evaluations_audit_adapter_service() -> None:
    from tldw_Server_API.app.core.Evaluations.audit_adapter import (
        shutdown_local_evaluations_audit_loop,
    )

    shutdown_local_evaluations_audit_loop()
