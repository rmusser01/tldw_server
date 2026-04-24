"""
Telemetry shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from loguru import logger


async def shutdown_telemetry_services(
    *,
    authnz_scheduler_started: bool,
    coordinated_legacy_component_names: set[str],
    import_exceptions: tuple[type[BaseException], ...],
) -> bool:
    """Stop shutdown-time telemetry dependencies and return the updated scheduler flag."""
    updated_authnz_scheduler_started = authnz_scheduler_started

    try:
        updated_authnz_scheduler_started = await _maybe_stop_authnz_scheduler_service(
            authnz_scheduler_started=authnz_scheduler_started,
            coordinated_legacy_component_names=coordinated_legacy_component_names,
            guard_exceptions=import_exceptions,
        )
        _shutdown_telemetry_service()
        logger.info("App Shutdown: Telemetry shutdown")
    except import_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down telemetry: {exc}")

    return updated_authnz_scheduler_started


async def _maybe_stop_authnz_scheduler_service(
    *,
    authnz_scheduler_started: bool,
    coordinated_legacy_component_names: set[str],
    guard_exceptions: tuple[type[BaseException], ...],
) -> bool:
    from tldw_Server_API.app.services.shutdown_authnz_scheduler import (
        maybe_stop_authnz_scheduler,
    )

    return await maybe_stop_authnz_scheduler(
        authnz_scheduler_started=authnz_scheduler_started,
        coordinated_legacy_component_names=coordinated_legacy_component_names,
        guard_exceptions=guard_exceptions,
    )


def _shutdown_telemetry_service() -> None:
    from tldw_Server_API.app.core.Metrics import shutdown_telemetry

    shutdown_telemetry()
