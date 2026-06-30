"""
Telemetry shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from loguru import logger


async def shutdown_telemetry_services(
    *,
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Stop shutdown-time telemetry dependencies."""
    try:
        _shutdown_telemetry_service()
        logger.info("App Shutdown: Telemetry shutdown")
    except import_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down telemetry: {exc}")


def _shutdown_telemetry_service() -> None:
    from tldw_Server_API.app.core.Metrics import shutdown_telemetry

    shutdown_telemetry()
