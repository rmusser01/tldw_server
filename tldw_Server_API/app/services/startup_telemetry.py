"""
Startup telemetry initialization extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


def initialize_startup_telemetry(
    *,
    app: Any,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> Any | None:
    logger.info("App Startup: Initializing telemetry and metrics...")

    try:
        telemetry_manager = _initialize_telemetry()
        if _otel_available():
            logger.info(
                f"App Startup: OpenTelemetry initialized for service: {telemetry_manager.config.service_name}"
            )
        else:
            logger.warning("App Startup: OpenTelemetry not available, using fallback metrics")

        try:
            if _instrument_fastapi_app(app, telemetry_manager):
                logger.info("App Startup: FastAPI instrumentation enabled")
        except startup_guard_exceptions as exc:
            logger.debug(f"App Startup: FastAPI instrumentation skipped: {exc}")

        return telemetry_manager
    except startup_guard_exceptions:
        logger.exception("App Startup: Failed to initialize telemetry")
        return None


def _initialize_telemetry() -> Any:
    from tldw_Server_API.app.core.Metrics import initialize_telemetry

    return initialize_telemetry()


def _otel_available() -> bool:
    from tldw_Server_API.app.core.Metrics import OTEL_AVAILABLE

    return bool(OTEL_AVAILABLE)


def _instrument_fastapi_app(app: Any, telemetry_manager: Any) -> bool:
    from tldw_Server_API.app.core.Metrics.telemetry import instrument_fastapi_app

    return bool(instrument_fastapi_app(app, telemetry_manager))
