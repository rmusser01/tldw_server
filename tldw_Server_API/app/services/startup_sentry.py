"""
Startup Sentry initialization extracted from the application lifespan.
"""

from __future__ import annotations

import os
from typing import Any


def initialize_startup_sentry(
    *,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    sentry_dsn = _getenv("SENTRY_DSN", "")
    if not sentry_dsn:
        return

    try:
        _init_sentry(
            dsn=sentry_dsn,
            traces_sample_rate=float(_getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=_getenv("DEPLOYMENT_ENV", "development"),
            release=_getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            send_default_pii=False,
        )
        logger.info("App Startup: Sentry error tracking initialized")
    except startup_guard_exceptions + import_exceptions as exc:
        logger.warning("App Startup: Sentry initialization failed")
        logger.debug(
            "App Startup: Sentry initialization failure type={}",
            type(exc).__name__,
        )


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _init_sentry(**kwargs: Any) -> None:
    import sentry_sdk

    sentry_sdk.init(**kwargs)
