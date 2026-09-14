"""
Startup Sentry initialization extracted from the application lifespan.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from tldw_Server_API.app.core.Security.standalone_html_request_guard import (
    is_standalone_sensitive_route,
)


def redact_standalone_sentry_event(
    event: dict[str, Any],
    hint: dict[str, Any],
) -> dict[str, Any]:
    """Replace source-bearing standalone route events with fixed diagnostics."""
    del hint
    request = event.get("request")
    if not isinstance(request, dict):
        return event
    method = str(request.get("method", ""))
    path = urlsplit(str(request.get("url", ""))).path
    if not is_standalone_sensitive_route(method, path):
        return event

    redacted: dict[str, Any] = {
        "message": "Standalone Slides source event redacted",
        "request": {
            "method": method.upper(),
            "url": "standalone_slides_route",
        },
        "tags": {"standalone_html_sensitive": "true"},
    }
    for key in ("event_id", "timestamp", "level", "platform"):
        if key in event:
            redacted[key] = event[key]
    if event.get("type") == "transaction":
        redacted["type"] = "transaction"
        redacted["transaction"] = "standalone_slides_route"
        for key in ("start_timestamp", "timestamp"):
            if key in event:
                redacted[key] = event[key]
    return redacted


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
            max_request_body_size="never",
            before_send=redact_standalone_sentry_event,
            before_send_transaction=redact_standalone_sentry_event,
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
