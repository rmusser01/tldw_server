from __future__ import annotations

from time import monotonic
from typing import Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.core.Logging.log_context import ensure_request_id


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Minimal structured access logging for each HTTP request.

    Emits a single info-level log line per request with:
    - request_id (propagated or synthesized)
    - method
    - host
    - path
    - status
    - duration_ms
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = monotonic()
        req_id = ensure_request_id(request)
        method = request.method
        try:
            host = request.headers.get("host") or request.url.hostname or ""
        except Exception:
            host = ""
        path = request.url.path

        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration_ms = int((monotonic() - start) * 1000)
            try:
                log = logger.bind(
                    request_id=req_id,
                    method=method,
                    host=host,
                    path=path,
                    status=status,
                    duration_ms=duration_ms,
                )
                level = "WARNING" if status >= 500 else "INFO"
                log.log(level, f"HTTP {method} {path} -> {status} in {duration_ms}ms")
            except Exception:
                # Never fail a request due to logging
                logger.debug("Access log middleware failed to emit request log")
