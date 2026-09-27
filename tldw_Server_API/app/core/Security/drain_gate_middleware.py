from __future__ import annotations

import re
from collections.abc import Callable

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from tldw_Server_API.app.services.app_lifecycle import is_lifecycle_draining

CORS_EXPOSE_HEADERS: tuple[str, ...] = (
    "Content-Disposition",
    "ETag",
    "Last-Modified",
    "Retry-After",
    "Content-Length",
    "X-Request-ID",
    "traceparent",
    "X-Trace-Id",
    "X-TLDW-TTS-Backend",
    "X-TLDW-TTS-Fallback-Used",
)
CORS_EXPOSE_HEADERS_VALUE = ", ".join(CORS_EXPOSE_HEADERS)

CONTROL_PLANE_DRAIN_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/internal/ready",
        "/ready",
        "/readyz",
        "/health/ready",
        "/healthz",
        "/api/v1/health",
        "/api/v1/healthz",
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/api/v1/readyz",
    }
)
CONTROL_PLANE_DRAIN_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {(method, path) for method in ("GET", "HEAD") for path in CONTROL_PLANE_DRAIN_PATHS}
)


def _is_allowlisted_control_plane_path(request: Request) -> bool:
    method = request.method.upper()
    path = request.url.path
    return (method, path) in CONTROL_PLANE_DRAIN_ALLOWLIST


def _resolve_drain_gate_allow_origin(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if not origin:
        return None

    config = getattr(getattr(request.app, "state", None), "_tldw_drain_gate_cors_config", None)
    if not isinstance(config, dict):
        return None

    normalized_origin = origin.rstrip("/")
    if config.get("allow_all_origins"):
        return origin if config.get("allow_credentials") else "*"

    allowed_origins = config.get("allowed_origins", set())
    if normalized_origin in allowed_origins:
        return origin

    allow_origin_regex = config.get("allow_origin_regex")
    if allow_origin_regex:
        try:
            if re.fullmatch(str(allow_origin_regex), origin):
                return origin
        except re.error:
            return None
    return None


def merge_vary_origin(headers: MutableHeaders) -> None:
    """Add ``Origin`` to a response's Vary values without dropping existing values."""
    vary = headers.get("Vary", "")
    values = [value.strip() for value in vary.split(",") if value.strip()]
    if not any(value.casefold() == "origin" for value in values):
        values.append("Origin")
        headers["Vary"] = ", ".join(values)


def _apply_drain_gate_cors_headers(request: Request, response: Response) -> Response:
    allow_origin = _resolve_drain_gate_allow_origin(request)
    if not allow_origin:
        return response

    config = getattr(getattr(request.app, "state", None), "_tldw_drain_gate_cors_config", {})
    response.headers.setdefault("Access-Control-Allow-Origin", allow_origin)
    if allow_origin != "*":
        merge_vary_origin(response.headers)
    if config.get("allow_credentials"):
        response.headers.setdefault("Access-Control-Allow-Credentials", "true")

    requested_method = request.headers.get("access-control-request-method")
    requested_headers = request.headers.get("access-control-request-headers")
    response.headers.setdefault("Access-Control-Allow-Methods", requested_method or "*")
    response.headers.setdefault("Access-Control-Allow-Headers", requested_headers or "*")
    response.headers.setdefault(
        "Access-Control-Expose-Headers",
        config.get("expose_headers", CORS_EXPOSE_HEADERS_VALUE),
    )
    return response


class DrainGateMiddleware(BaseHTTPMiddleware):
    """Reject non-control-plane requests while shutdown is draining."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if is_lifecycle_draining(request.app) and not _is_allowlisted_control_plane_path(request):
            response = JSONResponse(
                {"status": "not_ready", "reason": "shutdown_in_progress"},
                status_code=503,
            )
            return _apply_drain_gate_cors_headers(request, response)
        return await call_next(request)


__all__ = [
    "CORS_EXPOSE_HEADERS",
    "CORS_EXPOSE_HEADERS_VALUE",
    "CONTROL_PLANE_DRAIN_ALLOWLIST",
    "CONTROL_PLANE_DRAIN_PATHS",
    "DrainGateMiddleware",
    "_apply_drain_gate_cors_headers",
    "_is_allowlisted_control_plane_path",
    "merge_vary_origin",
]
