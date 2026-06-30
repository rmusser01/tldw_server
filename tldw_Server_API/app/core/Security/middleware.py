from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import is_truthy

DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "media-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "upgrade-insecure-requests"
)

# Permissive CSP for Setup UI fallback (only if SetupCSPMiddleware didn't set one)
RELAXED_CSP_SETUP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob: https:; "
    "font-src 'self' data:; "
    "media-src 'self' data: blob:; "
    "connect-src 'self' http: https: ws: wss:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "upgrade-insecure-requests"
)

# Relaxed CSP for API Docs (/docs, /redoc). Allows inline scripts and HTTPS CDN fallback if
# local assets are unavailable, while keeping other directives reasonably strict.
RELAXED_CSP_DOCS = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; "
    "style-src 'self' 'unsafe-inline' https:; "
    "img-src 'self' data: https:; "
    "font-src 'self' data: https:; "
    "media-src 'self' data: blob:; "
    # Allow fetching the OpenAPI schema using absolute URLs during dev
    "connect-src 'self' http: https:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "upgrade-insecure-requests"
)

DEFAULT_PERMISSIONS_POLICY = (
    "geolocation=(), "
    "microphone=(), "
    "camera=(), "
    "payment=(), "
    "usb=(), "
    "magnetometer=(), "
    "gyroscope=(), "
    "accelerometer=()"
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return is_truthy(raw)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply hardened security headers to every HTTP response."""

    def __init__(
        self,
        app,
        *,
        enabled: bool = True,
        strict_transport_security: bool | None = None,
        content_type_options: bool = True,
        frame_options: str | None = "DENY",
        xss_protection: bool = False,
        content_security_policy: str | None = None,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: str | None = None,
        custom_headers: Mapping[str, str] | None = None,
        remove_server_header: bool = True,
        ) -> None:
        super().__init__(app)
        self.enabled = enabled
        if strict_transport_security is None:
            strict_transport_security = _env_flag("SECURITY_ENABLE_HSTS", False)
        self.strict_transport_security = strict_transport_security
        self.content_type_options = content_type_options
        self.frame_options = frame_options
        self.xss_protection = xss_protection
        self.content_security_policy = content_security_policy
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy
        self.custom_headers = dict(custom_headers or {})
        self.remove_server_header = remove_server_header
        self.registry = get_metrics_registry()

    @staticmethod
    def _is_https_request(request: Request) -> bool:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        if forwarded_proto:
            return forwarded_proto == "https"
        return request.url.scheme == "https"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response: Response = await call_next(request)

        if not self.enabled:
            return response

        # Remove potentially sensitive headers
        if self.remove_server_header and "Server" in response.headers:
            del response.headers["Server"]

        response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")

        if self.content_type_options:
            response.headers.setdefault("X-Content-Type-Options", "nosniff")

        if self.frame_options:
            response.headers.setdefault("X-Frame-Options", self.frame_options)

        if self.xss_protection:
            response.headers.setdefault("X-XSS-Protection", "1; mode=block")

        if self.referrer_policy:
            response.headers.setdefault("Referrer-Policy", self.referrer_policy)

        # Path-scoped CSP: SetupCSPMiddleware is authoritative for /setup.
        # Only provide a fallback CSP here if none has been set.
        path = request.url.path or ""
        if path.startswith("/setup"):
            if "Content-Security-Policy" not in response.headers:
                response.headers.setdefault("Content-Security-Policy", RELAXED_CSP_SETUP)
        elif path.startswith("/docs") or path.startswith("/redoc"):
            # Docs UI often uses inline scripts; allow inline/eval and optional HTTPS CDNs
            if "Content-Security-Policy" not in response.headers:
                response.headers.setdefault("Content-Security-Policy", RELAXED_CSP_DOCS)
        else:
            csp_value = self.content_security_policy or DEFAULT_CSP
            response.headers.setdefault("Content-Security-Policy", csp_value)

        permissions_value = self.permissions_policy or DEFAULT_PERMISSIONS_POLICY
        response.headers.setdefault("Permissions-Policy", permissions_value)

        if self.strict_transport_security and self._is_https_request(request):
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains; preload",
            )

        for header_name, header_value in self.custom_headers.items():
            response.headers.setdefault(header_name, header_value)

        try:
            self.registry.increment("security_headers_responses_total", 1)
        except Exception:  # pragma: no cover - metrics failures should not impact request
            logger.debug("Security headers metric increment failed")

        return response


def create_security_headers_middleware(app, development_mode: bool = False) -> SecurityHeadersMiddleware:
    """Factory for convenience so existing imports continue to work."""
    if development_mode:
        return SecurityHeadersMiddleware(
            app,
            strict_transport_security=False,
            frame_options="SAMEORIGIN",
            content_security_policy=None,
            permissions_policy=None,
        )

    return SecurityHeadersMiddleware(app)


__all__ = ["SecurityHeadersMiddleware", "create_security_headers_middleware"]
