from __future__ import annotations

import os
from typing import Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.core.testing import is_truthy


def _build_setup_csp(*, allow_inline_scripts: bool, allow_eval: bool) -> str:
    """Build CSP for Setup UI.

    - /setup keeps inline scripts allowed since the setup flow relies on helpers.
    - Set env TLDW_SETUP_NO_EVAL=1 to drop 'unsafe-eval' if desired.
    """
    script_parts = ["'self'"]
    if allow_inline_scripts:
        script_parts.append("'unsafe-inline'")
    if allow_eval:
        script_parts.append("'unsafe-eval'")
    policy = (
        "default-src 'self'; "
        + f"script-src {' '.join(script_parts)}; "
        + "style-src 'self' 'unsafe-inline'; "
        + "img-src 'self' data: blob: https:; "
        + "font-src 'self' data:; "
        + "media-src 'self' data: blob:; "
        + "connect-src 'self' http: https: ws: wss:; "
        + "frame-ancestors 'none'; "
        + "base-uri 'self'; "
        + "form-action 'self'; "
        + "upgrade-insecure-requests"
    )
    return policy


class SetupCSPMiddleware(BaseHTTPMiddleware):
    """Add CSP headers for /setup without rewriting response bodies.

    We keep a relaxed CSP suitable for the setup flow by allowing inline scripts.
    Eval can be disabled via TLDW_SETUP_NO_EVAL.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path or ""
        if not path.startswith("/setup"):
            return await call_next(request)

        response = await call_next(request)
        try:
            allow_inline_scripts = True

            # Eval policy precedence:
            # - Respect TLDW_SETUP_NO_EVAL first if present.
            no_eval_env = os.getenv("TLDW_SETUP_NO_EVAL")
            if no_eval_env is not None:
                truthy = is_truthy(no_eval_env)
                allow_eval = not truthy
            else:
                allow_eval = True
            response.headers.setdefault(
                "Content-Security-Policy",
                _build_setup_csp(allow_inline_scripts=allow_inline_scripts, allow_eval=allow_eval),
            )
        except Exception:
            # Best-effort header set; return original response
            logger.debug("Setup CSP middleware failed to attach CSP header")
        return response


__all__ = ["SetupCSPMiddleware"]
