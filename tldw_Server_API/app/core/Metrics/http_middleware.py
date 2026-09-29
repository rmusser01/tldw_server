from time import monotonic
from typing import Callable

from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.Utils.fastapi_routes import served_route_for_scope


class HTTPMetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.registry = get_metrics_registry()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = monotonic()
        method = request.method
        status_code: int = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                status_code = status
            elif isinstance(exc, RequestValidationError):
                status_code = 422
            # Exception will propagate after recording metrics
            raise
        finally:
            duration = monotonic() - start
            # Try to get a stable route template; fallback to path
            # served_route_for_scope: scope["route"] carries only the router-local path
            # under FastAPI >= 0.137, which collapses every router's "/" into one label.
            route = served_route_for_scope(request.scope)
            endpoint = getattr(route, "path", None)
            if not endpoint:
                endpoint = getattr(request.scope.get("endpoint"), "__name__", None)
            if not endpoint:
                endpoint = "unmatched"
            # Record metrics
            self.registry.increment(
                "http_requests_total",
                1,
                labels={"method": method, "endpoint": endpoint, "status": str(status_code)},
            )
            self.registry.observe(
                "http_request_duration_seconds",
                duration,
                labels={"method": method, "endpoint": endpoint},
            )
