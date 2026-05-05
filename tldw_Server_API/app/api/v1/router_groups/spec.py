from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import APIRouter

RouterFactory = Callable[[], APIRouter]


@dataclass(frozen=True)
class RouterSpec:
    """Specification for a router to be registered on the app.

    Attributes:
        router: The APIRouter instance (or a callable returning one for lazy import).
        prefix: URL prefix for this router (e.g., "/api/v1/media").
        tags: OpenAPI tags for grouping.
        route_key: Config key for route_enabled() gating. Empty string means always enabled.
        default_stable: Passed to route_enabled() as default_stable kwarg.
        name: Optional display name for diagnostics; falls back to route_key.
        skip_context: Optional extra diagnostic context for skip logs.
        skip_exceptions: Resolution exceptions that should skip registration.
    """
    router: APIRouter | RouterFactory
    prefix: str = ""
    tags: tuple[str, ...] = ()
    route_key: str = ""
    default_stable: bool = True
    name: str = ""
    skip_context: str = ""
    skip_exceptions: tuple[type[Exception], ...] = (ImportError, AttributeError)
    _resolved_router: APIRouter | None = field(default=None, init=False, repr=False, compare=False)

    def resolve_router(self) -> APIRouter:
        """Resolve eager routers and lazy router factories to an APIRouter."""
        if self._resolved_router is not None:
            return self._resolved_router
        if isinstance(self.router, APIRouter):
            return self.router
        router = self.router()
        object.__setattr__(self, "_resolved_router", router)
        return router
