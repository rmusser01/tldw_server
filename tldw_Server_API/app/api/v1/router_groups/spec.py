from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    """
    router: APIRouter | RouterFactory
    prefix: str = ""
    tags: tuple[str, ...] = ()
    route_key: str = ""
    default_stable: bool = True

    def resolve_router(self) -> APIRouter:
        """Resolve eager routers and lazy router factories to an APIRouter."""
        if isinstance(self.router, APIRouter):
            return self.router
        return self.router()
