from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fastapi import APIRouter


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
    router: APIRouter
    prefix: str = ""
    tags: tuple[str, ...] = ()
    route_key: str = ""
    default_stable: bool = True
