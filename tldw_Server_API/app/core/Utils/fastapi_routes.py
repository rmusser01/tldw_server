"""Walk a FastAPI app's routes the way it serves them (FastAPI >= 0.141).

Since FastAPI 0.137, ``include_router`` no longer copies a router's routes into
``app.routes``. It adds a single ``_IncludedRouter`` entry (no ``.path``, no
``.methods``) and applies the prefix, tags and dependencies at request time. Code that
walks ``app.routes`` and filters on ``getattr(route, "path")`` or
``isinstance(route, APIRoute)`` therefore silently sees only the top-level routes.

The same split applies at request time: ``request.scope["route"]`` is the router's
original route (local path, local tags, no include-time dependencies), so middleware
reading it mislabels metrics and misses tag policies and include-time guards.

``iter_served_routes`` flattens included routers through FastAPI's
``iter_route_contexts`` and ``served_route_for_scope`` resolves the route serving a
request; both report the full served path and the effective tags and dependencies,
including those added by ``include_router``. Those values live on FastAPI 0.141 private
state (``RouteContext._effective_route`` and ``scope["fastapi"]["effective_route_context"]``,
whose ``starlette_route`` is the merged copy for websocket, plain Starlette and Mount
routes). The pin in pyproject.toml is one minor version wide, and
tests/Utils/test_fastapi_routes.py fails loudly if a FastAPI upgrade moves them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi.routing import iter_route_contexts


@dataclass(frozen=True)
class ServedRoute:
    """One route as the app serves it.

    Attributes:
        path: The full served path, including every ``include_router`` prefix.
        methods: HTTP methods; empty for websocket routes and mounts.
        name: The route name, if any.
        endpoint: The endpoint callable, if any.
        route: The original route object (APIRoute, Route, APIWebSocketRoute, Mount...).
        dependant: The effective dependant, including include-time dependencies; None
            for routes without one.
        dependencies: Route-level plus include-time ``Depends`` declarations.
        tags: Route tags merged with include-time tags.
    """

    path: str
    methods: frozenset[str]
    name: str | None
    endpoint: Any
    route: Any
    dependant: Any
    dependencies: tuple[Any, ...]
    tags: tuple[str, ...]


def _merged(route: Any) -> Any:
    """Return the include-merged copy of an effective route context, if it has one."""
    return getattr(route, "starlette_route", None) or route


def iter_served_routes(routes: Iterable[Any]) -> Iterator[ServedRoute]:
    """Yield every route in ``routes`` (usually ``app.routes``) as it is served.

    Args:
        routes: A route list such as ``app.routes`` or ``router.routes``. Plain objects
            carrying ``path``/``methods`` (test doubles) pass through unchanged.

    Returns:
        An iterator of ServedRoute, flattened across included routers.
    """
    for ctx in iter_route_contexts(list(routes)):
        original = getattr(ctx, "route", ctx)
        effective = _merged(getattr(ctx, "_effective_route", None) or original)
        yield ServedRoute(
            path=getattr(effective, "path", None) or getattr(original, "path", None) or "",
            methods=frozenset(getattr(effective, "methods", None) or ()),
            name=getattr(effective, "name", None),
            endpoint=getattr(effective, "endpoint", None),
            route=original,
            dependant=getattr(effective, "dependant", None),
            dependencies=tuple(getattr(effective, "dependencies", None) or ()),
            tags=tuple(getattr(effective, "tags", None) or ()),
        )


def served_route_for_scope(scope: Mapping[str, Any]) -> Any:
    """Return the route serving a request, with include-time state applied.

    Args:
        scope: The ASGI scope, after routing (in an endpoint, a dependency, or a
            middleware once ``call_next`` has returned).

    Returns:
        An object with the served ``path``, merged ``tags`` and a ``dependant`` that
        includes include-time dependencies; ``scope["route"]`` for top-level routes;
        None before routing or when nothing matched.
    """
    fastapi_scope = scope.get("fastapi")
    context = fastapi_scope.get("effective_route_context") if isinstance(fastapi_scope, dict) else None
    return _merged(context) if context is not None else scope.get("route")
