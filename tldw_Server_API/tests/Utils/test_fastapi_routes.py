"""The served-route helpers must see routes behind include_router (FastAPI >= 0.141).

These are also the guard for the private FastAPI state the helpers read
(RouteContext._effective_route and scope["fastapi"]["effective_route_context"]): if an
upgrade moves it, the include-time path, dependency and tag checks fail here rather
than every route-walking feature and middleware silently losing included routers.
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request, WebSocket
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.staticfiles import StaticFiles

from tldw_Server_API.app.core.Utils.fastapi_routes import ServedRoute, iter_served_routes, served_route_for_scope

pytestmark = pytest.mark.unit


def _auth() -> str:
    """Stands in for an include-time auth dependency."""
    return "user"


def _route_dep() -> str:
    """Stands in for a route-level dependency."""
    return "route"


def _app(seen: dict | None = None) -> FastAPI:
    """Nested routers, include-time prefix/tags/dependencies, a websocket and a mount.

    Endpoints record served_route_for_scope into ``seen`` when a request reaches them.
    """
    seen = seen if seen is not None else {}
    inner = APIRouter(tags=["inner"])

    @inner.get("/items", dependencies=[Depends(_route_dep)])
    def items(request: Request) -> list[str]:
        seen["items"] = served_route_for_scope(request.scope)
        return []

    @inner.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        seen["ws"] = served_route_for_scope(websocket.scope)
        await websocket.accept()
        await websocket.close()

    outer = APIRouter()
    outer.include_router(inner, prefix="/v1")
    app = FastAPI()
    app.include_router(outer, prefix="/api", tags=["outer"], dependencies=[Depends(_auth)])
    app.mount("/static", StaticFiles(directory=tempfile.mkdtemp()), name="static")

    @app.get("/top")
    def top(request: Request) -> None:
        seen["top"] = served_route_for_scope(request.scope)

    @app.middleware("http")
    async def record(request: Request, call_next):  # noqa: ANN001, ANN202
        response = await call_next(request)
        seen.setdefault("middleware", []).append(served_route_for_scope(request.scope))
        return response

    return app


def _served(path: str) -> ServedRoute:
    return next(r for r in iter_served_routes(_app().routes) if r.path == path)


def test_routes_behind_include_router_are_served_with_their_full_path() -> None:
    """The bug this exists for: app.routes alone has no /api/v1/items entry at all."""
    app = _app()
    assert all(getattr(r, "path", None) != "/api/v1/items" for r in app.routes)

    route = _served("/api/v1/items")

    assert route.methods == {"GET"}
    assert isinstance(route.route, APIRoute)


def test_include_time_dependencies_are_effective() -> None:
    """An auth dependency added by include_router must count, or every guarded route looks open."""
    route = _served("/api/v1/items")

    calls = {d.call for d in route.dependant.dependencies}
    assert {_auth, _route_dep} <= calls
    assert {getattr(d, "dependency", None) for d in route.dependencies} >= {_auth, _route_dep}


def test_include_time_tags_are_merged() -> None:
    assert set(_served("/api/v1/items").tags) == {"outer", "inner"}


def test_mounts_top_level_and_websocket_routes_appear_with_served_paths() -> None:
    paths = {r.path for r in iter_served_routes(_app().routes)}
    assert {"/openapi.json", "/static", "/top", "/api/v1/ws"} <= paths


def test_request_scope_resolves_the_included_route_as_served() -> None:
    """scope["route"] alone is the local /items route with no auth dependency."""
    seen: dict = {}
    TestClient(_app(seen)).get("/api/v1/items")

    route = seen["items"]
    assert route.path == "/api/v1/items"
    assert {"outer", "inner"} <= set(route.tags)
    assert {_auth, _route_dep} <= {d.call for d in route.dependant.dependencies}


def test_middleware_sees_the_served_route_after_call_next() -> None:
    seen: dict = {}
    client = TestClient(_app(seen))
    client.get("/api/v1/items")
    client.get("/top")
    client.get("/missing")

    assert [getattr(r, "path", None) for r in seen["middleware"]] == ["/api/v1/items", "/top", None]


def test_websocket_scope_resolves_the_served_path() -> None:
    seen: dict = {}
    with TestClient(_app(seen)).websocket_connect("/api/v1/ws"):
        pass

    assert seen["ws"].path == "/api/v1/ws"


def test_plain_test_doubles_pass_through() -> None:
    fake = SimpleNamespace(path="/x", methods={"GET"}, name="x", endpoint=None)

    (route,) = list(iter_served_routes([fake]))

    assert (route.path, route.methods, route.route) == ("/x", {"GET"}, fake)


def test_router_routes_version_changes_on_include() -> None:
    """policy_resolver rebuilds its index on this private FastAPI counter (read per request)."""
    app = FastAPI()
    before = app.router._routes_version
    extra = APIRouter()

    @extra.get("/x")
    def x() -> None:
        return None

    app.include_router(extra)
    assert isinstance(app.router._routes_version, int) and app.router._routes_version != before
