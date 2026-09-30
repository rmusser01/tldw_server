"""Policy resolution: by_path, then the innermost mapped tag, then default for /api/."""

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI

from tldw_Server_API.app.core.Resource_Governance.policy_resolver import (
    PolicyResolver,
    compile_route_glob,
    get_policy_resolver,
)
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


def _app():
    inner = APIRouter(tags=["writing"])

    @inner.get("/docs/{doc_id}")
    def doc(doc_id: str) -> dict:
        return {}

    @inner.get("/docs/search")
    def search() -> dict:
        return {}

    middle = APIRouter()
    middle.include_router(inner, prefix="/writing")
    app = FastAPI()
    app.include_router(middle, prefix="/api/v1", tags=["content"])

    @app.post("/api/v1/auth/login", tags=["authentication"])
    def login() -> dict:
        return {}

    return app


ROUTE_MAP = {
    "by_path": {"/api/v1/auth*": "authnz.default", "/api/v1/media/*/reprocess": "media.default"},
    "by_tag": {"writing": "writing.policy", "content": "content.policy", "authentication": "wrong.policy"},
}


def _resolver(route_map=ROUTE_MAP, app=None):
    return PolicyResolver(route_map, iter_served_routes((app or _app()).routes))


def test_path_beats_tag():
    assert _resolver().resolve("/api/v1/auth/login", "POST") == "authnz.default"


def test_innermost_tag_wins_through_nested_includes():
    assert _resolver().resolve("/api/v1/writing/docs/7", "GET") == "writing.policy"


def test_outer_tag_applies_when_inner_tag_unmapped():
    rm = {"by_path": {}, "by_tag": {"content": "content.policy"}}
    assert _resolver(rm).resolve("/api/v1/writing/docs/7", "GET") == "content.policy"


def test_first_served_match_wins_within_a_prefix_group():
    # /docs/{doc_id} and /docs/search share the 4-segment group; served order decides, as in Starlette.
    rm = {"by_path": {}, "by_tag": {"writing": "writing.policy"}}
    assert _resolver(rm).resolve("/api/v1/writing/docs/search", "GET") == "writing.policy"


def test_method_mismatch_falls_back_to_default():
    assert _resolver().resolve("/api/v1/writing/docs/7", "DELETE") == "default"


def test_head_request_falls_back_to_default():
    assert _resolver().resolve("/api/v1/writing/docs/7", "HEAD") == "default"


def test_unmapped_api_path_resolves_to_default():
    assert _resolver().resolve("/api/v1/nothing/here", "GET") == "default"


def test_non_api_path_is_ungoverned():
    assert _resolver().resolve("/docs", "GET") is None
    assert _resolver().resolve("/static/app.js", "GET") is None


def test_mid_path_glob_matches_one_or_more_segments():
    rx = compile_route_glob("/api/v1/media/*/reprocess")
    assert rx.match("/api/v1/media/42/reprocess")
    assert rx.match("/api/v1/media/a/b/reprocess")
    assert not rx.match("/api/v1/media/42/reprocess/extra")
    assert compile_route_glob("/api/v1/auth*").match("/api/v1/authnz/x")  # trailing * is a prefix


def test_resolver_cache_is_rebuilt_when_snapshot_changes():
    app = _app()
    snap1 = SimpleNamespace(route_map={"by_path": {}, "by_tag": {"writing": "one"}})
    loader = SimpleNamespace(get_snapshot=lambda: loader.snap, snap=snap1)
    app.state.rg_policy_loader = loader
    assert get_policy_resolver(app).resolve("/api/v1/writing/docs/7", "GET") == "one"
    loader.snap = SimpleNamespace(route_map={"by_path": {}, "by_tag": {"writing": "two"}})
    assert get_policy_resolver(app).resolve("/api/v1/writing/docs/7", "GET") == "two"


def test_resolver_is_rebuilt_when_routes_are_added():
    app = _app()
    snap = SimpleNamespace(route_map={"by_path": {}, "by_tag": {"late": "late.policy"}})
    app.state.rg_policy_loader = SimpleNamespace(get_snapshot=lambda: snap)
    assert get_policy_resolver(app).resolve("/api/v1/late", "GET") == "default"
    late = APIRouter(tags=["late"])

    @late.get("/late")
    def late_ep() -> dict:
        return {}

    app.include_router(late, prefix="/api/v1")
    assert get_policy_resolver(app).resolve("/api/v1/late", "GET") == "late.policy"


def test_no_loader_means_no_resolver():
    assert get_policy_resolver(FastAPI()) is None
