"""Route-order and OpenAPI guards for nested Notes semantic operations."""

from __future__ import annotations

from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints.notes_semantic_index import router
from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs


def test_semantic_router_precedes_graph_and_parameterized_notes_routes() -> None:
    names = [spec.name for spec in iter_content_router_specs()]

    assert names.index("notes_semantic_index") < names.index("notes_graph")
    assert names.index("notes_semantic_index") < names.index("notes")


def test_openapi_exposes_only_nested_domain_runs_and_no_feature_jobs_surface() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/notes")
    paths = app.openapi()["paths"]

    assert {path for path in paths if "semantic-index" in path} == {
        "/api/v1/notes/graph/semantic-index/capabilities",
        "/api/v1/notes/graph/semantic-index",
        "/api/v1/notes/graph/semantic-index/runs",
        "/api/v1/notes/graph/semantic-index/runs/{run_id}",
        "/api/v1/notes/graph/semantic-index/runs/{run_id}/cancel",
    }
    assert not any(
        path.startswith("/api/v1/jobs") or path.startswith("/api/v1/notes/jobs")
        for path in paths
    )
    run_schema = paths["/api/v1/notes/graph/semantic-index/runs"]["post"]
    body_schema = run_schema["requestBody"]["content"]["application/json"]["schema"]
    assert body_schema["$ref"].endswith("SemanticRunCreateRequest")


def test_static_semantic_routes_are_registered_before_run_parameter_route() -> None:
    paths = [route.path for route in router.routes]

    assert paths.index("/graph/semantic-index/capabilities") < paths.index(
        "/graph/semantic-index/runs/{run_id}"
    )
    assert paths.index("/graph/semantic-index/runs") < paths.index(
        "/graph/semantic-index/runs/{run_id}"
    )
