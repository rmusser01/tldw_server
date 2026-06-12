"""Contract tests: every explainer route declares an RBAC rate-limit resource."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import explainer as explainer_ep

pytestmark = pytest.mark.unit


def _route_resources() -> dict[tuple[str, str], list[str | None]]:
    return {
        (route.path, next(iter(sorted(route.methods or [])))): [
            getattr(dependency.call, "_tldw_rate_limit_resource", None)
            for dependency in route.dependant.dependencies
        ]
        for route in explainer_ep.router.routes
        if getattr(route, "path", "").startswith("/")
    }


def test_explainer_rate_limit_resources_are_stable() -> None:
    assert explainer_ep.EXPLAINER_READ_RATE_LIMIT_RESOURCE == "explainer.read"
    assert explainer_ep.EXPLAINER_WRITE_RATE_LIMIT_RESOURCE == "explainer.write"
    assert explainer_ep.EXPLAINER_EXPAND_RATE_LIMIT_RESOURCE == "explainer.expand"
    assert explainer_ep.EXPLAINER_EXPORT_RATE_LIMIT_RESOURCE == "explainer.export"


def test_explainer_routes_include_expected_rbac_rate_limits() -> None:
    route_resources = _route_resources()

    read = explainer_ep.EXPLAINER_READ_RATE_LIMIT_RESOURCE
    write = explainer_ep.EXPLAINER_WRITE_RATE_LIMIT_RESOURCE
    expand = explainer_ep.EXPLAINER_EXPAND_RATE_LIMIT_RESOURCE
    export = explainer_ep.EXPLAINER_EXPORT_RATE_LIMIT_RESOURCE

    assert read in route_resources[("/explainer/sessions", "GET")]
    assert read in route_resources[("/explainer/sessions/{session_id}", "GET")]
    assert read in route_resources[("/explainer/jobs/{job_id}", "GET")]

    assert write in route_resources[("/explainer/sessions", "POST")]
    assert write in route_resources[("/explainer/sessions/{session_id}", "PATCH")]
    assert write in route_resources[("/explainer/sessions/{session_id}", "DELETE")]
    assert write in route_resources[("/explainer/sessions/{session_id}/nodes", "POST")]
    assert write in route_resources[("/explainer/sessions/{session_id}/nodes/{node_id}", "PATCH")]
    assert write in route_resources[("/explainer/sessions/{session_id}/nodes/{node_id}", "DELETE")]
    assert write in route_resources[
        ("/explainer/sessions/{session_id}/nodes/{node_id}/answer-question", "POST")
    ]

    assert expand in route_resources[
        ("/explainer/sessions/{session_id}/nodes/{node_id}/expand", "POST")
    ]
    assert export in route_resources[("/explainer/sessions/{session_id}/export-chatbook", "POST")]


def test_every_explainer_route_declares_a_rate_limit() -> None:
    for (path, method), resources in _route_resources().items():
        assert any(resources), f"{method} {path} has no RBAC rate-limit dependency"
