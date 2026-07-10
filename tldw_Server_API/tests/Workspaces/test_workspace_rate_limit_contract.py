from __future__ import annotations

from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_ep
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT_RESOURCE,
    WORKSPACES_RATE_LIMIT_POLICY,
    WORKSPACES_READ_RATE_LIMIT_RESOURCE,
    WORKSPACES_WRITE_RATE_LIMIT_RESOURCE,
)


def test_workspace_rate_limit_resources_are_stable() -> None:
    assert WORKSPACES_READ_RATE_LIMIT_RESOURCE == "workspaces.read"
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE == "workspaces.write"
    assert WORKSPACES_DELETE_RATE_LIMIT_RESOURCE == "workspaces.delete"


def test_workspace_rate_limit_policy_contract_values() -> None:
    assert WORKSPACES_RATE_LIMIT_POLICY.auth_dependency_name == "get_request_user"
    assert WORKSPACES_RATE_LIMIT_POLICY.db_dependency_name == "get_chacha_db_for_user"
    assert WORKSPACES_RATE_LIMIT_POLICY.rate_limit_factory_name == "rbac_rate_limit"
    assert WORKSPACES_RATE_LIMIT_POLICY.read_resource == WORKSPACES_READ_RATE_LIMIT_RESOURCE
    assert WORKSPACES_RATE_LIMIT_POLICY.write_resource == WORKSPACES_WRITE_RATE_LIMIT_RESOURCE
    assert WORKSPACES_RATE_LIMIT_POLICY.delete_resource == WORKSPACES_DELETE_RATE_LIMIT_RESOURCE


def test_workspace_routes_include_expected_rbac_rate_limits() -> None:
    route_resources = {
        (route.path, next(iter(sorted(route.methods or [])))): [
            getattr(dependency.call, "_tldw_rate_limit_resource", None)
            for dependency in route.dependant.dependencies
        ]
        for route in workspaces_ep.router.routes
        if getattr(route, "path", "").startswith("/")
    }

    assert WORKSPACES_READ_RATE_LIMIT_RESOURCE in route_resources[("/", "GET")]
    assert WORKSPACES_READ_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}", "GET")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}", "PUT")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}", "PATCH")]
    assert WORKSPACES_DELETE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}", "DELETE")]

    assert WORKSPACES_READ_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources", "GET")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources", "POST")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources/{source_id}", "PUT")]
    assert WORKSPACES_DELETE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources/{source_id}", "DELETE")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources/selection", "PUT")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources/reorder", "PUT")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/sources/review-state", "PUT")]

    assert WORKSPACES_READ_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/artifacts", "GET")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/artifacts", "POST")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/artifacts/{artifact_id}", "PUT")]
    assert WORKSPACES_DELETE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/artifacts/{artifact_id}", "DELETE")]

    assert WORKSPACES_READ_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/notes", "GET")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/notes", "POST")]
    assert WORKSPACES_WRITE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/notes/{note_id}", "PUT")]
    assert WORKSPACES_DELETE_RATE_LIMIT_RESOURCE in route_resources[("/{workspace_id}/notes/{note_id}", "DELETE")]
