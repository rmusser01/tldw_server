"""Shared dependency policy for workspace endpoints.

Stage 1 contract freeze artifact:
- Keeps workspace RBAC rate-limit resources explicit and stable before endpoint wiring.
- Lets endpoints and tests reuse the same dependency callables.
"""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit

WORKSPACES_READ_RATE_LIMIT_RESOURCE = "workspaces.read"
WORKSPACES_WRITE_RATE_LIMIT_RESOURCE = "workspaces.write"
WORKSPACES_DELETE_RATE_LIMIT_RESOURCE = "workspaces.delete"

WORKSPACES_READ_RATE_LIMIT = rbac_rate_limit(WORKSPACES_READ_RATE_LIMIT_RESOURCE)
WORKSPACES_WRITE_RATE_LIMIT = rbac_rate_limit(WORKSPACES_WRITE_RATE_LIMIT_RESOURCE)
WORKSPACES_DELETE_RATE_LIMIT = rbac_rate_limit(WORKSPACES_DELETE_RATE_LIMIT_RESOURCE)


@dataclass(frozen=True)
class WorkspaceRoutePolicy:
    """Contract-only policy descriptor for workspace endpoint wiring."""

    auth_dependency_name: str = "get_request_user"
    db_dependency_name: str = "get_chacha_db_for_user"
    rate_limit_factory_name: str = "rbac_rate_limit"
    read_resource: str = WORKSPACES_READ_RATE_LIMIT_RESOURCE
    write_resource: str = WORKSPACES_WRITE_RATE_LIMIT_RESOURCE
    delete_resource: str = WORKSPACES_DELETE_RATE_LIMIT_RESOURCE


WORKSPACES_RATE_LIMIT_POLICY = WorkspaceRoutePolicy()
