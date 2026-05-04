"""Admin router group — administrative, billing, and org management endpoints."""
from __future__ import annotations

from typing import Iterable

from tldw_Server_API.app.api.v1.router_groups.conditional import (
    ImportedRouterSpec,
    append_imported_router_spec,
)
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec

API_V1_PREFIX = "/api/v1"


def iter_admin_router_specs() -> Iterable[RouterSpec]:
    """Yield admin/ops router specs."""
    specs: list[RouterSpec] = []

    for admin_spec in (
        # Admin endpoints
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.admin",
            log_name="admin",
            prefix=f"{API_V1_PREFIX}",
            tags=("admin",),
            route_key="admin",
        ),
        # Guardian and family safety endpoints
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.family_wizard",
            log_name="family_wizard",
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=("guardian",),
            route_key="guardian",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.guardian_controls",
            log_name="guardian_controls",
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=("guardian",),
            route_key="guardian",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.self_monitoring",
            log_name="self_monitoring",
            prefix=f"{API_V1_PREFIX}/self-monitoring",
            tags=("self-monitoring",),
            route_key="self-monitoring",
            default_stable=False,
        ),
        # Sandbox admin/ops endpoints.
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.sandbox",
            log_name="sandbox",
            prefix=f"{API_V1_PREFIX}",
            tags=("sandbox",),
            route_key="sandbox",
            default_stable=False,
        ),
        # Billing endpoints
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.billing",
            log_name="billing",
            prefix=f"{API_V1_PREFIX}",
            tags=("billing",),
            route_key="billing",
        ),
        # Benchmarks endpoints
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.benchmark_api",
            log_name="benchmark",
            prefix=f"{API_V1_PREFIX}",
            tags=("benchmarks",),
            route_key="benchmarks",
            default_stable=False,
        ),
        # MCP catalogs management
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage",
            log_name="mcp_catalogs",
            prefix=f"{API_V1_PREFIX}",
            route_key="mcp-catalogs",
        ),
        # MCP hub management
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mcp_hub_management",
            log_name="mcp_hub",
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-hub",),
            route_key="mcp-hub",
        ),
        # Organizations management
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.orgs",
            log_name="orgs",
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
            route_key="orgs",
        ),
        # Scoped shared key management
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped",
            log_name="shared_keys_scoped",
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
            route_key="orgs",
        ),
        # Privileges management
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.privileges",
            log_name="privileges",
            prefix=f"{API_V1_PREFIX}",
            tags=("privileges",),
            route_key="privileges",
        ),
        # Admin config diagnostics
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.config_admin",
            log_name="config_admin",
            prefix=f"{API_V1_PREFIX}",
            tags=("config", "admin"),
            route_key="config",
        ),
        # Resource Governor diagnostics
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.resource_governor",
            log_name="resource_governor",
            prefix=f"{API_V1_PREFIX}",
            tags=("resource-governor",),
            route_key="resource-governor",
        ),
        # Jobs admin diagnostics
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.jobs_admin",
            log_name="jobs_admin",
            prefix=f"{API_V1_PREFIX}",
            tags=("jobs",),
            route_key="jobs",
            default_stable=False,
        ),
        # Organization invites
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.org_invites",
            log_name="org_invites",
            prefix=f"{API_V1_PREFIX}",
            tags=("invites",),
            route_key="org-invites",
        ),
    ):
        append_imported_router_spec(specs, admin_spec)

    return specs
