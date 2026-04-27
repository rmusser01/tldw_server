"""Admin router group — administrative, billing, and org management endpoints."""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec

API_V1_PREFIX = "/api/v1"


def iter_admin_router_specs() -> Iterable[RouterSpec]:
    """Yield admin/ops router specs."""
    specs: list[RouterSpec] = []

    # Admin endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router

        specs.append(RouterSpec(
            router=admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("admin",),
            route_key="admin",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping admin router: {e}")

    # Guardian and family safety endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.family_wizard import router as family_wizard_router

        specs.append(RouterSpec(
            router=family_wizard_router,
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=("guardian",),
            route_key="guardian",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping family_wizard router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.guardian_controls import router as guardian_controls_router

        specs.append(RouterSpec(
            router=guardian_controls_router,
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=("guardian",),
            route_key="guardian",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping guardian_controls router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.self_monitoring import router as self_monitoring_router

        specs.append(RouterSpec(
            router=self_monitoring_router,
            prefix=f"{API_V1_PREFIX}/self-monitoring",
            tags=("self-monitoring",),
            route_key="self-monitoring",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping self_monitoring router: {e}")

    # Sandbox admin/ops endpoints.
    try:
        from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

        specs.append(RouterSpec(
            router=sandbox_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("sandbox",),
            route_key="sandbox",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping sandbox router: {e}")

    # Billing endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.billing import router as billing_router

        specs.append(RouterSpec(
            router=billing_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("billing",),
            route_key="billing",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping billing router: {e}")

    # Benchmarks endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.benchmark_api import router as benchmark_router

        specs.append(RouterSpec(
            router=benchmark_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("benchmarks",),
            route_key="benchmarks",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping benchmark router: {e}")

    # MCP catalogs management
    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage import (
            router as mcp_catalogs_manage_router,
        )

        specs.append(RouterSpec(
            router=mcp_catalogs_manage_router,
            prefix=f"{API_V1_PREFIX}",
            route_key="mcp-catalogs",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping mcp_catalogs router: {e}")

    # MCP hub management
    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_hub_management import (
            router as mcp_hub_management_router,
        )

        specs.append(RouterSpec(
            router=mcp_hub_management_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-hub",),
            route_key="mcp-hub",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping mcp_hub router: {e}")

    # Organizations management
    try:
        from tldw_Server_API.app.api.v1.endpoints.orgs import router as orgs_router

        specs.append(RouterSpec(
            router=orgs_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
            route_key="orgs",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping orgs router: {e}")

    # Scoped shared key management
    try:
        from tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped import (
            router as shared_keys_scoped_router,
        )

        specs.append(RouterSpec(
            router=shared_keys_scoped_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
            route_key="orgs",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping shared_keys_scoped router: {e}")

    # Privileges management
    try:
        from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router

        specs.append(RouterSpec(
            router=privileges_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("privileges",),
            route_key="privileges",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping privileges router: {e}")

    # Admin config diagnostics
    try:
        from tldw_Server_API.app.api.v1.endpoints.config_admin import router as config_admin_router

        specs.append(RouterSpec(
            router=config_admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("config", "admin"),
            route_key="config",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping config_admin router: {e}")

    # Resource Governor diagnostics
    try:
        from tldw_Server_API.app.api.v1.endpoints.resource_governor import router as resource_governor_router

        specs.append(RouterSpec(
            router=resource_governor_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("resource-governor",),
            route_key="resource-governor",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping resource_governor router: {e}")

    # Jobs admin diagnostics
    try:
        from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router

        specs.append(RouterSpec(
            router=jobs_admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("jobs",),
            route_key="jobs",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping jobs_admin router: {e}")

    # Organization invites
    try:
        from tldw_Server_API.app.api.v1.endpoints.org_invites import router as org_invites_router

        specs.append(RouterSpec(
            router=org_invites_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("invites",),
            route_key="org-invites",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping org_invites router: {e}")

    return specs
