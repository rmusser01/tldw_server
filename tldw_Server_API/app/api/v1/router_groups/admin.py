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
