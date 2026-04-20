"""Core router group — always-on infrastructure endpoints.

These routers are lightweight, have no heavy optional dependencies,
and are included in both full-app and minimal-test-app modes.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec

API_V1_PREFIX = "/api/v1"


def iter_core_router_specs() -> Iterable[RouterSpec]:
    """Yield core/always-on router specs.

    Each router is imported lazily inside this function to avoid
    import-time side effects at module level.
    """
    specs: list[RouterSpec] = []

    # Health endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.health import router as health_router

        specs.append(RouterSpec(
            router=health_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("health",),
            route_key="health",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping health router: {e}")

    # Moderation endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.moderation import router as moderation_router

        specs.append(RouterSpec(
            router=moderation_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("moderation",),
            route_key="moderation",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping moderation router: {e}")

    # Monitoring endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.monitoring import router as monitoring_router

        specs.append(RouterSpec(
            router=monitoring_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("monitoring",),
            route_key="monitoring",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping monitoring router: {e}")

    # Audit endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.audit import router as audit_router

        specs.append(RouterSpec(
            router=audit_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("audit",),
            route_key="audit",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping audit router: {e}")

    # Consent endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.consent import router as consent_router

        specs.append(RouterSpec(
            router=consent_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("consent",),
            route_key="consent",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping consent router: {e}")

    # Feedback endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.feedback import router as feedback_router

        specs.append(RouterSpec(
            router=feedback_router,
            prefix=f"{API_V1_PREFIX}/feedback",
            tags=("feedback",),
            route_key="feedback",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping feedback router: {e}")

    # Config info endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router

        specs.append(RouterSpec(
            router=config_info_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("config",),
            route_key="config",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping config_info router: {e}")

    # LLM Providers listing
    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_providers_router

        specs.append(RouterSpec(
            router=llm_providers_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            route_key="llm-providers",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping llm_providers router: {e}")

    # VLM backends listing
    try:
        from tldw_Server_API.app.api.v1.endpoints.vlm import router as vlm_router

        specs.append(RouterSpec(
            router=vlm_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("vlm",),
            route_key="vlm",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping vlm router: {e}")

    return specs
