"""Core router group — always-on infrastructure endpoints.

These routers are lightweight, have no heavy optional dependencies,
and are included in both full-app and minimal-test-app modes.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.conditional import (
    ImportedRouterSpec,
    append_imported_router_spec,
)
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime as _is_explicit_pytest_runtime

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

    # Metrics endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router

        specs.append(RouterSpec(
            router=metrics_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("metrics",),
            route_key="metrics",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping metrics router: {e}")

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

    # Setup endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.setup import router as setup_router

        specs.append(RouterSpec(
            router=setup_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("setup",),
            route_key="setup",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping setup router: {e}")

    # Authentication endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.auth import router as auth_router

        specs.append(RouterSpec(
            router=auth_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("authentication",),
            route_key="auth",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping auth router: {e}")

    # AuthNZ debug endpoints are enabled by default only in explicit pytest runtime.
    try:
        from tldw_Server_API.app.api.v1.endpoints.authnz_debug import router as authnz_debug_router

        specs.append(RouterSpec(
            router=authnz_debug_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("authnz-debug",),
            route_key="authnz-debug",
            default_stable=_is_explicit_pytest_runtime(),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping authnz_debug router: {e}")

    # User management endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.users import router as users_router

        specs.append(RouterSpec(
            router=users_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
            route_key="users",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping users router: {e}")

    # User key management endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.user_keys import router as user_keys_router

        specs.append(RouterSpec(
            router=user_keys_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
            route_key="users",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping user_keys router: {e}")

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

    # Sync endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.sync import router as sync_router

        specs.append(RouterSpec(
            router=sync_router,
            prefix=f"{API_V1_PREFIX}/sync",
            tags=("sync",),
            route_key="sync",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping sync router: {e}")

    # Chat endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.chat import router as chat_router

        specs.append(RouterSpec(
            router=chat_router,
            prefix=f"{API_V1_PREFIX}/chat",
            route_key="chat",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chat router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.chat_loop import router as chat_loop_router

        specs.append(RouterSpec(
            router=chat_loop_router,
            prefix=f"{API_V1_PREFIX}",
            route_key="chat",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chat_loop router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.chat import conversations_alias_router

        specs.append(RouterSpec(
            router=conversations_alias_router,
            prefix=f"{API_V1_PREFIX}/chats",
            tags=("chat",),
            route_key="chat",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping conversations_alias router: {e}")

    # Tools endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router

        specs.append(RouterSpec(
            router=tools_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("tools",),
            route_key="tools",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping tools router: {e}")

    # Agent Client Protocol (ACP) endpoints
    for acp_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.agent_client_protocol",
            log_name="acp",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp",),
            route_key="acp",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_schedules",
            log_name="acp_schedules",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-schedules",),
            route_key="acp",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_triggers",
            log_name="acp_triggers",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-triggers",),
            route_key="acp",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_permissions",
            log_name="acp_permissions",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-permissions",),
            route_key="acp",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_multiplex",
            log_name="acp_multiplex",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-multiplex",),
            route_key="acp",
            default_stable=False,
        ),
    ):
        append_imported_router_spec(specs, acp_spec)

    # LLM Providers listing
    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_providers_router

        specs.append(RouterSpec(
            router=llm_providers_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            route_key="llm",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping llm_providers router: {e}")

    # MLX endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.mlx import router as mlx_router

        specs.append(RouterSpec(
            router=mlx_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            route_key="llm",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping mlx router: {e}")

    # Message endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.messages import public_router as messages_public_router
        from tldw_Server_API.app.api.v1.endpoints.messages import router as messages_router

        specs.append(RouterSpec(
            router=messages_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("messages",),
            route_key="llm",
        ))
        specs.append(RouterSpec(
            router=messages_public_router,
            tags=("messages",),
            route_key="llm",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping messages routers: {e}")

    # llama.cpp endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import public_router as llamacpp_public_router
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import router as llamacpp_router

        specs.append(RouterSpec(
            router=llamacpp_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("llamacpp",),
            route_key="llamacpp",
        ))
        specs.append(RouterSpec(
            router=llamacpp_public_router,
            tags=("llamacpp",),
            route_key="llamacpp",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping llamacpp routers: {e}")

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

    # MCP unified endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router

        specs.append(RouterSpec(
            router=mcp_unified_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-unified",),
            route_key="mcp-unified",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping mcp_unified router: {e}")

    return specs
