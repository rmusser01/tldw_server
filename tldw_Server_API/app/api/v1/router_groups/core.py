"""Core router group — always-on infrastructure endpoints.

These routers are lightweight, have no heavy optional dependencies,
and are included in both full-app and minimal-test-app modes.
"""
from __future__ import annotations

from typing import Iterable

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

    # Basic infrastructure endpoints
    for infrastructure_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.health",
            log_name="health",
            prefix=f"{API_V1_PREFIX}",
            tags=("health",),
            route_key="health",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.moderation",
            log_name="moderation",
            prefix=f"{API_V1_PREFIX}",
            tags=("moderation",),
            route_key="moderation",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.monitoring",
            log_name="monitoring",
            prefix=f"{API_V1_PREFIX}",
            tags=("monitoring",),
            route_key="monitoring",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.metrics",
            log_name="metrics",
            prefix=f"{API_V1_PREFIX}",
            tags=("metrics",),
            route_key="metrics",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.audit",
            log_name="audit",
            prefix=f"{API_V1_PREFIX}",
            tags=("audit",),
            route_key="audit",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.consent",
            log_name="consent",
            prefix=f"{API_V1_PREFIX}",
            tags=("consent",),
            route_key="consent",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.setup",
            log_name="setup",
            prefix=f"{API_V1_PREFIX}",
            tags=("setup",),
            route_key="setup",
        ),
    ):
        append_imported_router_spec(specs, infrastructure_spec)

    # Identity, config, and sync endpoints
    for identity_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.auth",
            log_name="auth",
            prefix=f"{API_V1_PREFIX}",
            tags=("authentication",),
            route_key="auth",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.authnz_debug",
            log_name="authnz_debug",
            prefix=f"{API_V1_PREFIX}",
            tags=("authnz-debug",),
            route_key="authnz-debug",
            default_stable=_is_explicit_pytest_runtime(),
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.users",
            log_name="users",
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
            route_key="users",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.user_keys",
            log_name="user_keys",
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
            route_key="users",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.feedback",
            log_name="feedback",
            prefix=f"{API_V1_PREFIX}/feedback",
            tags=("feedback",),
            route_key="feedback",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.config_info",
            log_name="config_info",
            prefix=f"{API_V1_PREFIX}",
            tags=("config",),
            route_key="config",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.sync",
            log_name="sync",
            prefix=f"{API_V1_PREFIX}/sync",
            tags=("sync",),
            route_key="sync",
        ),
    ):
        append_imported_router_spec(specs, identity_spec)

    # Chat endpoints
    for chat_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chat",
            log_name="chat",
            prefix=f"{API_V1_PREFIX}/chat",
            route_key="chat",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chat_loop",
            log_name="chat_loop",
            prefix=f"{API_V1_PREFIX}",
            route_key="chat",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chat",
            log_name="conversations_alias",
            prefix=f"{API_V1_PREFIX}/chats",
            tags=("chat",),
            route_key="chat",
            attr_name="conversations_alias_router",
        ),
    ):
        append_imported_router_spec(specs, chat_spec)

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.tools",
            log_name="tools",
            prefix=f"{API_V1_PREFIX}",
            tags=("tools",),
            route_key="tools",
            default_stable=False,
        ),
    )

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

    # LLM/provider endpoints
    for provider_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.llm_providers",
            log_name="llm_providers",
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            route_key="llm",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mlx",
            log_name="mlx",
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            route_key="llm",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vllm_management",
            log_name="vllm_management",
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            route_key="llm",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.messages",
            log_name="messages",
            prefix=f"{API_V1_PREFIX}",
            tags=("messages",),
            route_key="llm",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.messages",
            log_name="messages_public",
            tags=("messages",),
            route_key="llm",
            attr_name="public_router",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.llamacpp",
            log_name="llamacpp",
            prefix=f"{API_V1_PREFIX}",
            tags=("llamacpp",),
            route_key="llamacpp",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.llamacpp",
            log_name="llamacpp_public",
            tags=("llamacpp",),
            route_key="llamacpp",
            attr_name="public_router",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vlm",
            log_name="vlm",
            prefix=f"{API_V1_PREFIX}",
            tags=("vlm",),
            route_key="vlm",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint",
            log_name="mcp_unified",
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-unified",),
            route_key="mcp-unified",
        ),
    ):
        append_imported_router_spec(specs, provider_spec)

    return specs
