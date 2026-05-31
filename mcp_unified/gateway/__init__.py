"""Standalone MCP gateway entrypoint helpers."""

from typing import TYPE_CHECKING, Any

from .bootstrap import (
    GatewayProfileBootstrap,
    bootstrap_profile_gateway,
    build_profile_gateway_runtime,
)
from .config import (
    GatewayConfigFormat,
    GatewayProfileBootstrapConfig,
    GatewayProfileStoreConfig,
    GatewayProfileStoreKind,
    bootstrap_profile_gateway_from_config,
    load_gateway_profile_bootstrap_config,
)
from .profile_runtime import ProfileAwareGatewayRuntime
from .profiles import (
    GatewayProfileManagementError,
    GatewayProfileManager,
    GatewayProfileStoreMetadata,
)
from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime
from .stdio import GatewayStdioServer, handle_stdio_line

if TYPE_CHECKING:
    from .fastapi import create_gateway_app, create_gateway_router

__all__ = [
    "GatewayPolicyDenied",
    "GatewayConfigFormat",
    "GatewayProfileBootstrap",
    "GatewayProfileBootstrapConfig",
    "GatewayProfileManagementError",
    "GatewayProfileManager",
    "GatewayProfileStoreMetadata",
    "GatewayRequestContext",
    "GatewayRuntime",
    "GatewayStdioServer",
    "GatewayProfileStoreConfig",
    "GatewayProfileStoreKind",
    "ProfileAwareGatewayRuntime",
    "bootstrap_profile_gateway",
    "bootstrap_profile_gateway_from_config",
    "build_profile_gateway_runtime",
    "create_gateway_app",
    "create_gateway_router",
    "handle_stdio_line",
    "load_gateway_profile_bootstrap_config",
]


def __getattr__(name: str) -> Any:
    """Lazily expose FastAPI helpers so stdio imports do not require FastAPI."""

    if name in {"create_gateway_app", "create_gateway_router"}:
        from .fastapi import create_gateway_app, create_gateway_router

        return {
            "create_gateway_app": create_gateway_app,
            "create_gateway_router": create_gateway_router,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
