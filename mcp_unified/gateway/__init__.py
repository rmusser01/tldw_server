"""Standalone MCP gateway entrypoint helpers."""

from typing import TYPE_CHECKING, Any

from .profile_runtime import ProfileAwareGatewayRuntime
from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime
from .stdio import GatewayStdioServer, handle_stdio_line

if TYPE_CHECKING:
    from .fastapi import create_gateway_app, create_gateway_router

__all__ = [
    "GatewayPolicyDenied",
    "GatewayRequestContext",
    "GatewayRuntime",
    "GatewayStdioServer",
    "ProfileAwareGatewayRuntime",
    "create_gateway_app",
    "create_gateway_router",
    "handle_stdio_line",
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
