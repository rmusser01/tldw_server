"""Runtime contracts for standalone MCP gateway transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class GatewayRequestContext:
    """Host-neutral context passed from gateway transports to runtimes."""

    request_id: str
    client_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GatewayRuntime(Protocol):
    """Minimal runtime surface needed by the standalone gateway skeleton."""

    name: str
    version: str

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return tools visible to the current request context."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute a tool call for the current request context."""
        ...
