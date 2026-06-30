"""Runtime contracts for standalone MCP gateway transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class GatewayPolicyDenied(PermissionError):
    """Raised when gateway profile policy denies execution."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        status: str = "denied",
        provenance: dict[str, Any] | None = None,
        warnings: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.status = status
        self.provenance = provenance or {}
        self.warnings = warnings or []

    def to_error_data(self) -> dict[str, Any]:
        """Return a JSON-serializable denial payload for JSON-RPC errors."""

        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "provenance": self.provenance,
            "warnings": self.warnings,
        }


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

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return resources visible to the current request context."""
        ...

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Read one resource for the current request context."""
        ...

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return prompts visible to the current request context."""
        ...

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Return one prompt result for the current request context."""
        ...

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return module registrations visible to the current request context."""
        ...

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Return module health details visible to the current request context."""
        ...
