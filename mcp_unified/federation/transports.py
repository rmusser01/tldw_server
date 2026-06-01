"""Transport contracts for standalone external federation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from .models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
)


class ExternalFederationTransport(Protocol):
    """Minimal external transport lifecycle used by the standalone gateway."""

    server_id: str

    @property
    def transport_name(self) -> str:
        """Return a human-readable transport name."""
        ...

    async def connect(self) -> None:
        """Start the transport lifecycle."""
        ...

    async def close(self) -> None:
        """Stop the logical transport lifecycle."""
        ...

    async def health_check(self) -> dict[str, bool]:
        """Return quick health indicators for this transport."""
        ...

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Return discovered external tool definitions."""
        ...

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Execute a logical external tool call."""
        ...


class FakeExternalTransport:
    """In-memory non-spawning transport for registry and policy tests."""

    transport_name = "fake"

    def __init__(
        self,
        *,
        server_id: str,
        tools: list[ExternalToolDefinition] | None = None,
        results: dict[str, ExternalToolCallResult] | None = None,
        health: dict[str, bool] | None = None,
    ) -> None:
        self.server_id = server_id
        self.connected = False
        self.connect_count = 0
        self.close_count = 0
        self.spawn_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.runtime_auth_seen: BrokeredExternalCredential | None = None
        self._tools = [tool.copy() for tool in tools or []]
        self._results = {
            name: result.copy()
            for name, result in (results or {}).items()
        }
        self._health = dict(health or {})

    async def connect(self) -> None:
        """Mark the fake transport connected without spawning a process."""
        self.connect_count += 1
        self.connected = True

    async def close(self) -> None:
        """Mark the fake transport closed."""
        self.close_count += 1
        self.connected = False

    async def health_check(self) -> dict[str, bool]:
        """Return deterministic fake transport health."""
        return {
            "configured": True,
            "connected": self.connected,
            "spawns_process": False,
            **self._health,
        }

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Return caller-owned fake tool definitions."""
        return [tool.copy() for tool in self._tools]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Return a configured fake result for the requested tool."""
        del context
        call_args = deepcopy(arguments or {})
        self.calls.append((tool_name, call_args))
        self.runtime_auth_seen = None if runtime_auth is None else runtime_auth.copy()
        result = self._results.get(tool_name)
        if result is None:
            return ExternalToolCallResult(
                content={"error": f"Unknown fake external tool '{tool_name}'"},
                is_error=True,
                metadata={"reason_code": "unknown_fake_external_tool"},
            )
        return result.copy()
