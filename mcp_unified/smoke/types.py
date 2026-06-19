"""Shared protocol types for MCP smoke transports and clients."""

from __future__ import annotations

from typing import Protocol

JsonObject = dict[str, object]
JsonRpcPayload = JsonObject | list[object]


class McpSmokeTransport(Protocol):
    """Async transport protocol consumed by MCP smoke clients and scenarios."""

    async def start(self) -> None:
        """Open any resources needed by this transport."""

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Send a JSON-RPC request or batch payload and return the decoded response."""

    async def notify(self, payload: JsonObject) -> object | None:
        """Send a JSON-RPC notification payload and return any observed response."""

    async def close(self) -> None:
        """Release any resources opened by this transport."""


__all__ = ["JsonObject", "JsonRpcPayload", "McpSmokeTransport"]
