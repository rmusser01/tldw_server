"""MCPTransport — abstract base for MCP transport implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tldw_Server_API.app.core.exceptions import ValidationError


class MCPTransport(ABC):
    """Abstract transport layer for communicating with an MCP server.

    Concrete implementations handle the protocol details (stdio, SSE,
    streamable HTTP) while exposing a uniform async interface to callers.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the MCP server."""

    @abstractmethod
    async def close(self) -> None:
        """Close the connection and release resources."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tools advertised by the MCP server."""

    @abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke a tool on the MCP server and return the result."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the transport connection is healthy."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is currently connected."""


def create_transport(protocol_config: dict[str, Any]) -> MCPTransport:
    """Factory: build the right MCPTransport from a protocol config dict.

    Parameters
    ----------
    protocol_config:
        Must contain ``"mcp_transport"`` key with one of ``"stdio"``,
        ``"sse"``, or ``"streamable_http"``.  Remaining keys are
        forwarded to the concrete transport constructor.

    Returns
    -------
    MCPTransport
        A concrete transport instance (not yet connected).

    Raises
    ------
    ValueError
        If the protocol is not recognised.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.sse import (
        MCPSSETransport,
    )
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.stdio import (
        MCPStdioTransport,
    )
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.streamable_http import (
        MCPStreamableHTTPTransport,
    )

    protocol = protocol_config.get("mcp_transport", "")

    def _require_key(key: str) -> Any:
        if key not in protocol_config:
            raise ValidationError(
                f"MCP transport {protocol!r} requires protocol_config key: {key}"
            )
        return protocol_config[key]

    if protocol == "stdio":
        return MCPStdioTransport(
            command=_require_key("command"),
            args=protocol_config.get("args"),
            env=protocol_config.get("env"),
        )
    elif protocol == "sse":
        return MCPSSETransport(
            sse_url=_require_key("sse_url"),
            post_url=protocol_config.get("post_url"),
            headers=protocol_config.get("headers"),
            timeout_sec=protocol_config.get("timeout_sec", 30),
            allow_private_network=protocol_config.get("allow_private_network"),
            allow_cross_origin_post_url=protocol_config.get("allow_cross_origin_post_url"),
        )
    elif protocol == "streamable_http":
        return MCPStreamableHTTPTransport(
            endpoint=_require_key("endpoint"),
            headers=protocol_config.get("headers"),
            timeout_sec=protocol_config.get("timeout_sec", 30),
            allow_private_network=protocol_config.get("allow_private_network"),
        )
    else:
        raise ValidationError(
            f"Unknown MCP transport protocol: {protocol!r}. "
            f"Supported: 'stdio', 'sse', 'streamable_http'."
        )
