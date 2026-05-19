"""Transport adapter contracts for external MCP server federation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from ..config_schema import ExternalMCPServerConfig


@dataclass(slots=True)
class ExternalToolDefinition:
    """Normalized external tool metadata used by federation logic."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExternalToolCallResult:
    """Normalized external tool execution result."""

    content: Any
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrokeredExternalCredential:
    """Ephemeral per-call auth material resolved outside long-lived adapter state."""

    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalMCPTransportAdapter(ABC):
    """Adapter contract for connecting to and invoking external MCP servers."""

    def __init__(self, server_id: str) -> None:
        self.server_id = server_id

    @property
    @abstractmethod
    def transport_name(self) -> str:
        """Human-readable transport name (e.g. websocket/stdio)."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish transport connection/session."""

    @abstractmethod
    async def close(self) -> None:
        """Close any active transport connection/session."""

    @abstractmethod
    async def health_check(self) -> dict[str, bool]:
        """Return quick health indicators for the upstream transport."""

    @abstractmethod
    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Discover external tools and return normalized definitions."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Optional["RequestContext"] = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Execute a tool on the external server and normalize the result."""


def adapter_supports_runtime_auth(adapter: ExternalMCPTransportAdapter) -> bool:
    """Return whether an adapter's call contract accepts runtime auth injection."""
    try:
        params = inspect.signature(adapter.call_tool).parameters
    except (TypeError, ValueError):
        return True
    return "runtime_auth" in params


def clone_external_server_config(
    server_config: "ExternalMCPServerConfig",
) -> "ExternalMCPServerConfig":
    """Deep-copy an external server config without mutating long-lived adapter state."""
    if hasattr(server_config, "model_copy"):
        return server_config.model_copy(deep=True)  # type: ignore[attr-defined]
    return server_config.copy(deep=True)


async def call_tool_with_ephemeral_adapter(
    *,
    server_config: "ExternalMCPServerConfig",
    adapter_factory: Callable[["ExternalMCPServerConfig"], ExternalMCPTransportAdapter],
    prepare_config: Callable[["ExternalMCPServerConfig"], None],
    tool_name: str,
    arguments: dict[str, Any],
) -> ExternalToolCallResult:
    """Run one external tool call against a short-lived adapter with cloned config."""
    ephemeral_config = clone_external_server_config(server_config)
    prepare_config(ephemeral_config)
    ephemeral_adapter = adapter_factory(ephemeral_config)
    try:
        await ephemeral_adapter.connect()
        call_kwargs: dict[str, Any] = {}
        if adapter_supports_runtime_auth(ephemeral_adapter):
            call_kwargs["runtime_auth"] = None
        return await ephemeral_adapter.call_tool(tool_name, arguments, **call_kwargs)
    finally:
        await ephemeral_adapter.close()
