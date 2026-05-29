"""Standalone external federation data contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

MCPAuthType = Literal["none", "bearer", "api_key"]


class MCPCatalogEntry(BaseModel):
    """One entry in the external MCP server catalog shown during setup."""

    key: str
    name: str
    description: str
    url_template: str
    auth_type: MCPAuthType = "none"
    category: str
    logo_key: str | None = None
    suggested_for: list[str] = Field(default_factory=list)


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return caller-owned mapping data for public result objects."""
    return deepcopy(value or {})


@dataclass(slots=True)
class BrokeredExternalCredential:
    """Ephemeral per-call auth material resolved outside long-lived adapter state."""

    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> BrokeredExternalCredential:
        """Return caller-owned brokered credential data."""
        return BrokeredExternalCredential(
            headers=dict(self.headers or {}),
            env=dict(self.env or {}),
            metadata=_copy_mapping(self.metadata),
        )


@dataclass(slots=True)
class ExternalToolDefinition:
    """Normalized external tool metadata discovered from a transport."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> ExternalToolDefinition:
        """Return a caller-owned copy of this tool definition."""
        return ExternalToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=_copy_mapping(self.input_schema),
            metadata=_copy_mapping(self.metadata),
        )


@dataclass(slots=True)
class ExternalToolCallResult:
    """Normalized external transport call result."""

    content: Any
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> ExternalToolCallResult:
        """Return a caller-owned copy of this transport result."""
        return ExternalToolCallResult(
            content=deepcopy(self.content),
            is_error=self.is_error,
            metadata=_copy_mapping(self.metadata),
        )


@dataclass(slots=True)
class VirtualExternalTool:
    """External tool exposed through a namespaced virtual tool name."""

    virtual_name: str
    server_id: str
    upstream_tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    metadata: dict[str, Any] = field(default_factory=dict)
    is_write: bool = False


@dataclass(slots=True)
class FederatedToolResult:
    """Result returned by the standalone federation manager."""

    content: Any
    server_id: str
    upstream_tool_name: str
    virtual_tool_name: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class FederationPolicyDenied(PermissionError):
    """Raised when effective policy blocks an external tool execution."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.payload = _copy_mapping(payload)
