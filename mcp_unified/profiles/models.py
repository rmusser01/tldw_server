"""Profile models for MCP tool and permission presets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for profile metadata defaults."""
    return datetime.now(timezone.utc)


class ProfilePolicy(BaseModel):
    """Enforceable tool and capability policy for an MCP profile."""

    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    denied_capabilities: list[str] = Field(default_factory=list)
    tool_patterns: list[str] = Field(default_factory=list)
    module_patterns: list[str] = Field(default_factory=list)
    risk_classes: list[str] = Field(default_factory=list)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)


class MCPProfile(BaseModel):
    """User-selectable MCP profile that defines an enforceable action boundary."""

    id: str
    name: str
    description: str = ""
    schema_version: int = 1
    preset_id: str | None = None
    preset_version: str | None = None
    enabled: bool = True
    policy_document: ProfilePolicy = Field(default_factory=ProfilePolicy)
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    path_scopes: list[dict[str, Any]] = Field(default_factory=list)
    external_server_grants: list[dict[str, Any]] = Field(default_factory=list)
    credential_grants: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
