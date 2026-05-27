"""Profile models for MCP tool and permission presets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for profile metadata defaults."""
    return datetime.now(timezone.utc)


class ProfilePolicy(BaseModel):
    """Enforceable tool and capability policy for an MCP profile."""

    model_config = ConfigDict(extra="allow")

    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    denied_capabilities: list[str] = Field(default_factory=list)
    tool_patterns: list[str] = Field(default_factory=list)
    module_patterns: list[str] = Field(default_factory=list)
    risk_classes: list[str] = Field(default_factory=list)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "allowed_tools",
        "denied_tools",
        "capabilities",
        "denied_capabilities",
        "tool_patterns",
        "module_patterns",
        "risk_classes",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_list(cls, value: Any) -> Any:
        """Treat explicit null list fields as omitted safe defaults."""
        return [] if value is None else value

    @field_validator("resource_constraints", mode="before")
    @classmethod
    def _coerce_none_to_dict(cls, value: Any) -> Any:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return {} if value is None else value


class MCPProfile(BaseModel):
    """User-selectable MCP profile that defines an enforceable action boundary."""

    model_config = ConfigDict(extra="allow")

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
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("policy_document", mode="before")
    @classmethod
    def _coerce_none_to_policy(cls, value: Any) -> Any:
        """Treat an explicit null policy as the empty safe policy."""
        return {} if value is None else value

    @field_validator(
        "path_scopes",
        "external_server_grants",
        "credential_grants",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_list(cls, value: Any) -> Any:
        """Treat explicit null list fields as omitted safe defaults."""
        return [] if value is None else value

    @field_validator(
        "approval_policy",
        "metadata",
        "provenance",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_dict(cls, value: Any) -> Any:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return {} if value is None else value
