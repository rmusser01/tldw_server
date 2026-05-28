"""Storage payload models for standalone MCP stores."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for storage payload defaults."""
    return datetime.now(timezone.utc)


def _copy_mapping(value: Any) -> dict[str, Any]:
    """Return a caller-owned mapping, treating explicit null as an empty mapping."""
    if value is None:
        return {}
    return deepcopy(value)


def _copy_list(value: Any) -> list[Any]:
    """Return a caller-owned list, treating explicit null as an empty list."""
    if value is None:
        return []
    return deepcopy(value)


class ProfileAssignment(BaseModel):
    """Principal/workspace binding for a stored MCP profile."""

    model_config = ConfigDict(extra="forbid")

    id: str
    profile_id: str
    principal_id: str | None = None
    workspace_id: str | None = None
    is_default: bool = False
    binding: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("binding", "provenance", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> dict[str, Any]:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return _copy_mapping(value)


class ApprovalPolicyDocument(BaseModel):
    """Reusable approval policy document for profile-bound actions."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    profile_id: str | None = None
    required_for: list[str] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("required_for", "rules", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[Any]:
        """Treat explicit null list fields as omitted safe defaults."""
        return _copy_list(value)

    @field_validator("metadata", "provenance", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> dict[str, Any]:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return _copy_mapping(value)


class CredentialGrant(BaseModel):
    """Credential broker grant metadata without embedded secret material."""

    model_config = ConfigDict(extra="forbid")

    id: str
    profile_id: str
    broker_id: str
    credential_slot: str
    external_server_id: str | None = None
    scopes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("scopes", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[Any]:
        """Treat explicit null list fields as omitted safe defaults."""
        return _copy_list(value)

    @field_validator("metadata", "provenance", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> dict[str, Any]:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return _copy_mapping(value)


class ExternalServerDefinition(BaseModel):
    """Stored definition for a configured upstream MCP server."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    transport: Literal["stdio", "websocket", "http"]
    command: list[str] = Field(default_factory=list)
    url: str | None = None
    cwd: str | None = None
    env_allowlist: list[str] = Field(default_factory=list)
    credential_slots: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    auto_start: bool = False
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("command", "env_allowlist", "credential_slots", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[Any]:
        """Treat explicit null list fields as omitted safe defaults."""
        return _copy_list(value)

    @field_validator("metadata", "provenance", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> dict[str, Any]:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return _copy_mapping(value)


class AuditEvent(BaseModel):
    """Append-only audit event payload for MCP policy and tool activity."""

    model_config = ConfigDict(extra="forbid")

    id: str
    event_type: str
    actor_id: str | None = None
    profile_id: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("payload", "provenance", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> dict[str, Any]:
        """Treat explicit null mapping fields as omitted safe defaults."""
        return _copy_mapping(value)
