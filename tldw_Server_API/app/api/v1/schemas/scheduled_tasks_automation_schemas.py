"""Schemas for Scheduled Tasks automation definition contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ScheduledTaskAutomationFamily = Literal["recurring_question", "agent_task"]
ScheduledTaskAutomationActionStatus = Literal["available", "unavailable", "planned", "disabled"]
ScheduledTaskAutomationFamilyAvailability = Literal["available", "planned", "unavailable", "degraded"]
ScheduledTaskPreviewMode = Literal["create", "update"]
ScheduledTaskPreviewStatus = Literal["valid", "invalid", "expired", "consumed"]
ScheduledTaskDefinitionLifecycle = Literal["configured", "paused", "archived", "disabled"]
ScheduledTaskDefinitionCreateLifecycle = Literal["configured", "paused"]
ScheduledTaskDefinitionHealth = Literal[
    "ready",
    "execution_unavailable",
    "capability_unavailable",
    "needs_attention",
    "permission_required",
]
ScheduledTaskDefinitionDisabledLockKind = Literal["none", "admin", "security", "system"]


class ScheduledTaskActionCapability(BaseModel):
    """Capability status for a single Scheduled Tasks automation action."""

    status: ScheduledTaskAutomationActionStatus
    reason: str | None = None
    required_permissions: list[str] = Field(default_factory=list)


class ScheduledTaskAutomationCapability(BaseModel):
    """Capability report for a Scheduled Tasks automation family."""

    family: ScheduledTaskAutomationFamily
    family_availability: ScheduledTaskAutomationFamilyAvailability
    actions: dict[str, ScheduledTaskActionCapability]
    missing_dependencies: list[str] = Field(default_factory=list)
    related_capabilities: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    schema_version: str = "2026-06-09"


class ScheduledTaskAutomationCapabilitiesResponse(BaseModel):
    """Response for Scheduled Tasks automation capability discovery."""

    items: list[ScheduledTaskAutomationCapability] = Field(default_factory=list)


class ScheduledTaskPreviewCreateRequest(BaseModel):
    """Request to validate and persist a Scheduled Tasks automation preview."""

    mode: ScheduledTaskPreviewMode = "create"
    family: ScheduledTaskAutomationFamily
    definition_id: str | None = None
    definition_version: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(default_factory=dict)
    schedule: dict[str, Any] = Field(default_factory=dict)
    visibility_policy: dict[str, Any] = Field(default_factory=dict)
    notification_policy: dict[str, Any] = Field(default_factory=dict)
    approval_policy: dict[str, Any] = Field(default_factory=dict)


class ScheduledTaskPreviewResponse(BaseModel):
    """Persisted Scheduled Tasks automation preview response."""

    id: str
    owner_id: str | None = None
    mode: ScheduledTaskPreviewMode
    family: ScheduledTaskAutomationFamily
    definition_id: str | None = None
    definition_version: int | None = Field(default=None, ge=1)
    status: ScheduledTaskPreviewStatus
    payload_hash: str | None = None
    normalized_config: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    risk_class: str | None = None
    visibility_policy: dict[str, Any] = Field(default_factory=dict)
    schedule_preview: dict[str, Any] = Field(default_factory=dict)
    redaction_policy: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None
    created_by: str | None = None
    created_at: datetime | None = None
    consumed_at: datetime | None = None
    created_definition_id: str | None = None


class ScheduledTaskPreviewListResponse(BaseModel):
    """Paginated list of Scheduled Tasks automation previews."""

    items: list[ScheduledTaskPreviewResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)


class ScheduledTaskDefinitionCreateRequest(BaseModel):
    """Request to create a definition from a valid preview."""

    preview_id: str = Field(..., min_length=1)
    initial_lifecycle: ScheduledTaskDefinitionCreateLifecycle = "configured"


class ScheduledTaskDefinitionUpdateRequest(BaseModel):
    """Request to update a definition from a valid update preview."""

    preview_id: str = Field(..., min_length=1)


class ScheduledTaskDefinitionResponse(BaseModel):
    """Persisted Scheduled Tasks automation definition response."""

    id: str
    owner_id: str | None = None
    version: int = Field(default=1, ge=1)
    family: ScheduledTaskAutomationFamily
    name: str
    description: str | None = None
    lifecycle: ScheduledTaskDefinitionLifecycle
    health: ScheduledTaskDefinitionHealth
    disabled_lock_kind: ScheduledTaskDefinitionDisabledLockKind | None = None
    disabled_reason: str | None = None
    schedule: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    visibility_policy: dict[str, Any] = Field(default_factory=dict)
    notification_policy: dict[str, Any] = Field(default_factory=dict)
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    preview_id: str | None = None
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    archived_at: datetime | None = None


class ScheduledTaskDefinitionListResponse(BaseModel):
    """Paginated list of Scheduled Tasks automation definitions."""

    items: list[ScheduledTaskDefinitionResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)


class ScheduledTaskAuditEventResponse(BaseModel):
    """Audit event for a Scheduled Tasks automation definition."""

    id: str
    definition_id: str
    event_type: str
    actor: str | None = None
    summary: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    created_at: datetime | None = None
    request_id: str | None = None
    idempotency_key: str | None = None


class ScheduledTaskAuditListResponse(BaseModel):
    """Paginated audit list for a Scheduled Tasks automation definition."""

    items: list[ScheduledTaskAuditEventResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)


class ScheduledTaskDuplicateRequest(BaseModel):
    """Request to duplicate an existing Scheduled Tasks automation definition."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
