"""Bounded public schemas for the canonical admin-webhook control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from tldw_Server_API.app.core.Admin_Webhooks.catalog import EVENT_CATALOG

_EVENT_TYPES = frozenset(item.event_type for item in EVENT_CATALOG)
_FAKE_SIGNING_SECRET = "whsec_" + ("0" * 64)


def _validate_event_types(values: list[str]) -> list[str]:
    """Reject wildcard, duplicate, empty, and unknown subscriptions."""
    if len(set(values)) != len(values) or any(value not in _EVENT_TYPES for value in values):
        raise ValueError("event_types must contain unique catalog events")
    return values


class WebhookCreateRequest(BaseModel):
    """Create one inactive registration with a server-generated secret."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "url": "https://receiver.example/hooks/private",
                    "event_types": ["incident.created"],
                    "description": "Incident receiver",
                    "timeout_seconds": 10,
                }
            ]
        },
    )

    url: str = Field(min_length=1, max_length=2_048)
    event_types: list[str] = Field(min_length=1, max_length=len(EVENT_CATALOG))
    description: str = Field(default="", max_length=500)
    timeout_seconds: int = Field(default=10, ge=1, le=30)

    _events_are_catalog_entries = field_validator("event_types")(_validate_event_types)


class WebhookPatchRequest(BaseModel):
    """Patch caller-visible registration metadata; null fields are forbidden."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"description": "Primary incident receiver"}]},
    )

    description: str | None = Field(default=None, max_length=500)
    url: str | None = Field(default=None, min_length=1, max_length=2_048)
    event_types: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=len(EVENT_CATALOG),
    )
    active: bool | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=30)

    @field_validator(
        "description",
        "url",
        "event_types",
        "active",
        "timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("patch fields cannot be null")
        return value

    @field_validator("event_types")
    @classmethod
    def _validate_optional_event_types(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return _validate_event_types(values)

    @model_validator(mode="after")
    def _require_one_field(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one patch field is required")
        return self


class AdminWebhookRegistrationResponse(BaseModel):
    """Redacted registration metadata returned by ordinary operations."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: int = Field(ge=1)
    description: str = Field(max_length=500)
    target_display: str = Field(min_length=1, max_length=2_048)
    target_hostname: str = Field(min_length=1, max_length=253)
    event_types: list[str] = Field(min_length=1, max_length=len(EVENT_CATALOG))
    active: bool
    timeout_seconds: int = Field(ge=1, le=30)
    revision: int = Field(ge=1)
    delivery_config_version: int = Field(ge=1)
    secret_version: int = Field(ge=1)
    secret_rotation_required: bool
    created_by: int | None = Field(default=None, ge=1)
    updated_by: int | None = Field(default=None, ge=1)
    created_at: datetime
    updated_at: datetime

    _events_are_catalog_entries = field_validator("event_types")(_validate_event_types)


class WebhookSecretResponse(BaseModel):
    """One-time create or rotation result containing the generated secret."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "registration": {
                        "id": 41,
                        "description": "Incident receiver",
                        "target_display": "https://receiver.example",
                        "target_hostname": "receiver.example",
                        "event_types": ["incident.created"],
                        "active": False,
                        "timeout_seconds": 10,
                        "revision": 1,
                        "delivery_config_version": 1,
                        "secret_version": 1,
                        "secret_rotation_required": False,
                        "created_by": 7,
                        "updated_by": 7,
                        "created_at": "2026-08-22T12:00:00Z",
                        "updated_at": "2026-08-22T12:00:00Z",
                    },
                    "signing_secret": _FAKE_SIGNING_SECRET,
                    "replayed": False,
                }
            ]
        },
    )

    registration: AdminWebhookRegistrationResponse
    signing_secret: str = Field(pattern=r"^whsec_[0-9a-f]{64}$")
    replayed: bool


class WebhookDeleteResponse(BaseModel):
    """Acknowledgement for a successful registration tombstone."""

    model_config = ConfigDict(extra="forbid")

    deleted: Literal[True] = True
    id: int = Field(ge=1)


class WebhookListResponse(BaseModel):
    """Deterministic offset page ordered by registration ID descending."""

    model_config = ConfigDict(extra="forbid")

    items: list[AdminWebhookRegistrationResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0, le=1_000)


class WebhookCatalogItemResponse(BaseModel):
    """One supported event from the immutable server catalog."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    event_type: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)


class WebhookCatalogResponse(BaseModel):
    """Versioned event catalog and effective registration limits."""

    model_config = ConfigDict(extra="forbid")

    api_version: str = Field(min_length=1, max_length=32)
    events: list[WebhookCatalogItemResponse] = Field(
        min_length=1,
        max_length=len(EVENT_CATALOG),
    )
    registration_limit: int = Field(ge=1, le=1_000)
    active_limit: int = Field(ge=1, le=1_000)


class WebhookLimitsResponse(BaseModel):
    """Configured and current bounded registration counts."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    registrations: int = Field(ge=1, le=1_000)
    active_registrations: int = Field(ge=1, le=1_000)
    current_registrations: int = Field(ge=0)
    current_active_registrations: int = Field(ge=0)
    registrations_over_limit: bool
    active_registrations_over_limit: bool


class WebhookMigrationStatusResponse(BaseModel):
    """Sanitized migration and structural rollback readiness."""

    model_config = ConfigDict(extra="forbid")

    phase: str = Field(min_length=1, max_length=64)
    imported_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    secret_rotation_required_count: int = Field(ge=0)
    legacy_file_restore_permitted: bool
    rollback_window_expires_at: datetime | None = None


class AdminWebhookStatusResponse(BaseModel):
    """Explicit canonical-mode readiness without artifact or credential data."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "migrate", "on"]
    route_selection: Literal["canonical", "legacy"]
    schema_ready: bool
    key_state: str = Field(min_length=1, max_length=128)
    delivery_capability_ready: bool
    limits: WebhookLimitsResponse
    migration: WebhookMigrationStatusResponse


class WebhookErrorDetail(BaseModel):
    """Stable, bounded error detail without rejected caller data."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    message: str = Field(min_length=1, max_length=200)
    request_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")


class WebhookErrorResponse(BaseModel):
    """Canonical error envelope used by every expected route failure."""

    model_config = ConfigDict(extra="forbid")

    error: WebhookErrorDetail
