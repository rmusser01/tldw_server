from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class UserProviderKeyUpsertRequest(BaseModel):
    provider: str = Field(..., description="Provider name (e.g., 'openai').")
    api_key: str = Field(..., description="Provider API key.")
    credential_fields: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider-specific credential fields (e.g., base_url).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional non-sensitive metadata tags.",
    )


class ProviderKeyTestRequest(BaseModel):
    provider: str = Field(..., description="Provider name (e.g., 'openai').")
    model: str | None = Field(
        default=None,
        description="Optional model override for the test call.",
    )


class UserProviderKeyResponse(BaseModel):
    provider: str
    status: Literal["stored"] = "stored"
    key_hint: str
    updated_at: datetime


class UserProviderKeyStatusItem(BaseModel):
    provider: str
    has_key: bool
    source: Literal["user", "team", "org", "server_default", "none", "disabled"]
    key_hint: str | None = None
    auth_source: Literal["api_key", "oauth"] | None = None
    last_used_at: datetime | None = None


class UserProviderKeysResponse(BaseModel):
    items: list[UserProviderKeyStatusItem]


class ProviderKeyTestResponse(BaseModel):
    provider: str
    status: Literal["valid"] = "valid"
    model: str | None = None


class SharedProviderKeyUpsertRequest(BaseModel):
    scope_type: Literal["org", "team"]
    scope_id: int
    provider: str
    api_key: str
    credential_fields: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SharedProviderKeyResponse(BaseModel):
    scope_type: Literal["org", "team"]
    scope_id: int
    provider: str
    status: Literal["stored"] = "stored"
    key_hint: str
    updated_at: datetime


class SharedProviderKeyTestRequest(BaseModel):
    scope_type: Literal["org", "team"]
    scope_id: int
    provider: str
    model: str | None = None


class SharedProviderKeyTestResponse(BaseModel):
    scope_type: Literal["org", "team"]
    scope_id: int
    provider: str
    status: Literal["valid"] = "valid"
    model: str | None = None


class SharedProviderKeyStatusItem(BaseModel):
    scope_type: Literal["org", "team"]
    scope_id: int
    provider: str
    key_hint: str | None = None
    last_used_at: datetime | None = None


class SharedProviderKeysResponse(BaseModel):
    items: list[SharedProviderKeyStatusItem]


class AdminUserKeyStatusItem(BaseModel):
    provider: str
    key_hint: str | None = None
    last_used_at: datetime | None = None
    allowed: bool


class AdminUserKeysResponse(BaseModel):
    user_id: int
    items: list[AdminUserKeyStatusItem]


class OpenAIOAuthAuthorizeRequest(BaseModel):
    credential_fields: dict[str, Any] | None = Field(
        default=None,
        description="Optional OpenAI credential fields (org_id/project_id).",
    )
    return_path: str | None = Field(
        default=None,
        description="Optional app-relative path used by frontend post-callback handling.",
    )


class OpenAIOAuthAuthorizeResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    auth_url: str
    auth_session_id: str
    expires_at: datetime


class OpenAIOAuthCallbackResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    status: Literal["stored"] = "stored"
    auth_source: Literal["oauth"] = "oauth"
    key_hint: str = "oauth"
    updated_at: datetime
    expires_at: datetime | None = None


class OpenAIOAuthStatusResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    connected: bool
    auth_source: Literal["api_key", "oauth", "none"] = "none"
    updated_at: datetime | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    scope: str | None = None


class OpenAIOAuthRefreshResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    status: Literal["refreshed"] = "refreshed"
    updated_at: datetime
    expires_at: datetime | None = None


class OpenAICredentialSourceSwitchRequest(BaseModel):
    auth_source: Literal["api_key", "oauth"] = Field(
        ...,
        description="Preferred OpenAI credential source to activate.",
    )


class OpenAICredentialSourceSwitchResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    auth_source: Literal["api_key", "oauth"]
    updated_at: datetime


class ByokValidationRunCreateRequest(BaseModel):
    """Request to create a shared authoritative BYOK validation run."""

    org_id: int | None = None
    provider: str | None = None


class ByokValidationRunItem(BaseModel):
    """Persisted BYOK validation run item returned to the admin UI."""

    id: str
    status: Literal["queued", "running", "complete", "failed"]
    org_id: int | None = None
    provider: str | None = None
    keys_checked: int | None = None
    valid_count: int | None = None
    invalid_count: int | None = None
    error_count: int | None = None
    requested_by_user_id: int | None = None
    requested_by_label: str | None = None
    job_id: str | None = None
    scope_summary: str
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ByokValidationRunListResponse(BaseModel):
    """Paginated list response for authoritative BYOK validation runs."""

    items: list[ByokValidationRunItem]
    total: int
    limit: int
    offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)
