"""Pydantic schemas for API key management endpoints."""

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)

VALID_SCOPES = {"read", "write", "admin", "service"}


class APIKeyCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Optional display name for the key")
    description: Optional[str] = Field(None, description="Optional description")
    scope: Union[str, list[str]] = Field(
        "read",
        description="Permission scope(s): 'read', 'write', 'admin', 'service' or a list of these"
    )
    expires_in_days: Optional[int] = Field(365, ge=1, description="Days until expiration (None = never)")

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: Union[str, list[str]]) -> Union[str, list[str]]:
        scopes = [v] if isinstance(v, str) else v
        if not isinstance(scopes, list):
            raise TypeError("scope must be a string or list of strings")
        for scope in scopes:
            if not isinstance(scope, str):
                raise TypeError("scope entries must be strings")
        invalid = set(scopes) - VALID_SCOPES
        if invalid:
            invalid_list = ", ".join(sorted(invalid))
            raise ValueError(f"Invalid scope(s): {invalid_list}")
        return v


class APIKeyRotateRequest(BaseModel):
    expires_in_days: Optional[int] = Field(365, ge=1, description="Expiration for the new key")


class APIKeyMetadata(BaseModel):
    id: int
    key_prefix: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    scope: Union[str, list[str]]
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    usage_count: Optional[int] = 0
    last_used_at: Optional[datetime] = None
    last_used_ip: Optional[str] = None


class APIKeyCreateResponse(APIKeyMetadata):
    key: Optional[str] = Field(None, description="The actual API key (returned only at creation/rotation)")
    message: Optional[str] = None


class APIKeyRevokeResponse(BaseModel):
    message: str
    user_id: int
    key_id: int


class APIKeyUpdateRequest(BaseModel):
    rate_limit: Optional[int] = Field(None, description="Requests per minute for this key")
    allowed_ips: Optional[list[str]] = Field(None, description="Restrict usage to these IPs")


class APIKeyAuditEntry(BaseModel):
    id: int
    api_key_id: int
    action: str
    user_id: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Any] = None
    created_at: Optional[datetime] = None


class APIKeyAuditListResponse(BaseModel):
    key_id: int
    items: list[APIKeyAuditEntry]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: Optional[bool] = Field(default=None, description="Alias for pagination.has_more")
    next_offset: Optional[int] = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)
