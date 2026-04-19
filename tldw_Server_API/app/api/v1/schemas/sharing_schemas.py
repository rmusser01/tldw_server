# sharing_schemas.py
# Description: Pydantic schemas for workspace sharing and share tokens
"""
Sharing Schemas
---------------

Pydantic models for workspace sharing CRUD, share tokens, and admin operations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, default_offset_pagination_aliases


class ShareScopeType(str, Enum):
    TEAM = "team"
    ORG = "org"


class AccessLevel(str, Enum):
    VIEW_CHAT = "view_chat"
    VIEW_CHAT_ADD = "view_chat_add"
    FULL_EDIT = "full_edit"


class ResourceType(str, Enum):
    CHATBOOK = "chatbook"
    WORKSPACE = "workspace"
    PROTOTYPE_WORKSPACE = "prototype_workspace"


# ── Workspace Sharing ──

class ShareWorkspaceRequest(BaseModel):
    share_scope_type: ShareScopeType = Field(..., description="Share with team or org")
    share_scope_id: int = Field(..., description="ID of the team or org")
    access_level: AccessLevel = Field(AccessLevel.VIEW_CHAT, description="Access tier")
    allow_clone: bool = Field(True, description="Whether cloning is allowed")


class UpdateShareRequest(BaseModel):
    access_level: AccessLevel | None = Field(None, description="New access level")
    allow_clone: bool | None = Field(None, description="Update clone permission")


class ShareResponse(BaseModel):
    id: int
    workspace_id: str
    owner_user_id: int
    share_scope_type: str
    share_scope_id: int
    access_level: str
    allow_clone: bool
    created_by: int
    created_at: str | None = None
    updated_at: str | None = None
    revoked_at: str | None = None
    is_revoked: bool = False


class ShareListResponse(BaseModel):
    shares: list[ShareResponse]
    total: int


# ── Shared With Me ──

class SharedWithMeItem(BaseModel):
    share_id: int
    workspace_id: str
    workspace_name: str | None = None
    owner_user_id: int
    owner_username: str | None = None
    access_level: str
    allow_clone: bool
    shared_at: str | None = None


class SharedWithMeResponse(BaseModel):
    items: list[SharedWithMeItem]
    total: int


# ── Share Tokens ──

class CreateTokenRequest(BaseModel):
    resource_type: ResourceType = Field(..., description="Type of resource to share")
    resource_id: str = Field(..., description="ID of the resource")
    access_level: AccessLevel = Field(AccessLevel.VIEW_CHAT)
    allow_clone: bool = Field(True)
    password: str | None = Field(None, min_length=4, max_length=128, description="Optional password protection")
    max_uses: int | None = Field(None, ge=1, le=10000, description="Maximum number of uses")
    expires_at: str | None = Field(None, description="ISO 8601 expiry timestamp")


class TokenResponse(BaseModel):
    id: int
    token_prefix: str
    resource_type: str
    resource_id: str
    access_level: str
    allow_clone: bool
    is_password_protected: bool = False
    max_uses: int | None = None
    use_count: int = 0
    expires_at: str | None = None
    created_at: str | None = None
    revoked_at: str | None = None
    is_revoked: bool = False
    raw_token: str | None = Field(None, description="Only returned on creation")


class TokenListResponse(BaseModel):
    tokens: list[TokenResponse]
    total: int


# ── Public Token Access ──

class PublicSharePreview(BaseModel):
    resource_type: str
    resource_name: str | None = None
    resource_description: str | None = None
    is_password_protected: bool = False
    access_level: str


class VerifyPasswordRequest(BaseModel):
    password: str = Field(..., min_length=1)


class VerifyPasswordResponse(BaseModel):
    verified: bool
    session_token: str | None = None


class PrototypeLinkExchangeRequest(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=120)
    password: str | None = Field(None, min_length=1, max_length=128)


class PrototypeLinkExchangeResponse(BaseModel):
    shared_actor_id: str
    actor_type: Literal["external_collaborator"] = "external_collaborator"
    session_token: str
    runtime_policy_profile: str


# ── Clone ──

class CloneWorkspaceRequest(BaseModel):
    new_name: str | None = Field(None, max_length=255, description="Name for the cloned workspace")


class CloneWorkspaceResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Clone job created"


# ── Admin ──

class AdminShareListResponse(BaseModel):
    shares: list[ShareResponse]
    total: int
    offset: int
    limit: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)


class UpdateConfigRequest(BaseModel):
    config: dict[str, str] = Field(..., description="Key-value config pairs to update")
    scope_type: str = Field("global", description="Config scope: global, org, team")
    scope_id: int | None = Field(None, description="Scope ID for org/team")


class AuditEventResponse(BaseModel):
    id: int
    event_type: str
    actor_user_id: int | None = None
    resource_type: str
    resource_id: str
    owner_user_id: int
    share_id: int | None = None
    token_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: str | None = None


class AuditLogResponse(BaseModel):
    events: list[AuditEventResponse]
    total: int
    offset: int
    limit: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)


# ── Proxy responses for shared-with-me ──

class SharedWorkspaceSourceResponse(BaseModel):
    id: str
    workspace_id: str
    media_id: int | None = None
    title: str = ""
    source_type: str = "media"
    url: str | None = None
    position: int = 0
    added_at: str | None = None


class SharedMediaResponse(BaseModel):
    id: int
    title: str = ""
    url: str | None = None
    media_type: str | None = None
    content: str | None = None
    author: str | None = None
    ingestion_date: str | None = None


class SharedChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="Chat query")
    model: str | None = Field(None, description="LLM model override")
    api_name: str | None = Field(None, description="LLM provider override")
    system_message: str | None = Field(None, description="System message override")
