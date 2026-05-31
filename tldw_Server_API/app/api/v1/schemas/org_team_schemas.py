from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class OrganizationCreateRequest(BaseModel):
    name: str = Field(..., min_length=2)
    slug: str | None = None
    owner_user_id: int | None = None


class OrganizationResponse(BaseModel):
    id: int
    name: str
    slug: str | None = None
    owner_user_id: int | None = None
    is_active: bool | None = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class OrganizationListResponse(BaseModel):
    items: list[OrganizationResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def default_pagination_aliases(self) -> "OrganizationListResponse":
        return _default_offset_pagination_aliases(self)


class TeamCreateRequest(BaseModel):
    name: str = Field(..., min_length=2)
    slug: str | None = None
    description: str | None = None


class TeamResponse(BaseModel):
    id: int
    org_id: int
    name: str
    slug: str | None = None
    description: str | None = None
    is_active: bool | None = True
    created_at: str | None = None
    updated_at: str | None = None


class TeamMemberAddRequest(BaseModel):
    user_id: int
    role: str | None = Field("member", pattern=r"^(owner|admin|lead|member)$")


class TeamMemberResponse(BaseModel):
    team_id: int
    user_id: int
    role: str
    org_id: int | None = None


class TeamMemberRemoveResponse(BaseModel):
    team_id: int
    user_id: int
    removed: bool
    message: str


class TeamMemberRoleUpdateRequest(BaseModel):
    role: str = Field(..., pattern=r"^(owner|admin|lead|member)$")


class TeamMembershipItem(BaseModel):
    team_id: int
    org_id: int
    role: str
    team_name: str | None = None
    org_name: str | None = None


class VirtualKeyCreateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    expires_in_days: int | None = Field(30, ge=1)
    org_id: int | None = None
    team_id: int | None = None
    allowed_endpoints: list[str] | None = Field(default_factory=lambda: ["chat.completions", "embeddings"])
    allowed_providers: list[str] | None = None
    allowed_models: list[str] | None = None
    budget_day_tokens: int | None = None
    budget_month_tokens: int | None = None
    budget_day_usd: float | None = None
    budget_month_usd: float | None = None
    # Additional generic constraints stored in metadata for non-LLM enforcement
    allowed_methods: list[str] | None = None
    allowed_paths: list[str] | None = None
    max_calls: int | None = Field(None, ge=0)
    max_runs: int | None = Field(None, ge=0)


# ============================
# Organization membership schemas
# ============================

class OrgMemberAddRequest(BaseModel):
    user_id: int
    role: str | None = Field("member", pattern=r"^(owner|admin|lead|member)$")


class OrgMemberResponse(BaseModel):
    org_id: int
    user_id: int
    role: str


class OrgMemberRemoveResponse(BaseModel):
    org_id: int
    user_id: int
    removed: bool
    message: str


class OrgMemberRoleUpdateRequest(BaseModel):
    role: str = Field(..., pattern=r"^(owner|admin|lead|member)$")


class OrgMemberListItem(BaseModel):
    user_id: int
    role: str
    status: str | None = None
    added_at: str | None = None


class OrgMembershipItem(BaseModel):
    org_id: int
    role: str


# ============================
# Organization settings: Watchlists
# ============================

class OrganizationWatchlistsSettingsUpdate(BaseModel):
    require_include_default: bool | None = Field(
        default=None,
        description="When true, include-only gating is enabled by default for jobs in this organization",
    )


class OrganizationWatchlistsSettingsResponse(BaseModel):
    org_id: int
    require_include_default: bool | None = None


class OrganizationSTTSettingsUpdate(BaseModel):
    delete_audio_after_success: bool | None = None
    audio_retention_hours: float | None = Field(default=None, ge=0.0)
    redact_pii: bool | None = None
    allow_unredacted_partials: bool | None = None
    redact_categories: list[str] | None = None

    @field_validator("redact_categories")
    @classmethod
    def _normalize_redact_categories(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in value:
            item = str(raw).strip().lower()
            if not item or item in seen:
                continue
            normalized.append(item)
            seen.add(item)
        return normalized


class OrganizationSTTSettingsResponse(BaseModel):
    org_id: int
    delete_audio_after_success: bool
    audio_retention_hours: float
    redact_pii: bool
    allow_unredacted_partials: bool
    redact_categories: list[str] = Field(default_factory=list)


# ============================
# Self-service org schemas
# ============================

class OrgSelfCreateRequest(BaseModel):
    """Request to create a new organization (self-service)."""
    name: str = Field(..., min_length=2, max_length=100)
    slug: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", max_length=50)


class OrgUpdateRequest(BaseModel):
    """Request to update an organization."""
    name: str | None = Field(None, min_length=2, max_length=100)
    slug: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", max_length=50)


class OrgDetailResponse(BaseModel):
    """Detailed organization response with membership info."""
    id: int
    name: str
    slug: str | None = None
    owner_user_id: int | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
    member_count: int | None = None
    team_count: int | None = None
    user_role: str | None = None


class OwnershipTransferRequest(BaseModel):
    """Request to transfer org ownership."""
    new_owner_user_id: int = Field(..., description="User ID of the new owner")


class TeamListResponse(BaseModel):
    """List of teams."""
    items: list[TeamResponse]
    total: int


class TeamUpdateRequest(BaseModel):
    """Request to update a team."""
    name: str | None = Field(None, min_length=2, max_length=100)
    slug: str | None = Field(None, max_length=50)
    description: str | None = Field(None, max_length=500)


class TeamMemberListResponse(BaseModel):
    """List of team members."""
    items: list[TeamMemberResponse]
    total: int


# ============================
# Organization invite schemas
# ============================

class OrgInviteCreateRequest(BaseModel):
    """Request to create an organization invite."""
    team_id: int | None = Field(None, description="Optional team ID for team-specific invite")
    role_to_grant: str = Field("member", pattern=r"^(admin|lead|member)$")
    max_uses: int = Field(1, ge=1, le=1000)
    expiry_days: int = Field(7, ge=1, le=365)
    description: str | None = Field(None, max_length=500)
    allowed_email_domain: str | None = Field(
        None,
        pattern=r"^@?[A-Za-z0-9.-]+$",
        description="Restrict invite redemption to this email domain",
    )


class OrgInviteResponse(BaseModel):
    """Organization invite details."""
    id: int
    code: str
    org_id: int
    org_name: str | None = None
    team_id: int | None = None
    team_name: str | None = None
    role_to_grant: str
    max_uses: int
    uses_count: int
    is_active: bool
    expires_at: str | None = None
    created_at: str | None = None
    created_by: int | None = None
    description: str | None = None
    allowed_email_domain: str | None = None


class OrgInviteListResponse(BaseModel):
    """List of organization invites."""
    items: list[OrgInviteResponse]
    total: int
    limit: int
    offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class OrgInvitePreviewResponse(BaseModel):
    """Public preview of an invite (no auth required)."""
    org_name: str | None = None
    org_slug: str | None = None
    team_name: str | None = None
    role_to_grant: str
    is_valid: bool
    status: str
    message: str | None = None
    expires_at: str | None = None
    allowed_email_domain: str | None = None


class OrgInviteRedeemRequest(BaseModel):
    """Request to redeem an invite code."""
    code: str = Field(..., min_length=8)


class OrgInviteRedeemResponse(BaseModel):
    """Result of redeeming an invite."""
    success: bool
    org_id: int | None = None
    org_name: str | None = None
    team_id: int | None = None
    team_name: str | None = None
    role: str | None = None
    was_already_member: bool = False
    message: str | None = None


class OrgInviteAcceptRequest(BaseModel):
    """Request to accept an org-scoped registration code."""
    code: str = Field(..., min_length=8)


class OrgInviteAcceptResponse(BaseModel):
    """Result of accepting an org-scoped registration code."""
    success: bool
    org_id: int | None = None
    team_id: int | None = None
    org_role: str | None = None
    was_already_member: bool = False
    message: str | None = None
