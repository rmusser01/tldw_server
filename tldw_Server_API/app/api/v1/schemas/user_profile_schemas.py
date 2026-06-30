# user_profile_schemas.py
"""Pydantic schemas for user profile endpoints."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta


class UserProfileCatalogEntrySchema(BaseModel):
    """Schema for a single user profile catalog entry."""

    key: str = Field(..., description="Catalog key identifier")
    label: str = Field(..., description="Human-readable label")
    description: Optional[str] = Field(None, description="Description for UI/help text")
    type: str = Field(..., description="Data type for the value")
    enum: Optional[list[Any]] = Field(None, description="Allowed values for enum types")
    minimum: Optional[float] = Field(None, description="Minimum numeric value")
    maximum: Optional[float] = Field(None, description="Maximum numeric value")
    default: Optional[Any] = Field(None, description="Default value when unset")
    editable_by: list[str] = Field(default_factory=list, description="Roles allowed to edit")
    sensitivity: str = Field(..., description="Sensitivity level: public|internal|secret")
    ui: Optional[str] = Field(None, description="UI hint for rendering")
    deprecated: bool = Field(False, description="Whether the key is deprecated")


class UserProfileCatalogResponse(BaseModel):
    """Catalog response for user profile configuration keys."""

    version: str = Field(..., description="Catalog version")
    updated_at: datetime = Field(..., description="Last update timestamp")
    entries: list[UserProfileCatalogEntrySchema] = Field(default_factory=list)


class UserProfileIdentity(BaseModel):
    """Identity information for a profile response."""

    id: int = Field(..., description="User ID")
    uuid: Optional[str] = Field(None, description="User UUID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: str = Field(..., description="User role")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verification status")
    is_locked: Optional[bool] = Field(None, description="Account locked status")
    created_at: datetime = Field(..., description="Account creation time")
    last_login: Optional[datetime] = Field(None, description="Last login time")


class UserProfileOrgMembership(BaseModel):
    """Organization membership entry."""

    org_id: int = Field(..., description="Organization ID")
    role: str = Field(..., description="Membership role")
    policy_summary: Optional[dict[str, Any]] = Field(
        None,
        description="Organization policy summary",
    )


class UserProfileTeamMembership(BaseModel):
    """Team membership entry."""

    team_id: int = Field(..., description="Team ID")
    role: str = Field(..., description="Membership role")
    org_id: Optional[int] = Field(None, description="Parent organization ID")
    policy_summary: Optional[dict[str, Any]] = Field(
        None,
        description="Team policy summary",
    )


class UserProfileMemberships(BaseModel):
    """Memberships section of the profile response."""

    orgs: list[UserProfileOrgMembership] = Field(default_factory=list)
    teams: list[UserProfileTeamMembership] = Field(default_factory=list)


class UserProfileApiKeySummary(BaseModel):
    """API key metadata summary."""

    id: int = Field(..., description="API key ID")
    name: Optional[str] = Field(None, description="Key name")
    last_used_at: Optional[datetime] = Field(None, description="Last usage time")


class UserProfileByokSummary(BaseModel):
    """BYOK key summary."""

    provider: str = Field(..., description="Provider name")
    has_key: bool = Field(..., description="Whether a key is configured")


class UserProfileSecurity(BaseModel):
    """Security section of the profile response."""

    mfa_enabled: bool = Field(..., description="Whether MFA is enabled")
    active_sessions: int = Field(..., description="Active session count")
    api_keys: list[UserProfileApiKeySummary] = Field(default_factory=list)
    byok_keys: list[UserProfileByokSummary] = Field(default_factory=list)


class UserProfileQuotas(BaseModel):
    """Quota and usage section of the profile response."""

    storage_quota_mb: int = Field(..., description="Storage quota in MB")
    storage_used_mb: float = Field(..., description="Storage used in MB")
    audio: Optional[dict[str, Any]] = Field(
        None,
        description="Audio quota/usage summary (daily minutes, concurrency, file size)",
    )
    evaluations: Optional[dict[str, Any]] = Field(
        None,
        description="Evaluations quota/usage summary",
    )
    prompt_studio: Optional[dict[str, Any]] = Field(
        None,
        description="Prompt Studio quota summary",
    )


class UserProfileRawOverrideEntry(BaseModel):
    """Raw stored override entry (masked for secrets)."""

    key: str = Field(..., description="Catalog key")
    value: Optional[Any] = Field(None, description="Stored override value (masked if secret)")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    updated_by: Optional[int] = Field(None, description="Updater user ID")
    masked: bool = Field(False, description="Whether the value is masked")
    hint: Optional[str] = Field(None, description="Optional masked value hint")
    org_id: Optional[int] = Field(None, description="Organization ID (org overrides)")
    team_id: Optional[int] = Field(None, description="Team ID (team overrides)")


class UserProfileRawOverrides(BaseModel):
    """Raw overrides grouped by scope."""

    user: list[UserProfileRawOverrideEntry] = Field(default_factory=list)
    orgs: list[UserProfileRawOverrideEntry] = Field(default_factory=list)
    teams: list[UserProfileRawOverrideEntry] = Field(default_factory=list)


class UserProfileResponse(BaseModel):
    """Unified user profile response."""

    profile_version: datetime = Field(..., description="Profile version timestamp")
    catalog_version: str = Field(..., description="Catalog version")
    user: Optional[UserProfileIdentity] = None
    memberships: Optional[UserProfileMemberships] = None
    security: Optional[UserProfileSecurity] = None
    quotas: Optional[UserProfileQuotas] = None
    preferences: Optional[dict[str, Any]] = None
    effective_config: Optional[dict[str, Any]] = None
    raw_overrides: Optional[UserProfileRawOverrides] = None
    section_errors: Optional[dict[str, str]] = None


class UserProfileUpdateEntry(BaseModel):
    """Single profile update entry."""

    key: str = Field(..., description="Catalog key to update")
    value: Optional[Any] = Field(None, description="New value (null clears override)")


class UserProfileUpdateRequest(BaseModel):
    """Profile update request payload."""

    updates: list[UserProfileUpdateEntry] = Field(default_factory=list)
    profile_version: Optional[datetime] = Field(
        None,
        description="Optional optimistic lock token from the last profile read",
    )
    dry_run: bool = Field(False, description="Validate only, do not persist")


class UserProfileUpdateError(BaseModel):
    """Per-key update error entry."""

    key: str = Field(..., description="Catalog key")
    message: str = Field(..., description="Reason for skipping or failure")


class UserProfileBulkUpdateDiff(BaseModel):
    """Diff summary for a bulk update entry."""

    key: str = Field(..., description="Catalog key")
    before: Optional[Any] = Field(None, description="Value before update")
    after: Optional[Any] = Field(None, description="Value after update")


class UserProfileUpdateResponse(BaseModel):
    """Profile update response payload."""

    profile_version: datetime = Field(..., description="Profile version timestamp")
    applied: list[str] = Field(default_factory=list, description="Keys applied")
    skipped: list[UserProfileUpdateError] = Field(default_factory=list, description="Skipped keys")


class UserProfileErrorDetail(BaseModel):
    """Structured error entry for profile update failures."""

    key: str = Field(..., description="Catalog key or field identifier")
    message: str = Field(..., description="Error message")


class UserProfileErrorResponse(BaseModel):
    """Structured error response for profile update endpoints."""

    error_code: str = Field(..., description="Machine-readable error identifier")
    detail: str = Field(..., description="Human-readable error detail")
    errors: list[UserProfileErrorDetail] = Field(default_factory=list)


class UserProfileBulkUpdateRequest(BaseModel):
    """Bulk update payload for admin profile changes."""

    updates: list[UserProfileUpdateEntry] = Field(default_factory=list)
    dry_run: bool = Field(False, description="Validate only, do not persist")
    confirm: bool = Field(False, description="Confirm large updates past the threshold")
    user_ids: Optional[list[int]] = Field(None, description="Explicit list of user IDs to update")
    org_id: Optional[int] = Field(None, description="Restrict to a specific organization")
    team_id: Optional[int] = Field(None, description="Restrict to a specific team")
    role: Optional[str] = Field(None, description="Filter by role")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    search: Optional[str] = Field(None, description="Search username/email")


class UserProfileBulkUpdateUserResult(BaseModel):
    """Per-user result for bulk updates."""

    user_id: int = Field(..., description="Target user ID")
    profile_version: Optional[datetime] = Field(None, description="Resulting profile version")
    applied: list[str] = Field(default_factory=list, description="Keys applied")
    skipped: list[UserProfileUpdateError] = Field(default_factory=list, description="Skipped keys")
    diffs: list[UserProfileBulkUpdateDiff] = Field(
        default_factory=list,
        description="Diff summary for applied updates",
    )
    error: Optional[str] = Field(None, description="Error message for failed updates")


class UserProfileBulkUpdateResponse(BaseModel):
    """Bulk update response payload."""

    total_targets: int = Field(..., description="Total users targeted")
    updated: int = Field(..., description="Users with at least one applied update")
    skipped: int = Field(..., description="Users with no applied updates")
    failed: int = Field(..., description="Users that failed validation or scope checks")
    dry_run: bool = Field(..., description="Whether this was a dry run")
    results: list[UserProfileBulkUpdateUserResult] = Field(default_factory=list)


class UserProfileBatchResponse(BaseModel):
    """Batch profile summary response payload."""

    profiles: list[UserProfileResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total profiles matching the filters")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total page count")
    pagination: PagePaginationMeta
