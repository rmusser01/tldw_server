# admin_schemas.py
# Description: Pydantic schemas for admin endpoints
from __future__ import annotations

from datetime import date, datetime

# Imports
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, NonNegativeInt, SecretStr, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, PagePaginationMeta
from tldw_Server_API.app.core.Security.egress import evaluate_url_policy


def _default_offset_pagination_aliases(response):
    if hasattr(response, "limit") and response.limit is None:
        response.limit = response.pagination.limit
    if hasattr(response, "offset") and response.offset is None:
        response.offset = response.pagination.offset
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


def _blank_string_to_none(value: Any) -> Any:
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if isinstance(value, str) and not value.strip():
        return None
    return value


def unwrap_optional_secret(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        raw_value = value.get_secret_value()
    else:
        raw_value = str(value)
    normalized = raw_value.strip()
    return normalized or None


def validate_admin_webhook_url(value: str) -> str:
    """Enforce public http(s) webhook targets allowed by the egress policy."""
    stripped = value.strip()
    parsed = urlsplit(stripped)
    if parsed.username or parsed.password:
        raise ValueError("Webhook URL must not include embedded credentials")

    policy = evaluate_url_policy(stripped)
    if not policy.allowed:
        reason = policy.reason or "blocked by egress policy"
        raise ValueError(f"Webhook URL is not allowed: {reason}")
    return stripped

#######################################################################################################################
#
# User Management Schemas

class UserUpdateRequest(BaseModel):
    """Request to update user information"""
    email: EmailStr | None = None
    role: str | None = Field(None, pattern="^(user|admin|service)$")
    is_active: bool | None = None
    is_verified: bool | None = None
    is_locked: bool | None = None
    storage_quota_mb: int | None = Field(None, ge=100)
    reason: str | None = Field(default=None, min_length=8, max_length=500)
    admin_password: SecretStr | None = Field(default=None, max_length=128, repr=False)
    admin_reauth_token: SecretStr | None = Field(default=None, max_length=4096, repr=False)

    @field_validator("admin_password", "admin_reauth_token", mode="before")
    @classmethod
    def normalize_blank_admin_password(cls, value: Any) -> Any:
        return _blank_string_to_none(value)

    model_config = ConfigDict(from_attributes=True)


class AdminPrivilegedActionRequest(BaseModel):
    """Request payload for privileged admin actions."""

    reason: str = Field(..., min_length=8, max_length=500)
    admin_password: SecretStr | None = Field(default=None, max_length=128, repr=False)
    admin_reauth_token: SecretStr | None = Field(default=None, max_length=4096, repr=False)

    @field_validator("admin_password", "admin_reauth_token", mode="before")
    @classmethod
    def normalize_blank_admin_password(cls, value: Any) -> Any:
        return _blank_string_to_none(value)

    model_config = ConfigDict(from_attributes=True)


class AdminPasswordResetRequest(AdminPrivilegedActionRequest):
    """Request payload for admin-initiated user password reset."""

    temporary_password: str = Field(..., min_length=10, max_length=128)
    force_password_change: bool = True


class AdminPasswordResetResponse(BaseModel):
    """Response payload for admin-initiated user password reset."""

    user_id: int
    force_password_change: bool
    message: str

    model_config = ConfigDict(from_attributes=True)


class AdminMfaRequirementRequest(AdminPrivilegedActionRequest):
    """Request payload for admin-managed MFA requirement flag."""

    require_mfa: bool = True


class AdminMfaRequirementResponse(BaseModel):
    """Response payload for admin-managed MFA requirement flag."""

    user_id: int
    require_mfa: bool
    message: str

    model_config = ConfigDict(from_attributes=True)


class AdminUserCreateRequest(BaseModel):
    """Request to create a user as an admin."""
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
    )
    email: EmailStr
    password: str = Field(..., min_length=10, max_length=128)
    role: str = Field("user", pattern="^(user|admin|service)$")
    is_active: bool = True
    is_verified: bool = True
    storage_quota_mb: int | None = Field(None, ge=100)

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        reserved = {"admin", "root", "system", "api", "null", "undefined"}
        normalized = v.strip()
        if normalized.lower() in reserved:
            raise ValueError("This username is reserved")
        return normalized.lower()

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: str) -> str:
        return v.lower()


class UserSummary(BaseModel):
    """Summary information about a user"""
    id: int
    uuid: UUID
    username: str
    email: str
    role: str
    is_active: bool
    is_verified: bool
    mfa_enabled: bool = False
    created_at: datetime
    last_login: datetime | None = None
    storage_quota_mb: int
    storage_used_mb: float

    model_config = ConfigDict(from_attributes=True)


class UserDetailResponse(BaseModel):
    """Detailed user information for admin endpoints."""

    id: int
    uuid: UUID | None = None
    username: str
    email: str
    role: str | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None
    is_verified: bool | None = None
    email_verified: bool | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login: datetime | None = None
    storage_quota_mb: int | None = None
    storage_used_mb: float | None = None
    is_locked: bool | None = None
    failed_login_attempts: int | None = None
    locked_until: datetime | None = None
    metadata: Any | None = None

    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """Response for user list endpoint"""
    users: list[UserSummary]
    total: int
    page: int
    limit: int
    pages: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class UserQuotaUpdateRequest(BaseModel):
    """Request to update user storage quota"""
    storage_quota_mb: int = Field(..., ge=100, le=1000000)  # 100MB to 1TB

    model_config = ConfigDict(from_attributes=True)


class AdminStartupWarningItem(BaseModel):
    """One current-process startup warning record."""

    component: str
    severity: str
    startup_action: str
    code: str
    summary: str
    remediation: str
    details: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminStartupWarningSummary(BaseModel):
    """Grouped summary for current-process startup warnings."""

    total: int
    by_component: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class AdminStartupWarningsResponse(BaseModel):
    """Response for the current-process startup warnings admin endpoint."""

    startup_id: str
    scope: Literal["current_process"] = "current_process"
    warnings_present: bool
    blocking_present: bool
    summary: AdminStartupWarningSummary
    items: list[AdminStartupWarningItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Registration Code Schemas

class RegistrationCodeRequest(BaseModel):
    """Request to create a registration code"""
    max_uses: int = Field(1, ge=1, le=100)
    expiry_days: int = Field(7, ge=1, le=365)
    role_to_grant: str = Field("user", pattern="^(user|admin|service)$")
    allowed_email_domain: str | None = Field(
        None,
        pattern=r"^@?[A-Za-z0-9.-]+$",
    )
    metadata: dict[str, Any] | None = None
    org_id: int | None = Field(None, ge=1)
    org_role: str | None = Field(None, pattern=r"^(owner|admin|lead|member)$")
    team_id: int | None = Field(None, ge=1)

    model_config = ConfigDict(from_attributes=True)


class RegistrationCodeResponse(BaseModel):
    """Response with registration code details"""
    id: int
    code: str
    max_uses: int
    times_used: int
    expires_at: datetime
    created_at: datetime
    created_by: int | None = None
    role_to_grant: str
    allowed_email_domain: str | None = None
    org_id: int | None = None
    org_role: str | None = None
    team_id: int | None = None
    org_name: str | None = None
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None
    is_valid: bool | None = None

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Calculate is_valid if not provided
        if self.is_valid is None:
            is_active = True if self.is_active is None else bool(self.is_active)
            self.is_valid = is_active and self.times_used < self.max_uses and self.expires_at > datetime.utcnow()


class RegistrationCodeListResponse(BaseModel):
    """Response for registration code list"""
    codes: list[RegistrationCodeResponse]

    model_config = ConfigDict(from_attributes=True)


class RegistrationSettingsResponse(BaseModel):
    """Registration configuration status for admin surfaces."""
    enable_registration: bool
    require_registration_code: bool
    auth_mode: str | None = None
    profile: str | None = None
    self_registration_allowed: bool | None = None

    model_config = ConfigDict(from_attributes=True)


class RegistrationSettingsUpdateRequest(BaseModel):
    """Request to update registration settings."""
    enable_registration: bool | None = None
    require_registration_code: bool | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# LLM Provider Overrides Schemas

class LLMProviderOverrideRequest(BaseModel):
    """Request to upsert LLM provider overrides."""
    is_enabled: bool | None = None
    allowed_models: list[str] | None = None
    config: dict[str, Any] | None = None
    api_key: str | None = None
    credential_fields: dict[str, Any] | None = None
    clear_api_key: bool | None = None

    model_config = ConfigDict(from_attributes=True)


class LLMProviderOverrideResponse(BaseModel):
    """Response payload for LLM provider overrides."""
    provider: str
    is_enabled: bool | None = None
    allowed_models: list[str] | None = None
    config: dict[str, Any] | None = None
    credential_fields: dict[str, Any] | None = None
    has_api_key: bool = False
    api_key_hint: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class LLMProviderOverrideListResponse(BaseModel):
    """Response payload for listing LLM provider overrides."""
    items: list[LLMProviderOverrideResponse]

    model_config = ConfigDict(from_attributes=True)


class LLMProviderTestRequest(BaseModel):
    """Request to test LLM provider connectivity."""
    provider: str
    model: str | None = None
    api_key: str | None = None
    credential_fields: dict[str, Any] | None = None
    use_override: bool = True

    model_config = ConfigDict(from_attributes=True)


class LLMProviderTestResponse(BaseModel):
    """Response payload for LLM provider test results."""
    provider: str
    status: str
    model: str | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# System Statistics Schemas

class UserStats(BaseModel):
    """User statistics"""
    total: int
    active: int
    verified: int
    admins: int
    new_last_30d: int

    model_config = ConfigDict(from_attributes=True)


class StorageStats(BaseModel):
    """Storage statistics"""
    total_used_mb: float
    total_quota_mb: float
    average_used_mb: float
    max_used_mb: float

    model_config = ConfigDict(from_attributes=True)


class SessionStats(BaseModel):
    """Session statistics"""
    active: int
    unique_users: int

    model_config = ConfigDict(from_attributes=True)


class TokensTodayStats(BaseModel):
    """Token usage stats for today"""
    prompt: int = 0
    completion: int = 0
    total: int = 0

    model_config = ConfigDict(from_attributes=True)


class SystemStatsResponse(BaseModel):
    """System statistics response"""
    users: UserStats
    storage: StorageStats
    sessions: SessionStats
    active_acp_sessions: int | None = None
    tokens_today: TokensTodayStats | None = None
    mcp_invocations_today: int | None = None

    model_config = ConfigDict(from_attributes=True)


class ActivityPoint(BaseModel):
    """Activity point for dashboard charts."""
    date: date
    bucket_start: datetime | None = None
    requests: NonNegativeInt
    users: NonNegativeInt

    model_config = ConfigDict(from_attributes=True)


class ActivitySummaryResponse(BaseModel):
    """Dashboard activity summary response."""
    days: int = Field(..., ge=0)
    granularity: Literal["hour", "day"] = "day"
    points: list[ActivityPoint]
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminPermissionDebugRequest(BaseModel):
    """Request payload for permission resolution debug endpoint."""

    user_id: int = Field(..., ge=1)

    model_config = ConfigDict(from_attributes=True)


class AdminPermissionDebugResponse(BaseModel):
    """Permission resolution debug output."""

    user_id: int
    roles: list[str]
    effective_permissions: list[str]
    permission_count: int | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminTokenDecodeRequest(BaseModel):
    """Request payload for token decode debug endpoint."""

    token: SecretStr = Field(..., repr=False)

    model_config = ConfigDict(from_attributes=True)


class AdminTokenDecodeResponse(BaseModel):
    """Token decode debug output without signature validation."""

    decoded: bool
    signature_verified: bool = False
    header: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    expired: bool | None = None
    expires_at: datetime | None = None
    issuer: str | None = None
    subject: str | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Security Alert Schemas

class SecurityAlertSinkStatus(BaseModel):
    """Represents the status of an individual security alert sink."""
    sink: str
    configured: bool
    min_severity: str | None = None
    last_status: bool | None = None
    last_error: str | None = None
    backoff_until: datetime | None = None


class SecurityAlertStatusResponse(BaseModel):
    """Aggregated security alert configuration and health."""
    enabled: bool
    min_severity: str
    last_dispatch_time: datetime | None
    last_dispatch_success: bool | None
    last_dispatch_error: str | None = None
    dispatch_count: int
    last_validation_time: datetime | None
    validation_errors: list[str] | None = None
    sinks: list[SecurityAlertSinkStatus]
    health: str

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Audit Log Schemas

class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    id: int
    user_id: int | None = None
    username: str | None = None
    action: str
    resource: str | None = None
    details: Any | None = None
    ip_address: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditLogResponse(BaseModel):
    """Response for audit log endpoint"""
    entries: list[AuditLogEntry]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Error Breakdown Schemas

class ErrorBreakdownItem(BaseModel):
    """A single row in the error breakdown aggregation."""
    endpoint: str
    status_code: int
    count: int
    last_occurred: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ErrorBreakdownResponse(BaseModel):
    """Aggregated error breakdown over a recent period."""
    items: list[ErrorBreakdownItem]
    total_errors: int
    period: str = "24h"

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Rate Limit Summary Schemas

class RateLimitThrottledEntity(BaseModel):
    """A frequently throttled user/IP/entity."""
    entity: str
    rejections: int
    last_rejected_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class RateLimitPolicyHeadroom(BaseModel):
    """Headroom utilization for a single policy."""
    policy_id: str
    resource_type: str | None = None
    scope: str | None = None
    total_decisions: int = 0
    total_denials: int = 0
    utilization_pct: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class RateLimitSummaryResponse(BaseModel):
    """Aggregated rate limit summary for a period."""
    total_throttle_events: int
    period: str = "24h"
    top_throttled_entities: list[RateLimitThrottledEntity]
    policy_headroom: list[RateLimitPolicyHeadroom]

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Data Ops Schemas (Backups, Retention, Exports)

class BackupItem(BaseModel):
    """Metadata for a backup artifact."""
    id: str
    dataset: str
    user_id: int | None = None
    status: str = "ready"
    size_bytes: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BackupListResponse(BaseModel):
    """Response for backup listing."""
    items: list[BackupItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class BackupCreateRequest(BaseModel):
    """Request to create a backup snapshot."""
    dataset: str
    user_id: int | None = None
    backup_type: str | None = Field("full", pattern="^(full|incremental)$")
    max_backups: int | None = Field(None, ge=1, le=1000)

    model_config = ConfigDict(from_attributes=True)


class BackupCreateResponse(BaseModel):
    """Response for backup creation."""
    item: BackupItem

    model_config = ConfigDict(from_attributes=True)


class BackupScheduleItem(BaseModel):
    """Authoritative backup schedule record for admin surfaces."""

    id: str
    dataset: str
    target_user_id: int | None = None
    frequency: Literal["daily", "weekly", "monthly"]
    time_of_day: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    timezone: str
    anchor_day_of_week: int | None = Field(default=None, ge=0, le=6)
    anchor_day_of_month: int | None = Field(default=None, ge=1, le=31)
    retention_count: int = Field(..., ge=1, le=1000)
    is_paused: bool = False
    schedule_description: str
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_status: str | None = None
    last_job_id: str | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class BackupScheduleListResponse(BaseModel):
    """Response for backup schedule listing."""

    items: list[BackupScheduleItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class BackupScheduleCreateRequest(BaseModel):
    """Request to create a platform-owned backup schedule."""

    dataset: str
    target_user_id: int | None = Field(default=None, ge=1)
    frequency: Literal["daily", "weekly", "monthly"]
    time_of_day: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    timezone: str | None = None
    retention_count: int = Field(..., ge=1, le=1000)

    model_config = ConfigDict(from_attributes=True)


class BackupScheduleUpdateRequest(BaseModel):
    """Request to update mutable backup schedule fields."""

    frequency: Literal["daily", "weekly", "monthly"] | None = None
    time_of_day: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    timezone: str | None = None
    retention_count: int | None = Field(default=None, ge=1, le=1000)

    model_config = ConfigDict(from_attributes=True)


class BackupScheduleMutationResponse(BaseModel):
    """Response for backup schedule create/update/pause/resume/delete operations."""

    status: str
    item: BackupScheduleItem

    model_config = ConfigDict(from_attributes=True)


class BackupRestoreRequest(BaseModel):
    """Request to restore a backup snapshot."""
    dataset: str
    user_id: int | None = None
    confirm: bool = False

    model_config = ConfigDict(from_attributes=True)


class BackupRestoreResponse(BaseModel):
    """Response for backup restore."""
    status: str
    message: str

    model_config = ConfigDict(from_attributes=True)


class MaintenanceRotationRunItem(BaseModel):
    """Authoritative maintenance rotation run record."""

    id: str
    mode: Literal["dry_run", "execute"]
    status: Literal["queued", "running", "complete", "failed"]
    domain: str | None = None
    queue: str | None = None
    job_type: str | None = None
    fields_json: str
    limit: int | None = Field(default=None, ge=1, le=100000)
    affected_count: int | None = None
    requested_by_user_id: int | None = None
    requested_by_label: str | None = None
    confirmation_recorded: bool = False
    job_id: str | None = None
    scope_summary: str
    key_source: str
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class MaintenanceRotationRunListResponse(BaseModel):
    """Response for maintenance rotation run listing."""

    items: list[MaintenanceRotationRunItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class MaintenanceRotationRunCreateRequest(BaseModel):
    """Request to create an authoritative maintenance rotation run."""

    mode: Literal["dry_run", "execute"]
    domain: str | None = None
    queue: str | None = None
    job_type: str | None = None
    fields: list[str] = Field(default_factory=lambda: ["payload", "result"])
    limit: int = Field(default=1000, ge=1, le=100000)
    confirmed: bool = False

    @field_validator("domain", "queue", "job_type", mode="before")
    @classmethod
    def normalize_optional_scope_text(cls, value: Any) -> Any:
        return _blank_string_to_none(value)

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, value: Any) -> Any:
        if value is None:
            return ["payload", "result"]
        if not isinstance(value, list):
            raise ValueError("fields must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError("fields must contain strings")
            item = entry.strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        if not normalized:
            raise ValueError("fields must contain at least one value")
        return normalized

    model_config = ConfigDict(from_attributes=True)


class MaintenanceRotationRunCreateResponse(BaseModel):
    """Response for maintenance rotation run creation."""

    item: MaintenanceRotationRunItem

    model_config = ConfigDict(from_attributes=True)


class RetentionPolicy(BaseModel):
    """Retention policy descriptor."""
    key: str
    days: int | None = None
    description: str | None = None

    model_config = ConfigDict(from_attributes=True)


class RetentionPoliciesResponse(BaseModel):
    """Response for retention policy listing."""
    policies: list[RetentionPolicy]

    model_config = ConfigDict(from_attributes=True)


class RetentionPolicyUpdateRequest(BaseModel):
    """Request to update a retention policy."""
    days: int = Field(..., ge=1, le=3650)
    preview_signature: str | None = Field(default=None, min_length=1, max_length=2048)

    model_config = ConfigDict(from_attributes=True)


class RetentionPolicyPreviewRequest(BaseModel):
    """Request to preview the impact of a retention policy change."""

    current_days: int = Field(..., ge=1, le=3650)
    days: int = Field(..., ge=1, le=3650)

    model_config = ConfigDict(from_attributes=True)


class RetentionPolicyPreviewCounts(BaseModel):
    """Authoritative count summary for a retention policy preview."""

    audit_log_entries: NonNegativeInt = 0
    job_records: NonNegativeInt = 0
    backup_files: NonNegativeInt = 0

    model_config = ConfigDict(from_attributes=True)


class RetentionPolicyPreviewResponse(BaseModel):
    """Response for an authoritative retention policy preview."""

    key: str
    current_days: int = Field(..., ge=1, le=3650)
    new_days: int = Field(..., ge=1, le=3650)
    counts: RetentionPolicyPreviewCounts
    preview_signature: str = Field(..., min_length=1, max_length=2048)
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestSummaryItem(BaseModel):
    """Authoritative per-category summary entry for a DSR preview/request."""

    key: str
    label: str
    count: NonNegativeInt

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestPreviewRequest(BaseModel):
    """Request payload for authoritative DSR preview."""

    requester_identifier: str = Field(..., min_length=1, max_length=255)
    request_type: Literal["access", "export", "erasure"] | None = None
    categories: list[str] | None = None

    @field_validator("requester_identifier")
    @classmethod
    def normalize_requester_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("requester_identifier is required")
        return normalized

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("categories must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError("categories must contain strings")
            item = entry.strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestPreviewResponse(BaseModel):
    """Response payload for authoritative DSR preview."""

    requester_identifier: str
    resolved_user_id: int
    request_type: Literal["access", "export", "erasure"] | None = None
    selected_categories: list[str]
    summary: list[DataSubjectRequestSummaryItem]
    counts: dict[str, NonNegativeInt]
    coverage_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestCreateRequest(BaseModel):
    """Request payload to record a DSR for review."""

    client_request_id: str = Field(..., min_length=3, max_length=128)
    requester_identifier: str = Field(..., min_length=1, max_length=255)
    request_type: Literal["access", "export", "erasure"]
    categories: list[str] | None = None
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("client_request_id", "requester_identifier")
    @classmethod
    def normalize_nonempty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value is required")
        return normalized

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("categories must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError("categories must contain strings")
            item = entry.strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestItem(BaseModel):
    """Persisted DSR record returned by admin endpoints."""

    id: int
    client_request_id: str
    requester_identifier: str
    resolved_user_id: int | None = None
    request_type: Literal["access", "export", "erasure"]
    status: str
    selected_categories: list[str]
    preview_summary: list[DataSubjectRequestSummaryItem]
    coverage_metadata: dict[str, Any] = Field(default_factory=dict)
    requested_by_user_id: int | None = None
    requested_at: datetime
    notes: str | None = None

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestCreateResponse(BaseModel):
    """Response payload for DSR record creation."""

    item: DataSubjectRequestItem

    model_config = ConfigDict(from_attributes=True)


class DataSubjectRequestListResponse(BaseModel):
    """Response payload for DSR request log listing."""

    items: list[DataSubjectRequestItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Admin Monitoring Schemas

class AdminAlertRuleResponse(BaseModel):
    """Admin alert-rule response payload."""

    id: int
    metric: str
    operator: str
    threshold: float
    duration_minutes: int
    severity: str
    enabled: bool = True
    created_by_user_id: int | None = None
    updated_by_user_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminAlertRuleListResponse(BaseModel):
    """List response for admin alert rules."""

    items: list[AdminAlertRuleResponse]

    model_config = ConfigDict(from_attributes=True)


class AdminAlertRuleCreateRequest(BaseModel):
    """Create request for an admin alert rule."""

    metric: str = Field(..., min_length=1, max_length=100)
    operator: str = Field(..., min_length=1, max_length=32)
    threshold: float
    duration_minutes: int = Field(..., ge=1, le=1440)
    severity: str = Field(..., min_length=1, max_length=32)
    enabled: bool = True

    model_config = ConfigDict(from_attributes=True)


class AdminAlertRuleCreateResponse(BaseModel):
    """Create response for an admin alert rule."""

    item: AdminAlertRuleResponse

    model_config = ConfigDict(from_attributes=True)


class AdminAlertRuleDeleteResponse(BaseModel):
    """Delete response for an admin alert rule."""

    status: str
    id: int

    model_config = ConfigDict(from_attributes=True)


class AdminAlertStateResponse(BaseModel):
    """Authoritative admin overlay state for a runtime alert."""

    alert_identity: str
    assigned_to_user_id: int | None = None
    snoozed_until: datetime | None = None
    escalated_severity: str | None = None
    acknowledged_at: datetime | None = None
    dismissed_at: datetime | None = None
    updated_by_user_id: int | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminAlertStateMutationResponse(BaseModel):
    """Mutation response for admin alert overlay changes."""

    item: AdminAlertStateResponse

    model_config = ConfigDict(from_attributes=True)


class AdminAlertAssignRequest(BaseModel):
    """Assign request for a monitoring alert."""

    assigned_to_user_id: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(from_attributes=True)


class AdminAlertSnoozeRequest(BaseModel):
    """Snooze request for a monitoring alert."""

    snoozed_until: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminAlertEscalateRequest(BaseModel):
    """Escalate request for a monitoring alert."""

    severity: Literal["warning", "critical"] = "critical"

    model_config = ConfigDict(from_attributes=True)


class AdminAlertEventResponse(BaseModel):
    """History item for admin monitoring alert actions."""

    id: int
    alert_identity: str
    action: str
    actor_user_id: int | None = None
    details: dict[str, Any] | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminAlertHistoryListResponse(BaseModel):
    """List response for admin monitoring alert history."""

    items: list[AdminAlertEventResponse]

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# System Ops Schemas (Logs, Incidents, Maintenance, Feature Flags)

class SystemLogEntry(BaseModel):
    """Single system log entry."""
    timestamp: datetime | None = None
    level: str | None = None
    message: str | None = None
    logger: str | None = None
    module: str | None = None
    function: str | None = None
    line: int | None = None
    request_id: str | None = None
    org_id: int | None = None
    user_id: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    correlation_id: str | None = None
    event: str | None = None

    model_config = ConfigDict(from_attributes=True)


class SystemLogsResponse(BaseModel):
    """Response for system log listing."""
    items: list[SystemLogEntry]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class MaintenanceState(BaseModel):
    """Maintenance mode state."""
    enabled: bool
    message: str = ""
    allowlist_user_ids: list[int] = []
    allowlist_emails: list[str] = []
    updated_at: datetime | None = None
    updated_by: str | None = None

    model_config = ConfigDict(from_attributes=True)


class MaintenanceUpdateRequest(BaseModel):
    """Request to update maintenance mode."""
    enabled: bool
    message: str | None = None
    allowlist_user_ids: list[int] | None = None
    allowlist_emails: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class FeatureFlagSnapshot(BaseModel):
    """Feature flag state snapshot for change diffs."""
    scope: Literal["global", "org", "user"]
    enabled: bool
    org_id: int | None = None
    user_id: int | None = None
    target_user_ids: list[int] = Field(default_factory=list)
    rollout_percent: int = Field(100, ge=0, le=100)
    variant_value: str | None = None

    model_config = ConfigDict(from_attributes=True)


class FeatureFlagHistoryEntry(BaseModel):
    """Feature flag change history entry."""
    timestamp: datetime
    enabled: bool
    actor: str | None = None
    note: str | None = None
    before: FeatureFlagSnapshot | None = None
    after: FeatureFlagSnapshot | None = None

    model_config = ConfigDict(from_attributes=True)


class FeatureFlagItem(BaseModel):
    """Feature flag descriptor."""
    key: str
    scope: Literal["global", "org", "user"]
    enabled: bool
    description: str | None = None
    org_id: int | None = None
    user_id: int | None = None
    target_user_ids: list[int] = Field(default_factory=list)
    rollout_percent: int = Field(100, ge=0, le=100)
    variant_value: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    history: list[FeatureFlagHistoryEntry] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class FeatureFlagsResponse(BaseModel):
    """Response for feature flag listing."""
    items: list[FeatureFlagItem]
    total: int

    model_config = ConfigDict(from_attributes=True)


class FeatureFlagUpsertRequest(BaseModel):
    """Request to upsert a feature flag."""
    scope: Literal["global", "org", "user"]
    enabled: bool
    description: str | None = None
    org_id: int | None = None
    user_id: int | None = None
    target_user_ids: list[int] | None = None
    rollout_percent: int | None = Field(None, ge=0, le=100)
    variant_value: str | None = Field(None, max_length=255)
    note: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("target_user_ids")
    @classmethod
    def validate_target_user_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        cleaned: list[int] = []
        for candidate in value:
            try:
                parsed = int(candidate)
            except (TypeError, ValueError) as exc:
                raise ValueError("target_user_ids must contain integers") from exc
            if parsed <= 0:
                raise ValueError("target_user_ids must contain positive integers")
            cleaned.append(parsed)
        return sorted(set(cleaned))

    @field_validator("variant_value")
    @classmethod
    def normalize_variant_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class IncidentEvent(BaseModel):
    """Incident timeline entry."""
    id: str
    message: str
    created_at: datetime
    actor: str | None = None

    model_config = ConfigDict(from_attributes=True)


class IncidentActionItem(BaseModel):
    """Structured incident action item."""
    id: str
    text: str
    done: bool = False

    model_config = ConfigDict(from_attributes=True)


class IncidentItem(BaseModel):
    """Incident summary with timeline."""
    id: str
    title: str
    status: Literal["open", "investigating", "mitigating", "resolved"]
    severity: Literal["low", "medium", "high", "critical"]
    summary: str | None = None
    tags: list[str] = []
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None
    acknowledged_at: datetime | None = None
    created_by: str | None = None
    updated_by: str | None = None
    timeline: list[IncidentEvent] = []
    assigned_to_user_id: int | None = None
    assigned_to_label: str | None = None
    root_cause: str | None = None
    impact: str | None = None
    runbook_url: str | None = None
    action_items: list[IncidentActionItem] = []
    time_to_acknowledge_seconds: int | None = None
    time_to_resolve_seconds: int | None = None

    model_config = ConfigDict(from_attributes=True)


class IncidentListResponse(BaseModel):
    """Response for incident listing."""
    items: list[IncidentItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class IncidentCreateRequest(BaseModel):
    """Request to create an incident."""
    title: str
    status: Literal["open", "investigating", "mitigating", "resolved"] | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    summary: str | None = None
    tags: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class IncidentUpdateRequest(BaseModel):
    """Request to update an incident.

    Partial-update contract for workflow fields:
    - omitted means unchanged
    - explicit ``null`` means clear

    Callers must preserve that distinction via ``model_fields_set`` or
    ``model_dump(exclude_unset=True)`` when invoking the service layer.
    """
    title: str | None = None
    status: Literal["open", "investigating", "mitigating", "resolved"] | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    summary: str | None = None
    tags: list[str] | None = None
    assigned_to_user_id: int | None = None
    root_cause: str | None = None
    impact: str | None = None
    acknowledged_at: datetime | None = None
    runbook_url: str | None = None
    action_items: list[IncidentActionItem] | None = None
    update_message: str | None = None

    model_config = ConfigDict(from_attributes=True)


class IncidentEventCreateRequest(BaseModel):
    """Request to append a timeline entry."""
    message: str

    model_config = ConfigDict(from_attributes=True)


class IncidentNotifyRequest(BaseModel):
    """Request to notify stakeholders about an incident."""
    recipients: list[str]
    message: str | None = None

    model_config = ConfigDict(from_attributes=True)


class IncidentNotifyRecipientResult(BaseModel):
    """Per-recipient delivery result."""
    email: str
    status: str
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


class IncidentNotifyResponse(BaseModel):
    """Response from incident stakeholder notification.

    Used by both the email notification endpoint (populates ``notifications``)
    and the webhook dispatch endpoint (populates ``webhooks_delivered``).
    """
    incident_id: str
    notifications: list[IncidentNotifyRecipientResult] = []
    notified: bool = True
    webhooks_delivered: int = 0

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Email Delivery Schemas


class EmailDeliveryItem(BaseModel):
    """A single email delivery log entry."""
    id: str
    recipient: str = ""
    subject: str = ""
    template: str | None = None
    status: str = ""
    error: str | None = None
    sent_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class EmailDeliveryListResponse(BaseModel):
    """Response for email delivery log listing."""
    items: list[EmailDeliveryItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Webhook Schemas


class WebhookItem(BaseModel):
    """Webhook summary (secret redacted)."""
    id: str
    url: str
    events: list[str] = []
    enabled: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class WebhookCreateResponse(BaseModel):
    """Response for webhook creation, includes the secret (shown once)."""
    id: str
    url: str
    secret: str
    events: list[str] = []
    enabled: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class WebhookListResponse(BaseModel):
    """Response for webhook listing."""
    items: list[WebhookItem]
    total: int
    limit: int | None = Field(default=None, ge=1, description="Alias for pagination.limit")
    offset: int | None = Field(default=None, ge=0, description="Alias for pagination.offset")
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class WebhookCreateRequest(BaseModel):
    """Request to create a webhook."""
    url: str
    events: list[str]
    enabled: bool = True

    model_config = ConfigDict(from_attributes=True)


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook (partial update)."""
    url: str | None = None
    events: list[str] | None = None
    enabled: bool | None = None

    model_config = ConfigDict(from_attributes=True)


class WebhookDeliveryItem(BaseModel):
    """A single webhook delivery record."""
    id: str
    webhook_id: str
    event_type: str = ""
    status_code: int | None = None
    response_time_ms: int | None = None
    success: bool = False
    error: str | None = None
    attempted_at: str | None = None
    payload_preview: str | None = None

    model_config = ConfigDict(from_attributes=True)


class WebhookDeliveryListResponse(BaseModel):
    """Response for webhook delivery listing."""
    items: list[WebhookDeliveryItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Batch Operation Schemas

class BatchUserOperation(BaseModel):
    """Batch operation on multiple users"""
    user_ids: list[int]
    operation: str = Field(..., pattern="^(activate|deactivate|verify|lock|unlock|delete)$")

    model_config = ConfigDict(from_attributes=True)


class BatchOperationResponse(BaseModel):
    """Response for batch operations"""
    success_count: int
    failed_count: int
    failed_ids: list[int] = []
    message: str

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Usage Reporting Schemas

class UsageDailyRow(BaseModel):
    """Single usage_daily record."""
    user_id: int
    day: date | str
    requests: int
    errors: int
    bytes_total: int
    bytes_in_total: int | None = None
    latency_avg_ms: float | None = None

    model_config = ConfigDict(from_attributes=True)


class UsageDailyResponse(BaseModel):
    """Response for daily usage query."""
    items: list[UsageDailyRow]
    total: int
    page: int
    limit: int
    pagination: PagePaginationMeta

    model_config = ConfigDict(from_attributes=True)


class UsageTopRow(BaseModel):
    """Aggregated usage by user for a date range."""
    user_id: int
    requests: int
    errors: int
    bytes_total: int
    bytes_in_total: int | None = None
    latency_avg_ms: float | None = None

    model_config = ConfigDict(from_attributes=True)


class UsageTopResponse(BaseModel):
    items: list[UsageTopRow]

    model_config = ConfigDict(from_attributes=True)


class UsageAggregateResponse(BaseModel):
    """Response for manual usage aggregation."""
    status: str
    day: str | None = None
    reason: str | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Budget Governance Schemas

_BUDGET_FIELD_KEYS = {
    "budget_day_usd",
    "budget_month_usd",
    "budget_day_tokens",
    "budget_month_tokens",
}


def _normalize_threshold_list(values: list[Any]) -> list[int]:
    if not values:
        raise ValueError("Alert thresholds must not be empty")
    cleaned: list[int] = []
    for val in values:
        try:
            num = int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError("Alert thresholds must be integers") from exc
        if num < 1 or num > 100:
            raise ValueError("Alert thresholds must be between 1 and 100")
        cleaned.append(num)
    return sorted(set(cleaned))


def _validate_usd_precision(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("USD budgets must be valid decimals") from exc
    if dec.as_tuple().exponent < -2:
        raise ValueError("USD budgets must have at most 2 decimal places")
    return float(dec)


class BudgetAlertThresholds(BaseModel):
    """Alert thresholds for budgets (global + per-metric)."""
    global_: list[int] | None = Field(default=None, alias="global")
    per_metric: dict[str, list[int] | None] | None = None

    @field_validator("global_")
    @classmethod
    def validate_global_thresholds(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        return _normalize_threshold_list(v)

    @field_validator("per_metric")
    @classmethod
    def validate_per_metric_thresholds(
        cls, v: dict[str, list[int] | None] | None
    ) -> dict[str, list[int] | None] | None:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("Per-metric thresholds must be a mapping")
        cleaned: dict[str, list[int] | None] = {}
        for key, values in v.items():
            if key not in _BUDGET_FIELD_KEYS:
                raise ValueError("Unknown per-metric budget key")
            if values is None:
                cleaned[key] = None
                continue
            cleaned[key] = _normalize_threshold_list(values)
        return cleaned

    model_config = ConfigDict(populate_by_name=True)


class BudgetEnforcementMode(BaseModel):
    """Enforcement mode for budgets (global + per-metric)."""
    global_: Literal["none", "soft", "hard"] | None = Field(default=None, alias="global")
    per_metric: dict[str, Literal["none", "soft", "hard"] | None] | None = None

    @field_validator("per_metric")
    @classmethod
    def validate_per_metric_modes(
        cls, v: dict[str, Literal["none", "soft", "hard"] | None] | None
    ) -> dict[str, Literal["none", "soft", "hard"] | None] | None:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("Per-metric enforcement must be a mapping")
        cleaned: dict[str, Literal["none", "soft", "hard"] | None] = {}
        for key, value in v.items():
            if key not in _BUDGET_FIELD_KEYS:
                raise ValueError("Unknown per-metric budget key")
            if value is None:
                cleaned[key] = None
                continue
            if value not in ("none", "soft", "hard"):
                raise ValueError("Enforcement mode must be none, soft, or hard")
            cleaned[key] = value
        return cleaned

    model_config = ConfigDict(populate_by_name=True)


class BudgetSettings(BaseModel):
    """Budget configuration for an organization."""
    budget_day_usd: float | None = Field(None, ge=0)
    budget_month_usd: float | None = Field(None, ge=0)
    budget_day_tokens: int | None = Field(None, ge=0)
    budget_month_tokens: int | None = Field(None, ge=0)
    alert_thresholds: BudgetAlertThresholds | None = None
    enforcement_mode: BudgetEnforcementMode | None = None

    @field_validator("budget_day_usd", "budget_month_usd", mode="before")
    @classmethod
    def validate_usd_precision(cls, v: Any | None) -> float | None:
        return _validate_usd_precision(v)

    @field_validator("alert_thresholds", mode="before")
    @classmethod
    def coerce_alert_thresholds(cls, v: Any | None) -> Any | None:
        if v is None:
            return v
        if isinstance(v, list):
            return {"global": v}
        return v

    @field_validator("enforcement_mode", mode="before")
    @classmethod
    def coerce_enforcement_mode(cls, v: Any | None) -> Any | None:
        if v is None:
            return v
        if isinstance(v, str):
            return {"global": v}
        return v


class OrgBudgetUpdateRequest(BaseModel):
    """Upsert budget settings for an organization."""
    org_id: int = Field(..., ge=1)
    budgets: BudgetSettings | None = None
    clear_budgets: bool = False


class OrgBudgetSelfUpdateRequest(BaseModel):
    """Upsert budget settings for the current organization context."""
    budgets: BudgetSettings | None = None
    clear_budgets: bool = False


class OrgBudgetItem(BaseModel):
    """Budget details for an organization."""
    org_id: int
    org_name: str
    org_slug: str | None = None
    plan_name: str
    plan_display_name: str
    budgets: BudgetSettings = Field(default_factory=BudgetSettings)
    custom_limits: dict[str, Any] = Field(default_factory=dict)
    effective_limits: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class OrgBudgetListResponse(BaseModel):
    items: list[OrgBudgetItem]
    total: int
    page: int
    limit: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class BudgetForecastResponse(BaseModel):
    """Forecast summary for organization budgets."""

    org_id: int
    forecast_available: bool
    reason: str | None = None
    monthly_limit_usd: float | None = None
    note: str | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# LLM Usage Schemas

class LLMUsageLogRow(BaseModel):
    id: int
    ts: datetime
    user_id: int | None = None
    key_id: int | None = None
    endpoint: str | None = None
    operation: str | None = None
    provider: str | None = None
    model: str | None = None
    status: int | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    billable_input_tokens: int | None = None
    estimate_source: str | None = None
    currency: str | None = None
    estimated: bool | None = None
    request_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class LLMUsageLogResponse(BaseModel):
    items: list[LLMUsageLogRow]
    total: int
    page: int
    limit: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class LLMUsageSummaryRow(BaseModel):
    group_value: str
    group_value_secondary: str | None = None
    requests: int
    errors: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost_usd: float
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    billable_input_tokens: int = 0
    provider_usage_count: int = 0
    stream_estimate_count: int = 0
    disconnect_estimate_count: int = 0
    missing_usage_count: int = 0
    local_diagnostic_count: int = 0
    estimated_usage_count: int = 0
    latency_avg_ms: float | None = None

    model_config = ConfigDict(from_attributes=True)


class LLMUsageSummaryResponse(BaseModel):
    items: list[LLMUsageSummaryRow]

    model_config = ConfigDict(from_attributes=True)


class LLMTopSpenderRow(BaseModel):
    user_id: int
    total_cost_usd: float
    requests: int

    model_config = ConfigDict(from_attributes=True)


class LLMTopSpendersResponse(BaseModel):
    items: list[LLMTopSpenderRow]

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Router Analytics Schemas

RouterAnalyticsRange = Literal["realtime", "1h", "8h", "24h", "7d", "30d"]
RouterAnalyticsGranularity = Literal["1m", "5m", "15m", "1h"]


class RouterAnalyticsRangeQuery(BaseModel):
    """Shared query parameters for router analytics endpoints."""

    range: RouterAnalyticsRange = "8h"
    org_id: int | None = Field(None, ge=1)
    provider: str | None = None
    model: str | None = None
    token_id: int | None = Field(None, ge=1)
    granularity: RouterAnalyticsGranularity | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsDataWindow(BaseModel):
    """Effective data window used for an analytics response."""

    start: datetime
    end: datetime
    range: RouterAnalyticsRange

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsSeriesPoint(BaseModel):
    """Single timeseries point for status chart rendering."""

    ts: datetime
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    avg_latency_ms: float | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsBreakdownRow(BaseModel):
    """Shared row shape for breakdown tables."""

    key: str
    label: str | None = None
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0
    avg_latency_ms: float | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsStatusKpis(BaseModel):
    """KPI cards displayed in the router analytics status tab."""

    requests: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    avg_latency_ms: float | None = None
    avg_gen_toks_per_s: float | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsStatusResponse(BaseModel):
    """Payload for /admin/router-analytics/status."""

    kpis: RouterAnalyticsStatusKpis
    series: list[RouterAnalyticsSeriesPoint] = Field(default_factory=list)
    providers_available: int = 0
    providers_online: int = 0
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsBreakdownsResponse(BaseModel):
    """Payload for /admin/router-analytics/status/breakdowns."""

    providers: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    models: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    token_names: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    remote_ips: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    user_agents: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsMetaOption(BaseModel):
    """Reusable metadata option entry."""

    value: str
    label: str
    key_id: int | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsMetaResponse(BaseModel):
    """Payload for /admin/router-analytics/meta."""

    providers: list[RouterAnalyticsMetaOption] = Field(default_factory=list)
    models: list[RouterAnalyticsMetaOption] = Field(default_factory=list)
    tokens: list[RouterAnalyticsMetaOption] = Field(default_factory=list)
    ranges: list[RouterAnalyticsRange] = Field(default_factory=lambda: ["realtime", "1h", "8h", "24h", "7d", "30d"])
    granularities: list[RouterAnalyticsGranularity] = Field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    generated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsQuotaMetric(BaseModel):
    """Budget metric usage versus configured limit."""

    used: float = 0.0
    limit: float = 0.0
    utilization_pct: float | None = None
    exceeded: bool = False

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsQuotaRow(BaseModel):
    """Quota row for a key/token entity."""

    key_id: int
    token_name: str
    requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    day_tokens: RouterAnalyticsQuotaMetric | None = None
    month_tokens: RouterAnalyticsQuotaMetric | None = None
    day_usd: RouterAnalyticsQuotaMetric | None = None
    month_usd: RouterAnalyticsQuotaMetric | None = None
    over_budget: bool = False
    reasons: list[str] = Field(default_factory=list)
    last_seen_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsQuotaSummary(BaseModel):
    """Quota overview summary counters."""

    keys_total: int = 0
    keys_over_budget: int = 0
    budgeted_keys: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsQuotaResponse(BaseModel):
    """Payload for /admin/router-analytics/quota."""

    summary: RouterAnalyticsQuotaSummary
    items: list[RouterAnalyticsQuotaRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsProviderRow(BaseModel):
    """Provider-level aggregate row for router analytics providers tab."""

    provider: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float | None = None
    errors: int = 0
    success_rate_pct: float | None = None
    online: bool = False

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsProvidersSummary(BaseModel):
    """Providers overview summary counters."""

    providers_total: int = 0
    providers_online: int = 0
    failover_events: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsProvidersResponse(BaseModel):
    """Payload for /admin/router-analytics/providers."""

    summary: RouterAnalyticsProvidersSummary
    items: list[RouterAnalyticsProviderRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsAccessSummary(BaseModel):
    """Access tab overview summary counters."""

    token_names_total: int = 0
    remote_ips_total: int = 0
    user_agents_total: int = 0
    anonymous_requests: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsAccessResponse(BaseModel):
    """Payload for /admin/router-analytics/access."""

    summary: RouterAnalyticsAccessSummary
    token_names: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    remote_ips: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    user_agents: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsNetworkSummary(BaseModel):
    """Network tab overview summary counters."""

    remote_ips_total: int = 0
    endpoints_total: int = 0
    operations_total: int = 0
    error_requests: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsNetworkResponse(BaseModel):
    """Payload for /admin/router-analytics/network."""

    summary: RouterAnalyticsNetworkSummary
    remote_ips: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    endpoints: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    operations: list[RouterAnalyticsBreakdownRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsModelRow(BaseModel):
    """Model-level aggregate row for router analytics models tab."""

    model: str
    provider: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float | None = None
    errors: int = 0
    success_rate_pct: float | None = None
    online: bool = False

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsModelsSummary(BaseModel):
    """Models overview summary counters."""

    models_total: int = 0
    models_online: int = 0
    providers_total: int = 0
    error_requests: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsModelsResponse(BaseModel):
    """Payload for /admin/router-analytics/models."""

    summary: RouterAnalyticsModelsSummary
    items: list[RouterAnalyticsModelRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsConversationRow(BaseModel):
    """Conversation-level aggregate row for router analytics conversations tab."""

    conversation_id: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float | None = None
    errors: int = 0
    success_rate_pct: float | None = None
    last_seen_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsConversationsSummary(BaseModel):
    """Conversations overview summary counters."""

    conversations_total: int = 0
    active_conversations: int = 0
    avg_requests_per_conversation: float | None = None
    error_requests: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsConversationsResponse(BaseModel):
    """Payload for /admin/router-analytics/conversations."""

    summary: RouterAnalyticsConversationsSummary
    items: list[RouterAnalyticsConversationRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsLogRow(BaseModel):
    """Row entry for router analytics log tab."""

    ts: datetime
    request_id: str | None = None
    conversation_id: str = "unknown"
    provider: str = "unknown"
    model: str = "unknown"
    token_name: str = "unknown"
    endpoint: str = "unknown"
    operation: str = "unknown"
    status: int | None = None
    latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    remote_ip: str = "unknown"
    user_agent: str = "unknown"
    estimated: bool = False
    error: bool = False

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsLogSummary(BaseModel):
    """Log tab summary counters."""

    requests_total: int = 0
    error_requests: int = 0
    estimated_requests: int = 0
    request_ids_total: int = 0

    model_config = ConfigDict(from_attributes=True)


class RouterAnalyticsLogResponse(BaseModel):
    """Payload for /admin/router-analytics/log."""

    summary: RouterAnalyticsLogSummary
    items: list[RouterAnalyticsLogRow] = Field(default_factory=list)
    generated_at: datetime
    data_window: RouterAnalyticsDataWindow
    stale_seconds: int | None = None
    partial: bool = False
    warnings: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Tool Permission Schemas (MCP Integration)

class ToolPermissionCreateRequest(BaseModel):
    """Create a tool execute permission.

    If tool_name is "*", creates tools.execute:* (wildcard).
    """
    tool_name: str = Field(..., min_length=1)
    description: str | None = None


class ToolPermissionResponse(BaseModel):
    name: str
    description: str | None = None
    category: str | None = None


class ToolPermissionGrantRequest(BaseModel):
    """Grant a tool execution permission to a role.

    tool_name '*' means tools.execute:*
    """
    tool_name: str = Field(..., min_length=1)


class ToolPermissionBatchRequest(BaseModel):
    """Grant multiple tool execution permissions to a role in one call."""
    tool_names: list[str] = Field(..., min_length=1)


class ToolPermissionPrefixRequest(BaseModel):
    """Grant/Revoke all tool permissions matching a name prefix.

    Examples:
      {"prefix": "tools.execute:media."} or {"prefix": "media."}
    """
    prefix: str = Field(..., min_length=1)


#######################################################################################################################
#
# MCP Tool Catalog Schemas

class ToolCatalogCreateRequest(BaseModel):
    """Create a new MCP tool catalog."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    org_id: int | None = None
    team_id: int | None = None
    is_active: bool | None = True


class ToolCatalogResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    org_id: int | None = None
    team_id: int | None = None
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ToolCatalogEntryCreateRequest(BaseModel):
    """Add a tool entry to a catalog."""
    tool_name: str = Field(..., min_length=1, max_length=200)
    module_id: str | None = Field(None, max_length=200)


class ToolCatalogEntryResponse(BaseModel):
    catalog_id: int
    tool_name: str
    module_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


#
# Kanban FTS Maintenance Schema

class KanbanFtsMaintenanceResponse(BaseModel):
    """Response for Kanban FTS maintenance actions."""
    user_id: int
    action: Literal["optimize", "rebuild"]
    status: str = "ok"

#
#
# Notes Title Settings Schema

class NotesTitleSettingsUpdate(BaseModel):
    """Update payload for Notes auto-title settings.

    - llm_enabled: enable/disable LLM-backed title generation
    - default_strategy: default strategy to use when clients send "heuristic"
    """
    model_config = ConfigDict(extra='forbid')

    llm_enabled: bool | None = Field(default=None)
    default_strategy: Literal['heuristic', 'llm', 'llm_fallback'] | None = Field(default=None)


class NotesTitleSettingsResponse(BaseModel):
    """Response payload for Notes auto-title settings."""
    llm_enabled: bool
    default_strategy: str
    effective_strategy: str
    strategies: list[str]

    model_config = ConfigDict(from_attributes=True)


#
# Cleanup worker settings (admin)

class AdminCleanupSettingsUpdate(BaseModel):
    """Update payload for ephemeral cleanup worker settings.

    - enabled: turn cleanup worker on/off
    - interval_sec: run interval in seconds (60..604800)
    """
    model_config = ConfigDict(extra='forbid')

    enabled: bool | None = Field(default=None)
    interval_sec: int | None = Field(default=None, ge=60, le=604800)


class AdminCleanupSettingsResponse(BaseModel):
    """Response payload for cleanup worker settings."""
    enabled: bool
    interval_sec: int

    model_config = ConfigDict(from_attributes=True)


#
# Unified Circuit Breaker (admin)

class AdminCircuitBreakerStatus(BaseModel):
    """Circuit breaker status row exposed by `/api/v1/admin/circuit-breakers`.

    `source` semantics:
    - `memory`: present only in the current process registry
    - `persistent`: present only in shared persisted registry storage
    - `mixed`: present in both memory and persisted storage (expected when
      persistence is enabled for in-process breakers)
    """

    name: str
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"]
    category: str | None = None
    service: str | None = None
    operation: str | None = None
    failure_count: int
    success_count: int
    last_failure_time: float | None = None
    last_state_change_time: float | None = None
    half_open_calls: int
    current_recovery_timeout: float
    # "memory": in-process only, "persistent": store-only row, "mixed": both
    # (common in persistent mode for active in-process breakers)
    source: Literal["memory", "persistent", "mixed"] = "memory"
    settings: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class AdminCircuitBreakerListResponse(BaseModel):
    """List response for unified circuit breaker admin endpoint."""

    items: list[AdminCircuitBreakerStatus]
    total: int

    model_config = ConfigDict(from_attributes=True)


class AdminCircuitBreakerListFilters(BaseModel):
    """Filter contract for unified circuit breaker admin listing."""

    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] | None = None
    category: str | None = None
    service: str | None = None
    name_prefix: str | None = None

    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Per-API-Key Usage Attribution

class ApiKeyDailySnapshot(BaseModel):
    """A single day's usage snapshot for an API key."""
    date: str
    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


class ApiKeyUsageSummary(BaseModel):
    """Aggregated usage summary for a single API key."""
    key_id: str
    request_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    last_used_at: str | None = None
    daily_snapshots: list[ApiKeyDailySnapshot] = Field(default_factory=list)


class ApiKeyUsageTopItem(BaseModel):
    """A ranked entry in the top-keys-by-usage list."""
    key_id: str
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    last_used_at: str | None = None


class ApiKeyUsageTopResponse(BaseModel):
    """Response for the top-keys-by-usage endpoint."""
    items: list[ApiKeyUsageTopItem] = Field(default_factory=list)


#######################################################################################################################
#
# Admin Webhooks Schemas


class AdminWebhookCreateRequest(BaseModel):
    """Request to register a new admin webhook."""

    url: str = Field(..., min_length=1, max_length=2048)
    event_types: list[str] = Field(default_factory=lambda: ["*"])
    description: str = Field(default="", max_length=500)
    secret: str | None = Field(default=None, max_length=256)
    active: bool = True
    retry_count: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=10, ge=1, le=60)

    @field_validator("secret", mode="before")
    @classmethod
    def normalize_secret(cls, value: Any) -> Any:
        return _blank_string_to_none(value)

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        """Only allow webhook URLs that pass egress safety checks."""
        return validate_admin_webhook_url(v)

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookUpdateRequest(BaseModel):
    """Request to update an existing admin webhook."""

    url: str | None = Field(default=None, min_length=1, max_length=2048)
    event_types: list[str] | None = None
    description: str | None = Field(default=None, max_length=500)
    secret: str | None = Field(default=None, max_length=256)
    active: bool | None = None
    retry_count: int | None = Field(default=None, ge=0, le=10)
    timeout_seconds: int | None = Field(default=None, ge=1, le=60)

    @field_validator("secret", mode="before")
    @classmethod
    def normalize_secret(cls, value: Any) -> Any:
        return _blank_string_to_none(value)

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str | None) -> str | None:
        """Only allow webhook URLs that pass egress safety checks."""
        if v is None:
            return v
        return validate_admin_webhook_url(v)

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookResponse(BaseModel):
    """Single webhook record returned by the API."""

    id: int
    url: str
    event_types: list[str]
    description: str
    active: bool
    retry_count: int
    timeout_seconds: int
    created_by: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookListResponse(BaseModel):
    """Paginated list of admin webhooks."""

    items: list[AdminWebhookResponse]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookDeleteResponse(BaseModel):
    """Response after deleting a webhook."""

    deleted: bool
    id: int

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookDeliveryLogEntry(BaseModel):
    """Single delivery log entry."""

    id: int
    webhook_id: int
    event_type: str
    status_code: int | None = None
    latency_ms: int | None = None
    retry_attempt: int = 0
    error_message: str | None = None
    delivered_at: datetime | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookDeliveryLogListResponse(BaseModel):
    """Paginated delivery log list."""

    items: list[AdminWebhookDeliveryLogEntry]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)

    model_config = ConfigDict(from_attributes=True)


class AdminWebhookTestResponse(BaseModel):
    """Response from testing a webhook."""

    success: bool
    status_code: int | None = None
    latency_ms: int | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


#
## End of admin_schemas.py
#######################################################################################################################
