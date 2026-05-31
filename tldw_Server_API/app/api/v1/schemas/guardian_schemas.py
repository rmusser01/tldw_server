"""
Pydantic schemas for Guardian Controls & Self-Monitoring API.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
    validate_offset_pagination_aliases,
)

# ── Guardian Relationship Schemas ────────────────────────────

class GuardianRelationshipCreate(BaseModel):
    dependent_user_id: str = Field(..., description="User ID of the dependent (child) account")
    relationship_type: Literal["parent", "legal_guardian", "institutional"] = "parent"
    dependent_visible: bool = Field(True, description="Whether the dependent can see that monitoring is active")


class GuardianRelationshipResponse(BaseModel):
    id: str
    guardian_user_id: str
    dependent_user_id: str
    relationship_type: str
    status: str
    consent_given_by_dependent: bool
    consent_given_at: str | None = None
    dependent_visible: bool
    dissolution_reason: str | None = None
    dissolved_at: str | None = None
    created_at: str
    updated_at: str
    model_config = ConfigDict(from_attributes=True)


class GuardianRelationshipList(BaseModel):
    items: list[GuardianRelationshipResponse]
    total: int


class DissolveRequest(BaseModel):
    reason: str = Field("manual", description="Reason for dissolving the relationship")


# ── Supervised Policy Schemas ────────────────────────────────

class SupervisedPolicyCreate(BaseModel):
    relationship_id: str = Field(..., description="Guardian relationship this policy belongs to")
    governance_policy_id: str | None = Field(None, description="Optional governance policy group")
    policy_type: Literal["block", "notify"] = "block"
    category: str = Field("", description="Topic category (e.g. 'explicit_content', 'self_harm')")
    pattern: str = Field("", description="Regex or literal pattern to match")
    pattern_type: Literal["literal", "regex"] = "literal"
    action: Literal["block", "redact", "warn", "notify"] = "block"
    phase: Literal["input", "output", "both"] = "both"
    severity: Literal["info", "warning", "critical"] = "warning"
    notify_guardian: bool = True
    notify_context: Literal["topic_only", "snippet", "full_message"] = "topic_only"
    message_to_dependent: str | None = Field(
        None,
        description="Custom message shown to dependent when content is blocked",
    )
    enabled: bool = True


class SupervisedPolicyUpdate(BaseModel):
    governance_policy_id: str | None = None
    policy_type: Literal["block", "notify"] | None = None
    category: str | None = None
    pattern: str | None = None
    pattern_type: Literal["literal", "regex"] | None = None
    action: Literal["block", "redact", "warn", "notify"] | None = None
    phase: Literal["input", "output", "both"] | None = None
    severity: Literal["info", "warning", "critical"] | None = None
    notify_guardian: bool | None = None
    notify_context: Literal["topic_only", "snippet", "full_message"] | None = None
    message_to_dependent: str | None = None
    enabled: bool | None = None


class SupervisedPolicyResponse(BaseModel):
    id: str
    relationship_id: str
    governance_policy_id: str | None = None
    policy_type: str
    category: str
    pattern: str
    pattern_type: str
    action: str
    phase: str
    severity: str
    notify_guardian: bool
    notify_context: str
    message_to_dependent: str | None = None
    enabled: bool
    created_at: str
    updated_at: str
    model_config = ConfigDict(from_attributes=True)


class SupervisedPolicyList(BaseModel):
    items: list[SupervisedPolicyResponse]
    total: int


# ── Supervision Audit Schemas ────────────────────────────────

class SupervisionAuditResponse(BaseModel):
    id: str
    relationship_id: str
    actor_user_id: str
    action: str
    target_user_id: str | None = None
    policy_id: str | None = None
    detail: str
    created_at: str
    model_config = ConfigDict(from_attributes=True)


class SupervisionAuditList(BaseModel):
    items: list[SupervisionAuditResponse]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "SupervisionAuditList":
        return default_offset_pagination_aliases(self)

    @model_validator(mode="after")
    def _validate_pagination_aliases(self) -> "SupervisionAuditList":
        return validate_offset_pagination_aliases(self)


# ── Governance Policy Schemas ────────────────────────────────

class GovernancePolicyCreate(BaseModel):
    name: str = Field(..., description="Name for the policy group")
    description: str = ""
    policy_mode: Literal["guardian", "self"] = "self"
    scope_chat_types: str = Field("all", description="Comma-separated chat types or 'all'")
    enabled: bool = True
    schedule_start: str | None = Field(None, description="HH:MM (24h) start time")
    schedule_end: str | None = Field(None, description="HH:MM (24h) end time")
    schedule_days: str | None = Field(None, description="Comma-separated: mon,tue,wed,...")
    schedule_timezone: str = "UTC"
    transparent: bool = Field(False, description="If true, managed user can see rule names")


class GovernancePolicyResponse(BaseModel):
    id: str
    owner_user_id: str
    name: str
    description: str
    policy_mode: str
    scope_chat_types: str
    enabled: bool
    schedule_start: str | None = None
    schedule_end: str | None = None
    schedule_days: str | None = None
    schedule_timezone: str
    transparent: bool
    created_at: str
    updated_at: str
    model_config = ConfigDict(from_attributes=True)


class GovernancePolicyList(BaseModel):
    items: list[GovernancePolicyResponse]
    total: int


# ── Self-Monitoring Schemas ──────────────────────────────────

class SelfMonitoringRuleCreate(BaseModel):
    name: str = Field(..., description="Human-readable name for this monitoring rule")
    category: str = Field(..., description="Category: 'fitness', 'mental_health', 'professional', 'custom'")
    patterns: list[str] = Field(..., description="List of patterns (literal or regex) to watch for")
    pattern_type: Literal["literal", "regex"] = "literal"
    except_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns that exclude false positives",
    )
    rule_type: Literal["block", "notify"] = "notify"
    action: Literal["block", "redact", "notify"] = "notify"
    phase: Literal["input", "output", "both"] = "both"
    severity: Literal["info", "warning", "critical"] = "info"
    display_mode: Literal["inline_banner", "sidebar_note", "post_session_summary", "silent_log"] = "inline_banner"
    block_message: str | None = Field(None, description="Custom message for self-block action")
    context_note: str | None = Field(None, description="Personal reminder note")
    notification_frequency: Literal[
        "every_message", "once_per_conversation", "once_per_day", "once_per_session"
    ] = "once_per_conversation"
    notification_channels: list[str] = Field(
        default_factory=lambda: ["in_app"],
        description="Channels: in_app, email, webhook, trusted_contact",
    )
    webhook_url: str | None = None
    trusted_contact_email: str | None = None
    crisis_resources_enabled: bool = Field(
        False,
        description="Show crisis helpline resources when triggered",
    )
    cooldown_minutes: int = Field(
        0,
        description="Minutes before rule can be disabled after creation (anti-impulsive-disable)",
    )
    bypass_protection: Literal["none", "cooldown", "confirmation", "partner_approval"] = "none"
    bypass_partner_user_id: str | None = None
    escalation_session_threshold: int = Field(0, description="Triggers in session before escalation (0=disabled)")
    escalation_session_action: str | None = None
    escalation_window_days: int = Field(0, description="Rolling window days for cross-session escalation")
    escalation_window_threshold: int = 0
    escalation_window_action: str | None = None
    min_context_length: int = Field(0, description="Minimum chars for match to fire")
    governance_policy_id: str | None = None
    enabled: bool = True

    @field_validator("patterns")
    @classmethod
    def _validate_patterns(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one pattern is required")
        return [p.strip() for p in v if p.strip()]


class SelfMonitoringRuleUpdate(BaseModel):
    name: str | None = None
    category: str | None = None
    patterns: list[str] | None = None
    pattern_type: Literal["literal", "regex"] | None = None
    except_patterns: list[str] | None = None
    rule_type: Literal["block", "notify"] | None = None
    action: Literal["block", "redact", "notify"] | None = None
    phase: Literal["input", "output", "both"] | None = None
    severity: Literal["info", "warning", "critical"] | None = None
    display_mode: Literal["inline_banner", "sidebar_note", "post_session_summary", "silent_log"] | None = None
    block_message: str | None = None
    context_note: str | None = None
    notification_frequency: Literal[
        "every_message", "once_per_conversation", "once_per_day", "once_per_session"
    ] | None = None
    notification_channels: list[str] | None = None
    webhook_url: str | None = None
    trusted_contact_email: str | None = None
    crisis_resources_enabled: bool | None = None
    cooldown_minutes: int | None = None
    bypass_protection: Literal["none", "cooldown", "confirmation", "partner_approval"] | None = None
    bypass_partner_user_id: str | None = None
    escalation_session_threshold: int | None = None
    escalation_session_action: str | None = None
    escalation_window_days: int | None = None
    escalation_window_threshold: int | None = None
    escalation_window_action: str | None = None
    min_context_length: int | None = None
    governance_policy_id: str | None = None
    enabled: bool | None = None


class SelfMonitoringRuleResponse(BaseModel):
    id: str
    user_id: str
    governance_policy_id: str | None = None
    name: str
    category: str
    patterns: list[str]
    pattern_type: str
    except_patterns: list[str] = []
    rule_type: str
    action: str
    phase: str
    severity: str
    display_mode: str
    block_message: str | None = None
    context_note: str | None = None
    notification_frequency: str
    notification_channels: list[str]
    webhook_url: str | None = None
    trusted_contact_email: str | None = None
    crisis_resources_enabled: bool
    cooldown_minutes: int
    bypass_protection: str
    bypass_partner_user_id: str | None = None
    escalation_session_threshold: int = 0
    escalation_session_action: str | None = None
    escalation_window_days: int = 0
    escalation_window_threshold: int = 0
    escalation_window_action: str | None = None
    min_context_length: int = 0
    enabled: bool
    can_disable: bool = Field(True, description="Whether rule can currently be disabled (cooldown check)")
    pending_deactivation_at: str | None = None
    created_at: str
    updated_at: str
    model_config = ConfigDict(from_attributes=True)


class SelfMonitoringRuleList(BaseModel):
    items: list[SelfMonitoringRuleResponse]
    total: int


class SelfMonitoringAlertResponse(BaseModel):
    id: str
    rule_id: str
    rule_name: str
    category: str
    severity: str
    matched_pattern: str
    context_snippet: str | None = None
    snippet_mode: str = "full_snippet"
    conversation_id: str | None = None
    session_id: str | None = None
    chat_type: str | None = None
    phase: str = "input"
    action_taken: str = "notified"
    notification_sent: bool
    notification_channels_used: list[str]
    crisis_resources_shown: bool
    display_mode: str = "inline_banner"
    escalation_info: dict[str, Any] | None = None
    is_read: bool = False
    created_at: str
    model_config = ConfigDict(from_attributes=True)


class SelfMonitoringAlertList(BaseModel):
    items: list[SelfMonitoringAlertResponse]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "SelfMonitoringAlertList":
        return default_offset_pagination_aliases(self)

    @model_validator(mode="after")
    def _validate_pagination_aliases(self) -> "SelfMonitoringAlertList":
        return validate_offset_pagination_aliases(self)


class MarkAlertsReadRequest(BaseModel):
    alert_ids: list[str] = Field(..., description="Alert IDs to mark as read")


# ── Crisis Resources ─────────────────────────────────────────

class CrisisResource(BaseModel):
    name: str
    description: str
    contact: str
    url: str | None = None
    available_24_7: bool = True


class CrisisResourceList(BaseModel):
    resources: list[CrisisResource]
    disclaimer: str = (
        "tldw is not a crisis service. If you are in immediate danger, "
        "please call emergency services (911) or contact the resources listed. "
        "These resources are provided for informational purposes only."
    )


class DeactivationConfirmRequest(BaseModel):
    token: str = Field(..., description="Confirmation token from deactivation request")


class DeactivationApproveRequest(BaseModel):
    token: str = Field(..., description="Confirmation token from deactivation request")


class DetailResponse(BaseModel):
    detail: str
    model_config = ConfigDict(from_attributes=True)
