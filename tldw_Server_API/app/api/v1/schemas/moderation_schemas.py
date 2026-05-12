# moderation_schemas.py
# Description: Pydantic models for Moderation admin endpoints

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModerationOverrideRule(BaseModel):
    """A single phrase rule that can block or warn during input/output moderation."""

    id: str = Field(..., min_length=1, description="Stable rule identifier")
    pattern: str = Field(..., min_length=1, description="Literal phrase or regex source")
    is_regex: bool = Field(False, description="Whether pattern should be treated as regex")
    action: Literal['block', 'warn'] = Field(..., description="Per-rule moderation action")
    phase: Literal['input', 'output', 'both'] = Field(
        'both',
        description="Which moderation phase this rule should apply to",
    )


class ModerationUserOverride(BaseModel):
    enabled: Optional[bool] = Field(None, description="Enable moderation for this user")
    input_enabled: Optional[bool] = Field(None, description="Enable input moderation for this user")
    output_enabled: Optional[bool] = Field(None, description="Enable output moderation for this user")
    input_action: Optional[Literal['block', 'redact', 'warn']] = Field(None, description="Action for input violations")
    output_action: Optional[Literal['block', 'redact', 'warn']] = Field(None, description="Action for output violations")
    redact_replacement: Optional[str] = Field(None, description="Replacement text for redaction")
    categories_enabled: Optional[list[str]] = Field(
        None,
        description="Categories to enable for this user (comma-separated string or list, e.g., 'pii,confidential')",
    )
    rules: Optional[list[ModerationOverrideRule]] = Field(
        None,
        description="Optional per-user phrase rules for block or warn actions",
    )

    @field_validator("categories_enabled", mode="before")
    @classmethod
    def _normalize_categories_enabled(cls, v: Any) -> Optional[list[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            tokens = [token.strip() for token in v.split(",")]
            return [token for token in tokens if token]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return v

    @field_validator('redact_replacement')
    @classmethod
    def _non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.strip()) == 0:
            raise ValueError("redact_replacement cannot be empty if provided")
        return v


class ModerationBlocklistUpdate(BaseModel):
    lines: list[str] = Field(default_factory=list, description="Blocklist lines; regex lines can be wrapped in /.../")


class ModerationUserOverridesResponse(BaseModel):
    overrides: dict[str, dict[str, Any]]


class ModerationUserOverrideLookupResponse(BaseModel):
    exists: bool
    override: dict[str, Any]


class BlocklistManagedItem(BaseModel):
    id: int
    line: str


class BlocklistManagedResponse(BaseModel):
    version: str = Field(..., description="Content hash for optimistic concurrency")
    items: list[BlocklistManagedItem]


class BlocklistAppendRequest(BaseModel):
    line: str = Field(..., min_length=1)

    @field_validator("line")
    @classmethod
    def _single_line(cls, v: str) -> str:
        if "\n" in v or "\r" in v:
            raise ValueError("line must be a single line")
        return v


class BlocklistAppendResponse(BaseModel):
    version: str
    index: int
    count: int


class BlocklistDeleteResponse(BaseModel):
    version: str
    count: int


class BlocklistLintRequest(BaseModel):
    lines: Optional[list[str]] = None
    line: Optional[str] = None

    @field_validator('line')
    @classmethod
    def _ensure_any(cls, v, info):
        # Validation occurs after both fields parsed; check neither set later in endpoint
        return v


class BlocklistLintItem(BaseModel):
    index: int
    line: str
    ok: bool
    pattern_type: Optional[Literal['literal', 'regex', 'comment', 'empty']] = None
    action: Optional[Literal['block', 'redact', 'warn']] = None
    replacement: Optional[str] = None
    categories: Optional[list[str]] = None
    error: Optional[str] = None
    warning: Optional[str] = None
    sample: Optional[str] = None


class BlocklistLintResponse(BaseModel):
    items: list[BlocklistLintItem]
    valid_count: int
    invalid_count: int


class ModerationTestRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID to apply effective policy")
    dependent_user_id: Optional[str] = Field(
        None,
        description="Dependent user ID to simulate for guardian overlay; defaults to user_id when omitted",
    )
    chat_type: Optional[Literal["regular", "character", "persona"]] = Field(
        None,
        description="Chat type to simulate for guardian overlay; defaults to 'regular' when omitted",
    )
    apply_guardian_overlay: bool = Field(
        False,
        description="Apply guardian-supervised policy overlay using live-chat simulation rules",
    )
    phase: Literal['input', 'output'] = Field('input', description="Moderation phase to test")
    text: str = Field(..., description="Sample text to test against moderation policy")


class ModerationTestResponse(BaseModel):
    flagged: bool
    action: Literal['block', 'redact', 'warn', 'notify', 'pass']
    sample: Optional[str] = None
    redacted_text: Optional[str] = None
    effective: dict[str, Any]
    category: Optional[str] = None


class ModerationSettingsResponse(BaseModel):
    pii_enabled: Optional[bool] = Field(None, description="Runtime override for pii_enabled or None if not overridden")
    categories_enabled: Optional[list[str]] = Field(None, description="Runtime override for categories_enabled or None if not overridden")
    effective: dict[str, Any] = Field(..., description="Effective settings after merge with config")


class ModerationSettingsUpdate(BaseModel):
    pii_enabled: Optional[bool] = None
    categories_enabled: Optional[list[str]] = None
    persist: Optional[bool] = Field(False, description="Persist runtime overrides to file")


ModerationReviewStatus = Literal[
    "needs_review",
    "approved",
    "blocked",
    "redacted",
    "dismissed",
    "escalated",
]
ModerationDecisionAction = Literal["approve", "block", "redact", "dismiss", "escalate"]
ModerationSeverity = Literal["low", "medium", "high", "critical"]


class ModerationReviewMatch(BaseModel):
    rule_id: str | None = None
    pattern_type: Literal["literal", "regex", "pii", "category"] | None = None
    category: str | None = None
    action: Literal["pass", "block", "redact", "warn"] | None = None
    sample: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)


class ModerationReviewItem(BaseModel):
    id: str
    status: ModerationReviewStatus
    phase: Literal["input", "output"]
    source_type: str | None = None
    source_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    created_at: str
    updated_at: str | None = None
    severity: ModerationSeverity | None = None
    category: str | None = None
    safe_fields: dict[str, bool] = Field(default_factory=dict)
    excerpt: str
    context: dict[str, str] | None = None
    effective_policy: dict[str, Any] = Field(default_factory=dict)
    matches: list[ModerationReviewMatch] = Field(default_factory=list)
    recommended_action: ModerationDecisionAction | None = None
    retention_expires_at: str | None = None
    content_redacted_at: str | None = None


class ModerationReviewListResponse(BaseModel):
    items: list[ModerationReviewItem]
    next_cursor: str | None = None
    total: int | None = None


class ModerationReviewDecisionRequest(BaseModel):
    action: ModerationDecisionAction
    reason: str | None = Field(None, max_length=2000)
    actor_id: str | None = Field(None, description="Ignored; actor is always derived from the authenticated principal")


class ModerationReviewDecision(BaseModel):
    id: str
    item_id: str
    action: ModerationDecisionAction
    status: ModerationReviewStatus
    previous_status: ModerationReviewStatus
    decided_by: str
    reason: str | None = None
    decided_at: str
    undo_token: str | None = None


class ModerationReviewDecisionResponse(BaseModel):
    item: ModerationReviewItem
    decision: ModerationReviewDecision
    undo_token: str | None = None


class ModerationReviewUndoRequest(BaseModel):
    undo_token: str = Field(..., min_length=1)


class ModerationReviewBulkDecisionRequest(BaseModel):
    item_ids: list[str] = Field(..., min_length=1, max_length=500)
    action: ModerationDecisionAction
    reason: str | None = Field(None, max_length=2000)


class ModerationReviewBulkDecisionResult(BaseModel):
    item_id: str
    ok: bool
    item: ModerationReviewItem | None = None
    decision: ModerationReviewDecision | None = None
    undo_token: str | None = None
    error: str | None = None


class ModerationReviewBulkDecisionResponse(BaseModel):
    results: list[ModerationReviewBulkDecisionResult]
    ok_count: int
    error_count: int


class ModerationReviewAuditEvent(BaseModel):
    id: str
    item_id: str | None = None
    decision_id: str | None = None
    actor_id: str | None = None
    action: str
    summary: str | None = None
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModerationReviewAuditResponse(BaseModel):
    events: list[ModerationReviewAuditEvent]
    next_cursor: str | None = None
