"""Pydantic schemas for the RPG runtime REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class RPGCampaignCreateRequest(BaseModel):
    """Request body for creating an RPG campaign."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    default_adapter_key: str = Field(min_length=1, max_length=80)


class RPGCampaignResponse(BaseModel):
    """Serialized RPG campaign metadata."""

    id: int
    title: str
    description: str | None
    default_adapter_key: str
    default_adapter_version: str
    status: str
    version: int


class RPGSessionCreateRequest(BaseModel):
    """Request body for creating an RPG session."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    adapter_key: str = Field(min_length=1, max_length=80)


class RPGSessionResponse(BaseModel):
    """Serialized RPG session metadata."""

    id: int
    campaign_id: int
    title: str
    status: str
    adapter_key: str
    adapter_version: str
    current_snapshot_version: int
    last_event_sequence: int
    version: int


class RPGEventInput(BaseModel):
    """Client-supplied RPG event envelope."""

    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(min_length=1, max_length=120)
    event_payload: dict[str, Any]


class RPGRecordEventsRequest(BaseModel):
    """Request body for recording RPG session events."""

    model_config = ConfigDict(extra="forbid")

    expected_last_event_sequence: int = Field(ge=0)
    events: list[RPGEventInput] = Field(min_length=1, max_length=20)


class RPGRecordEventsResponse(BaseModel):
    """Response body for event recording and proposal workflows."""

    committed_events: list[dict[str, Any]]
    proposal: dict[str, Any] | None


class RPGRulesLookupRequest(BaseModel):
    """Request body for RPG rules lookup or grounded answer generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    mode: Literal["lookup", "answer"] = "lookup"
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=600, ge=64, le=2000)


class RPGRulesPackRefInput(BaseModel):
    """Client-supplied rules-pack source reference."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["media_item", "media_collection"]
    source_id: int = Field(gt=0)
    display_name: str | None = None
    enabled: StrictBool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RPGRulesPackRefsReplaceRequest(BaseModel):
    """Whole-list rules-pack reference replacement request."""

    model_config = ConfigDict(extra="forbid")

    refs: list[RPGRulesPackRefInput] = Field(default_factory=list, max_length=50)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=128)


class RPGRulesPackRefResponse(BaseModel):
    """Serialized normalized rules-pack reference."""

    ref_id: str
    source_type: Literal["media_item", "media_collection"]
    source_id: int
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class RPGRulesPackRefsResponse(BaseModel):
    """Response body for rules-pack reference list operations."""

    refs: list[RPGRulesPackRefResponse]
    version: int
    replayed: bool = False


class RPGRuleCitationResponse(BaseModel):
    """Citation metadata for a returned rules lookup item."""

    source_type: str
    source_id: int | None = None
    source_title: str
    source_url: str | None
    license: str | None
    license_url: str | None
    attribution: str | None
    trust_level: str
    content_hash: str
    snippet_id: str
    adapter_key: str | None = None
    source_version: str | None = None
    content_pack_version: str | None = None


class RPGRuleLookupItemResponse(BaseModel):
    """Single rules lookup result item."""

    origin: Literal["user_provided", "bundled_citation"]
    text: str
    citation: RPGRuleCitationResponse
    score: float


class RPGRulesLookupDiagnostics(BaseModel):
    """Diagnostics describing rules lookup source selection and retrieval."""

    model_config = ConfigDict(extra="allow")

    bundled_policy: Literal["citations_only", "no_match"]
    result_mode: str
    linked_rules_pack_count: int
    enabled_rules_pack_count: int = 0
    ready_media_item_count: int = 0
    retrieval_result_count: int = 0
    bundled_citation_count: int = 0
    skipped_refs: list[dict[str, Any]] = Field(default_factory=list)
    broad_fallback_used: bool = False


class RPGRulesLookupResponse(BaseModel):
    """Response body for rules lookup and answer mode."""

    query: str
    mode: Literal["lookup", "answer"]
    results: list[RPGRuleLookupItemResponse]
    answer: str | None = None
    answer_status: str
    answer_citation_ids: list[str] = Field(default_factory=list)
    diagnostics: RPGRulesLookupDiagnostics


class RPGContextBuildRequest(BaseModel):
    """Request body for building an RPG session context bundle."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=500)
    max_chars: int = Field(default=24_000, ge=1000, le=24_000)


class RPGContextDiagnostics(BaseModel):
    """Diagnostics for context construction and truncation."""

    model_config = ConfigDict(extra="allow")

    truncated: bool
    max_chars: int
    original_chars: int
    returned_chars: int
    rules_result_count: int
    rules_lookup: dict[str, Any] = Field(default_factory=dict)
    omitted_sections: list[str]


class RPGContextResponse(BaseModel):
    """Response body for RPG session context construction."""

    text: str
    diagnostics: RPGContextDiagnostics


class RPGProposalApplyRequest(BaseModel):
    """Request body for applying a pending RPG proposal."""

    model_config = ConfigDict(extra="forbid")

    expected_last_event_sequence: int = Field(ge=0)
    review_notes: str | None = Field(default=None, max_length=2000)


class RPGProposalRejectRequest(BaseModel):
    """Request body for rejecting a pending RPG proposal."""

    model_config = ConfigDict(extra="forbid")

    review_notes: str | None = Field(default=None, max_length=2000)


class RPGCampaignUpdateRequest(BaseModel):
    """Request body for PATCH /rpg/campaigns/{campaign_id}."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "archived"] | None = None
    expected_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _require_mutable_field(self) -> "RPGCampaignUpdateRequest":
        if self.title is None and self.description is None and self.status is None:
            raise ValueError("no_mutable_fields")
        return self


class RPGSessionUpdateRequest(BaseModel):
    """Request body for PATCH /rpg/sessions/{session_id}."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    status: Literal["active", "archived"] | None = None
    expected_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _require_mutable_field(self) -> "RPGSessionUpdateRequest":
        if self.title is None and self.status is None:
            raise ValueError("no_mutable_fields")
        return self


class RPGAuthoritySettingsReplaceRequest(BaseModel):
    """Request body for PUT /rpg/sessions/{session_id}/authority-settings."""

    model_config = ConfigDict(extra="forbid")

    import_auto_commit: StrictBool
    mcp_auto_commit: StrictBool
    model_auto_commit: StrictBool
    model_auto_commit_event_types: list[Annotated[str, Field(min_length=1, max_length=120)]]
    expected_version: int = Field(ge=1)


class RPGCampaignListItem(RPGCampaignResponse):
    """Campaign row as returned in list endpoints."""

    created_at: datetime
    updated_at: datetime


class RPGCampaignDetail(RPGCampaignListItem):
    """Campaign row as returned by the single-campaign read endpoint."""

    session_count: int


class RPGSessionListItem(RPGSessionResponse):
    """Session row as returned in list endpoints."""

    pending_proposal_count: int
    created_at: datetime
    updated_at: datetime


class RPGSessionDetail(RPGSessionListItem):
    """Session row as returned by the single-session read endpoint."""

    authority_settings: dict[str, Any]
    linked_chat_id: int | None


class RPGCampaignListResponse(BaseModel):
    """Response body for campaign list endpoints."""

    items: list[RPGCampaignListItem]
    total_count: int
    limit: int
    offset: int


class RPGSessionListResponse(BaseModel):
    """Response body for session list endpoints."""

    items: list[RPGSessionListItem]
    total_count: int
    limit: int
    offset: int


class RPGEventResponse(BaseModel):
    """Serialized RPG session event, mirroring RPGSessionEvent."""

    id: int
    session_id: int
    sequence_number: int
    event_type: str
    event_payload: dict[str, Any]
    source_type: str
    source_actor_id: str | None
    source_label: str | None
    operation_id: int | None
    event_schema_version: str
    adapter_key: str
    adapter_version: str
    proposal_id: int | None
    created_at: datetime


class RPGEventsListResponse(BaseModel):
    """Response body for session event listing."""

    items: list[RPGEventResponse]
    next_after_sequence: int
    limit: int


class RPGProposalListResponse(BaseModel):
    """Response body for session proposal listing."""

    items: list[dict[str, Any]]
    total_count: int
    limit: int
    offset: int


class RPGSnapshotResponse(BaseModel):
    """Response body for reading a session snapshot."""

    snapshot_version: int
    last_event_sequence: int
    snapshot: dict[str, Any]
    diagnostics: dict[str, Any]


class RPGSnapshotRebuildResponse(BaseModel):
    """Response body for rebuilding a session snapshot."""

    snapshot_version: int
    last_event_sequence: int
    event_count: int
    replayed: bool
    diagnostics: dict[str, Any]
