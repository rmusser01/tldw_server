from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class RPGCampaignCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    default_adapter_key: str = Field(min_length=1, max_length=80)


class RPGCampaignResponse(BaseModel):
    id: int
    title: str
    description: str | None
    default_adapter_key: str
    default_adapter_version: str
    status: str
    version: int


class RPGSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    adapter_key: str = Field(min_length=1, max_length=80)


class RPGSessionResponse(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(min_length=1, max_length=120)
    event_payload: dict[str, Any]


class RPGRecordEventsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_last_event_sequence: int = Field(ge=0)
    events: list[RPGEventInput] = Field(min_length=1, max_length=20)


class RPGRecordEventsResponse(BaseModel):
    committed_events: list[dict[str, Any]]
    proposal: dict[str, Any] | None


class RPGRulesLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    mode: Literal["lookup", "answer"] = "lookup"
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=600, ge=64, le=2000)


class RPGRulesPackRefInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["media_item", "media_collection"]
    source_id: int = Field(gt=0)
    display_name: str | None = None
    enabled: StrictBool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RPGRulesPackRefsReplaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refs: list[RPGRulesPackRefInput] = Field(default_factory=list, max_length=50)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=128)


class RPGRulesPackRefResponse(BaseModel):
    ref_id: str
    source_type: Literal["media_item", "media_collection"]
    source_id: int
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class RPGRulesPackRefsResponse(BaseModel):
    refs: list[RPGRulesPackRefResponse]
    version: int
    replayed: bool = False


class RPGRuleCitationResponse(BaseModel):
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
    origin: Literal["user_provided", "bundled_citation"]
    text: str
    citation: RPGRuleCitationResponse
    score: float


class RPGRulesLookupDiagnostics(BaseModel):
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
    query: str
    mode: Literal["lookup", "answer"]
    results: list[RPGRuleLookupItemResponse]
    answer: str | None = None
    answer_status: str
    answer_citation_ids: list[str] = Field(default_factory=list)
    diagnostics: RPGRulesLookupDiagnostics


class RPGContextBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=500)
    max_chars: int = Field(default=24_000, ge=1000, le=24_000)


class RPGContextDiagnostics(BaseModel):
    model_config = ConfigDict(extra="allow")

    truncated: bool
    max_chars: int
    original_chars: int
    returned_chars: int
    rules_result_count: int
    rules_lookup: dict[str, Any] = Field(default_factory=dict)
    omitted_sections: list[str]


class RPGContextResponse(BaseModel):
    text: str
    diagnostics: RPGContextDiagnostics


class RPGProposalApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_last_event_sequence: int = Field(ge=0)
    review_notes: str | None = Field(default=None, max_length=2000)


class RPGProposalRejectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_notes: str | None = Field(default=None, max_length=2000)
