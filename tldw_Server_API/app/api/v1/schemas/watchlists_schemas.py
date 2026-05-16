from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

SourceType = Literal["rss", "site", "forum"]  # forums are feature-flagged for Phase 3
WatchlistDomain = Literal["cti_osint", "news", "general"]
WatchlistStatus = Literal["active", "paused", "archived"]
WatchlistPriority = Literal["low", "medium", "high", "critical"]
WatchlistContentAlertRuleKind = Literal["keyword", "regex", "descriptor", "classification", "entity", "ioc", "cve"]
WatchlistContentAlertMatchMode = Literal["contains", "exact", "regex"]
WatchlistContentAlertSeverity = Literal["info", "low", "medium", "high", "critical"]
WatchlistContentAlertStatus = Literal["unread", "read", "dismissed"]
WatchlistReportPreset = Literal["auto", "cti_osint", "news_briefing", "general_research"]
WatchlistReportReadinessState = Literal["ready", "warning", "blocked", "legacy_live_only"]
WatchlistReportReadinessWarningSeverity = Literal["info", "warning", "blocking"]
ScrapedItemMutableStatus = Literal["ingested", "filtered", "ignored", "reviewed"]
ScrapedItemSavedViewSmartFilter = Literal["all", "today", "today_unread", "todayUnread", "unread", "reviewed", "queued"]
ScrapedItemSortMode = Literal[
    "created_desc",
    "created_asc",
    "published_desc",
    "published_asc",
    "unread_first",
    "source_asc",
    "alert_severity_desc",
]


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


# --------------------
# Job Filters
# --------------------
FilterType = Literal["keyword", "author", "date_range", "regex", "all"]
FilterAction = Literal["include", "exclude", "flag"]


class WatchlistFilter(BaseModel):
    type: FilterType
    action: FilterAction
    value: dict[str, Any] = Field(default_factory=dict)
    priority: int | None = None
    is_active: bool = True


class WatchlistFiltersPayload(BaseModel):
    filters: list[WatchlistFilter] = Field(default_factory=list)
    require_include: bool | None = Field(
        default=None,
        description=(
            "Include-only gating toggle. When true and any include rules exist, only include-"
            "matched items are ingested; others are treated as filtered. If unset, the job"
            " inherits the organization default when available (organizations.metadata.watchlists."
            "require_include_default or flat key watchlists_require_include_default) and finally"
            " the WATCHLISTS_REQUIRE_INCLUDE_DEFAULT env var."
        ),
    )


class WatchlistIngestPrefs(BaseModel):
    persist_to_media_db: bool = Field(
        default=False,
        description="When true, also persist ingested items to the Media DB.",
    )


class WatchlistCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    objective: str | None = None
    domain: WatchlistDomain = "general"
    status: WatchlistStatus = "active"
    priority: WatchlistPriority = "medium"
    tags: list[str] = Field(default_factory=list)


class WatchlistUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    objective: str | None = None
    domain: WatchlistDomain | None = None
    status: WatchlistStatus | None = None
    priority: WatchlistPriority | None = None
    tags: list[str] | None = None


class WatchlistContainer(BaseModel):
    id: int
    name: str
    description: str | None = None
    objective: str | None = None
    domain: WatchlistDomain
    status: WatchlistStatus
    priority: WatchlistPriority
    tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    archived_at: str | None = None
    deleted_at: str | None = None
    restore_expires_at: str | None = None


class WatchlistsListResponse(BaseModel):
    items: list[WatchlistContainer]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class WatchlistContentAlertRuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    rule_kind: WatchlistContentAlertRuleKind
    match_mode: WatchlistContentAlertMatchMode = "contains"
    pattern: str = Field(..., min_length=1, max_length=1000)
    severity: WatchlistContentAlertSeverity = "medium"
    enabled: bool = True
    classification: str | None = Field(default=None, max_length=200)
    descriptor: str | None = Field(default=None, max_length=200)
    entity_type: str | None = Field(default=None, max_length=200)
    source_constraints: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("name", "pattern")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class WatchlistContentAlertRuleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    enabled: bool | None = None
    rule_kind: WatchlistContentAlertRuleKind | None = None
    match_mode: WatchlistContentAlertMatchMode | None = None
    pattern: str | None = Field(default=None, min_length=1, max_length=1000)
    severity: WatchlistContentAlertSeverity | None = None
    classification: str | None = Field(default=None, max_length=200)
    descriptor: str | None = Field(default=None, max_length=200)
    entity_type: str | None = Field(default=None, max_length=200)
    source_constraints: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("name", "pattern")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class WatchlistContentAlertRule(BaseModel):
    id: int
    watchlist_id: int
    name: str
    enabled: bool
    rule_kind: WatchlistContentAlertRuleKind
    match_mode: WatchlistContentAlertMatchMode
    pattern: str
    severity: WatchlistContentAlertSeverity
    classification: str | None = None
    descriptor: str | None = None
    entity_type: str | None = None
    source_constraints: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class WatchlistContentAlertRuleList(BaseModel):
    items: list[WatchlistContentAlertRule]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class WatchlistContentAlertUpdate(BaseModel):
    status: WatchlistContentAlertStatus


class WatchlistContentAlert(BaseModel):
    id: int
    watchlist_id: int
    rule_id: int
    item_id: int
    run_id: int
    job_id: int
    source_id: int
    severity: WatchlistContentAlertSeverity
    status: WatchlistContentAlertStatus
    title: str | None = None
    snippet: str | None = None
    matched_text: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str
    created_at: str
    read_at: str | None = None
    dismissed_at: str | None = None


class WatchlistContentAlertList(BaseModel):
    items: list[WatchlistContentAlert]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class SourceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    url: AnyUrl
    source_type: SourceType
    active: bool = True
    settings: dict[str, Any] | None = None
    tags: list[str] | None = Field(default=None, description="Tag names; server normalizes and resolves to IDs")
    group_ids: list[int] | None = None
    watchlist_id: int | None = Field(default=None, ge=1)


class SourceUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    url: AnyUrl | None = None
    source_type: SourceType | None = None
    active: bool | None = None
    settings: dict[str, Any] | None = None
    tags: list[str] | None = Field(default=None, description="Replace tags with these names")
    group_ids: list[int] | None = None


class SourceTestRequest(BaseModel):
    url: AnyUrl
    source_type: SourceType
    settings: dict[str, Any] | None = None


class Source(BaseModel):
    id: int
    name: str
    url: str
    source_type: SourceType
    active: bool
    tags: list[str] = Field(default_factory=list)
    group_ids: list[int] = Field(default_factory=list)
    watchlist_ids: list[int] = Field(default_factory=list)
    settings: dict[str, Any] | None = None
    last_scraped_at: str | None = None
    status: str | None = None
    created_at: str
    updated_at: str


class SourcesListResponse(BaseModel):
    items: list[Source]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ReversibleDeleteResponse(BaseModel):
    success: bool = True
    restore_window_seconds: int = Field(..., ge=1)
    restore_expires_at: str


class SourceDeleteResponse(ReversibleDeleteResponse):
    source_id: int


class WatchlistDeleteResponse(ReversibleDeleteResponse):
    watchlist_id: int


class SourceSeenStats(BaseModel):
    source_id: int
    user_id: int
    seen_count: int = 0
    latest_seen_at: str | None = None
    defer_until: str | None = None
    consec_not_modified: int | None = None
    recent_keys: list[str] = Field(default_factory=list)


class SourceSeenResetResponse(BaseModel):
    source_id: int
    user_id: int
    cleared: int
    cleared_backoff: bool


class SourcesCheckNowRequest(BaseModel):
    source_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="One or more source IDs to manually check now.",
    )


class SourceCheckNowItem(BaseModel):
    source_id: int
    status: Literal["ok", "error", "not_found", "inactive"]
    detail: str | None = None
    last_scraped_at: str | None = None
    run_id: int | None = None


class SourcesCheckNowResponse(BaseModel):
    items: list[SourceCheckNowItem]
    total: int
    success: int
    failed: int


class SourcesBulkCreateRequest(BaseModel):
    sources: list[SourceCreateRequest]


# --------------------
# Bulk Sources Response (per-entry status)
# --------------------
class SourcesBulkCreateItem(BaseModel):
    name: str | None = None
    url: str
    id: int | None = None
    status: Literal["created", "error"]
    error: str | None = None
    source_type: SourceType | None = None


class SourcesBulkCreateResponse(BaseModel):
    items: list[SourcesBulkCreateItem]
    total: int
    created: int
    errors: int


class GroupCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    parent_group_id: int | None = None


class GroupUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    parent_group_id: int | None = None


class Group(BaseModel):
    id: int
    name: str
    description: str | None = None
    parent_group_id: int | None = None


class GroupsListResponse(BaseModel):
    items: list[Group]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class Tag(BaseModel):
    id: int
    name: str


class TagsListResponse(BaseModel):
    items: list[Tag]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class JobCreateRequest(BaseModel):
    name: str
    description: str | None = None
    scope: dict[str, Any] = Field(default_factory=dict, description="Selection: {sources:[], groups:[], tags:[]}")
    schedule_expr: str | None = Field(None, description="Cron or interval expression; stored as provided")
    timezone: str | None = Field(None, description="Timezone for schedule; PRD default UTC")
    active: bool = True
    max_concurrency: int | None = None
    per_host_delay_ms: int | None = None
    retry_policy: dict[str, Any] | None = None
    output_prefs: dict[str, Any] | None = None
    ingest_prefs: WatchlistIngestPrefs | None = Field(
        default=None,
        description="Optional ingest preferences (e.g., Media DB persistence).",
    )
    job_filters: WatchlistFiltersPayload | None = Field(
        default=None,
        description="Optional job-level filters payload (bridge from SUBS Import Rules)",
    )
    watchlist_id: int | None = Field(default=None, ge=1)


class JobUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    scope: dict[str, Any] | None = None
    schedule_expr: str | None = None
    timezone: str | None = None
    active: bool | None = None
    max_concurrency: int | None = None
    per_host_delay_ms: int | None = None
    retry_policy: dict[str, Any] | None = None
    output_prefs: dict[str, Any] | None = None
    ingest_prefs: WatchlistIngestPrefs | None = Field(
        default=None,
        description="Optional ingest preferences (replace).",
    )
    job_filters: WatchlistFiltersPayload | None = Field(
        default=None,
        description="Optional job-level filters payload (replace)",
    )
    watchlist_id: int | None = Field(default=None, ge=1)


class Job(BaseModel):
    id: int
    name: str
    description: str | None = None
    watchlist_id: int | None = None
    scope: dict[str, Any]
    schedule_expr: str | None
    timezone: str | None
    active: bool
    max_concurrency: int | None = None
    per_host_delay_ms: int | None = None
    retry_policy: dict[str, Any] | None = None
    output_prefs: dict[str, Any] | None = None
    ingest_prefs: WatchlistIngestPrefs | None = None
    job_filters: WatchlistFiltersPayload | None = None
    created_at: str
    updated_at: str
    last_run_at: str | None = None
    next_run_at: str | None = None
    wf_schedule_id: str | None = None


class JobsListResponse(BaseModel):
    items: list[Job]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class JobDeleteResponse(ReversibleDeleteResponse):
    job_id: int


class Run(BaseModel):
    id: int
    job_id: int
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    stats: dict[str, Any] | None = None
    error_msg: str | None = None


class RunCancelResponse(BaseModel):
    run_id: int
    status: str
    cancelled: bool
    message: str | None = None


class RunsListResponse(BaseModel):
    items: list[Run]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class WatchlistOnboardingTelemetryIngestRequest(BaseModel):
    session_id: str = Field(..., max_length=128)
    event_type: str = Field(..., max_length=128)
    event_at: str | None = None
    details: dict[str, Any] | None = None


class WatchlistOnboardingTelemetryIngestResponse(BaseModel):
    accepted: bool = True
    code: str | None = None


class WatchlistOnboardingTelemetrySummaryResponse(BaseModel):
    counters: dict[str, int] = Field(default_factory=dict)
    rates: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    since: str | None = None
    until: str | None = None


WatchlistIaExperimentVariant = Literal["baseline", "experimental"]


class WatchlistIaExperimentTelemetryIngestRequest(BaseModel):
    variant: WatchlistIaExperimentVariant = "experimental"
    session_id: str = Field(..., min_length=8, max_length=128)
    previous_tab: str | None = Field(default=None, max_length=64)
    current_tab: str = Field(..., min_length=1, max_length=64)
    transitions: int = Field(default=0, ge=0, le=100_000)
    visited_tabs: list[str] = Field(default_factory=list, max_length=64)
    first_seen_at: str | None = None
    last_seen_at: str | None = None


class WatchlistIaExperimentTelemetryIngestResponse(BaseModel):
    accepted: bool = True


class WatchlistIaExperimentVariantSummary(BaseModel):
    variant: WatchlistIaExperimentVariant
    events: int = 0
    sessions: int = 0
    reached_target_sessions: int = 0
    avg_transitions: float = 0.0
    avg_visited_tabs: float = 0.0
    avg_session_seconds: float = 0.0


class WatchlistIaExperimentTelemetrySummaryResponse(BaseModel):
    items: list[WatchlistIaExperimentVariantSummary]
    since: str | None = None
    until: str | None = None


class WatchlistTelemetryThresholdSummary(BaseModel):
    id: str
    label: str
    status: Literal["ok", "potential_breach"] = "ok"
    reporting_only: bool = True
    metric_value: float | None = None
    baseline_value: float | None = None
    delta: float | None = None
    notes: str | None = None


class WatchlistRcTelemetrySummaryResponse(BaseModel):
    onboarding: WatchlistOnboardingTelemetrySummaryResponse
    uc2_backend: dict[str, Any] = Field(default_factory=dict)
    ia_experiment: dict[str, Any] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)
    thresholds: list[WatchlistTelemetryThresholdSummary] = Field(default_factory=list)
    since: str | None = None
    until: str | None = None


class PreviewItem(BaseModel):
    source_id: int
    source_type: SourceType
    url: str | None = None
    title: str | None = None
    summary: str | None = None
    published_at: str | None = None
    decision: Literal["ingest", "filtered"]
    matched_action: FilterAction | None = None
    matched_filter_key: str | None = None
    matched_filter_id: int | None = None
    matched_filter_type: FilterType | None = None
    flagged: bool = False


class PreviewResponse(BaseModel):
    items: list[PreviewItem]
    total: int
    pagination: OffsetPaginationMeta
    ingestable: int
    filtered: int


class RunDetail(BaseModel):
    id: int
    job_id: int
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    stats: dict[str, int] = Field(default_factory=lambda: {"items_found": 0, "items_ingested": 0})
    filter_tallies: dict[str, int] | None = None
    error_msg: str | None = None
    log_text: str | None = None
    log_path: str | None = None
    truncated: bool = False
    filtered_sample: list[dict[str, Any]] | None = None
    audio_briefing_limit: int | None = None
    audio_briefing_items_total: int | None = None
    audio_briefing_items_used: int | None = None
    audio_briefing_truncated: bool = False


class ScrapedItemAlertSummary(BaseModel):
    total: int = 0
    unread: int = 0
    read: int = 0
    dismissed: int = 0
    highest_severity: WatchlistContentAlertSeverity | None = None
    latest_alert_id: int | None = None
    latest_alert_status: WatchlistContentAlertStatus | None = None
    latest_alert_created_at: str | None = None
    latest_matched_text: str | None = None
    rule_ids: list[int] = Field(default_factory=list)
    severities: list[WatchlistContentAlertSeverity] = Field(default_factory=list)


class ScrapedItem(BaseModel):
    id: int
    run_id: int
    job_id: int
    source_id: int
    media_id: int | None = None
    media_uuid: UUID | None = None
    url: str | None = None
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    published_at: str | None = None
    tags: list[str] = []
    status: str
    reviewed: bool
    queued_for_briefing: bool = False
    created_at: str
    alert_summary: ScrapedItemAlertSummary | None = None


class ScrapedItemsListResponse(BaseModel):
    items: list[ScrapedItem]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ScrapedItemSmartCountsResponse(BaseModel):
    all: int = 0
    today: int = 0
    today_unread: int = 0
    unread: int = 0
    reviewed: int = 0
    queued: int = 0


class ScrapedItemUpdateRequest(BaseModel):
    reviewed: bool | None = None
    status: str | None = Field(None, description="Optional status override (e.g., reviewed, ignored)")
    queued_for_briefing: bool | None = Field(
        None,
        description="Optional explicit queue toggle for briefing/report inclusion.",
    )


class ScrapedItemBatchScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int | None = Field(default=None, ge=1)
    job_id: int | None = Field(default=None, ge=1)
    source_id: int | None = Field(default=None, ge=1)
    status: str | None = None
    reviewed: bool | None = None
    queued_for_briefing: bool | None = None
    q: str | None = Field(default=None, description="Search by title/summary/content substring")
    search: str | None = Field(default=None, description="Alias for q")
    since: str | None = None
    until: str | None = None
    has_alert: bool | None = None
    alert_status: WatchlistContentAlertStatus | None = None
    alert_severity: WatchlistContentAlertSeverity | None = None
    alert_rule_id: int | None = Field(default=None, ge=1)


class ScrapedItemBatchUpdateRequest(BaseModel):
    watchlist_id: int = Field(..., ge=1)
    item_ids: list[int] | None = Field(default=None, description="Explicit selected item IDs to update")
    scope: ScrapedItemBatchScope | None = Field(default=None, description="Filter scope for all-filtered updates")
    reviewed: bool | None = None
    status: ScrapedItemMutableStatus | None = Field(default=None, description="Explicit item status update")
    queued_for_briefing: bool | None = None
    limit: int = Field(default=500, ge=1, le=5000)

    @model_validator(mode="after")
    def _validate_batch_request(self):
        has_ids = bool(self.item_ids)
        has_scope = self.scope is not None
        if has_ids == has_scope:
            raise ValueError("provide_exactly_one_item_ids_or_scope")
        if self.reviewed is None and self.status is None and self.queued_for_briefing is None:
            raise ValueError("provide_at_least_one_update")
        if self.item_ids:
            self.item_ids = list(dict.fromkeys(int(item_id) for item_id in self.item_ids))
        return self


class ScrapedItemBatchUpdateResponse(BaseModel):
    matched: int = 0
    changed: int = 0
    unchanged: int = 0
    failed: int = 0
    matched_ids: list[int] = Field(default_factory=list)
    changed_ids: list[int] = Field(default_factory=list)
    unchanged_ids: list[int] = Field(default_factory=list)
    failed_ids: list[int] = Field(default_factory=list)
    capped: bool = False
    exhausted: bool = True
    limit: int = 500


class WatchlistItemSavedViewBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    filters: dict[str, Any] = Field(default_factory=dict)
    sort: ScrapedItemSortMode = "created_desc"
    is_default: bool = False


class WatchlistItemSavedViewCreate(WatchlistItemSavedViewBase):
    pass


class WatchlistItemSavedViewUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    filters: dict[str, Any] | None = None
    sort: ScrapedItemSortMode | None = None
    is_default: bool | None = None


class WatchlistItemSavedView(BaseModel):
    id: int
    watchlist_id: int
    name: str
    filters: dict[str, Any] = Field(default_factory=dict)
    sort: ScrapedItemSortMode
    is_default: bool = False
    created_at: str
    updated_at: str


class WatchlistItemSavedViewsList(BaseModel):
    items: list[WatchlistItemSavedView]


class WatchlistReportReadinessWarning(BaseModel):
    code: str
    severity: WatchlistReportReadinessWarningSeverity = "warning"
    message: str
    affected_item_ids: list[int] = Field(default_factory=list)


class WatchlistReportReadiness(BaseModel):
    state: WatchlistReportReadinessState
    score: int = Field(ge=0, le=100)
    warnings: list[WatchlistReportReadinessWarning] = Field(default_factory=list)


class WatchlistReportEvidenceAlert(BaseModel):
    id: int
    rule_id: int
    rule_name: str | None = None
    severity: str
    status: str
    title: str | None = None
    snippet: str | None = None
    matched_text: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class WatchlistReportEvidenceItem(BaseModel):
    id: int
    title: str | None = None
    url: str | None = None
    source_id: int | None = None
    source_name: str | None = None
    published_at: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    reviewed: bool = False
    queued_for_briefing: bool = False
    alerts: list[WatchlistReportEvidenceAlert] = Field(default_factory=list)


class WatchlistReportExcludedItem(BaseModel):
    id: int
    title: str | None = None
    url: str | None = None
    reason: str


class WatchlistReportEvidenceSnapshot(BaseModel):
    schema_version: int = 1
    snapshot_id: str
    generated_at: str
    preset: WatchlistReportPreset
    watchlist_id: int | None = None
    job_id: int
    run_id: int
    output_id: int | None = None
    included_items: list[WatchlistReportEvidenceItem] = Field(default_factory=list)
    excluded_items: list[WatchlistReportExcludedItem] = Field(default_factory=list)
    source_summary: dict[str, Any] = Field(default_factory=dict)
    included_count: int = 0
    excluded_count: int = 0
    excluded_total_count: int | None = None
    excluded_items_truncated: bool = False
    alert_count: int = 0
    critical_alert_count: int = 0
    readiness: WatchlistReportReadiness


class WatchlistOutputEvidenceResponse(BaseModel):
    output_id: int
    immutable_snapshot: bool
    snapshot: WatchlistReportEvidenceSnapshot | None = None
    readiness: WatchlistReportReadiness


class WatchlistOutputEmailDelivery(BaseModel):
    enabled: bool = True
    recipients: list[str] | None = Field(default=None, description="Explicit recipient emails; defaults to user email when empty")
    subject: str | None = Field(default=None, description="Overrides default subject (defaults to output title)")
    attach_file: bool = Field(default=True, description="Attach rendered content as a file")
    body_format: Literal["auto", "text", "html"] = Field(default="auto", description="Controls email body format")


class WatchlistOutputChatbookDelivery(BaseModel):
    enabled: bool = True
    title: str | None = Field(default=None, description="Override Chatbook document title")
    description: str | None = Field(default=None, description="Optional description stored alongside the document")
    conversation_id: int | None = Field(default=None, description="Optional conversation association")
    provider: str | None = Field(default="watchlists", description="Provider marker for the generated document")
    model: str | None = Field(default="watchlists", description="Model marker for the generated document")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata stored with the document")


class WatchlistOutputDeliveries(BaseModel):
    email: WatchlistOutputEmailDelivery | None = None
    chatbook: WatchlistOutputChatbookDelivery | None = None


class WatchlistOutputCreateRequest(BaseModel):
    run_id: int = Field(..., description="Run identifier the output is based on")
    item_ids: list[int] | None = Field(None, description="Explicit list of scraped item IDs to include")
    title: str | None = Field(None, description="Optional title to embed in the generated output")
    type: str = Field("briefing_markdown", description="Output template/type identifier")
    format: Literal["md", "html"] | None = Field(None, description="Rendered output format (overrides template)")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata stored alongside the output")
    report_preset: WatchlistReportPreset = Field(
        default="auto",
        description="Report evidence/readiness preset. Defaults to Watchlist domain when available.",
    )
    include_evidence_table: bool = Field(default=True, description="Expose evidence details to report templates")
    include_excluded_items: bool = Field(default=True, description="Capture excluded same-run items in evidence snapshots")
    require_reviewed_items: bool = Field(default=False, description="Warn when queued report items have not been reviewed")
    allow_weak_evidence: bool = Field(default=True, description="Allow generation when readiness warnings are present")
    template_name: str | None = Field(None, description="Name of a stored template to render with")
    template_version: int | None = Field(
        default=None,
        ge=1,
        description="Optional version of a watchlists template to render (supports template history).",
    )
    summarize: bool = Field(default=False, description="Generate LLM-based per-article summaries before rendering")
    summarize_prompt: str | None = Field(
        default=None,
        description="Custom prompt for per-article summarization. Default: 2-3 sentence summary.",
    )
    generate_mece: bool = Field(default=False, description="Generate a MECE variant output")
    mece_template_name: str | None = Field(default=None, description="Override template name for MECE output")
    generate_tts: bool = Field(default=False, description="Generate a TTS audio variant output")
    tts_template_name: str | None = Field(default=None, description="Override template name for TTS output")
    generate_audio: bool = Field(default=False, description="Generate a multi-voice audio briefing via workflow")
    target_audio_minutes: int = Field(default=10, ge=1, le=60, description="Target audio briefing duration in minutes")
    audio_model: str | None = Field(default=None, description="TTS model for audio briefing, e.g., 'kokoro'")
    audio_voice: str | None = Field(default=None, description="Default voice for audio briefing, e.g., 'af_heart'")
    audio_speed: float | None = Field(default=None, ge=0.25, le=4.0, description="Audio briefing speed override")
    background_audio_uri: str | None = Field(
        default=None,
        description="Optional file:// URI for a background track to mix under generated briefing audio",
    )
    background_volume: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Background track volume multiplier when mixing briefing audio",
    )
    background_delay_ms: int | None = Field(
        default=None,
        ge=0,
        le=120000,
        description="Delay before background track starts, in milliseconds",
    )
    background_fade_seconds: float | None = Field(
        default=None,
        ge=0.0,
        le=30.0,
        description="Fade-in/out duration applied to background track, in seconds",
    )
    audio_language: str | None = Field(
        default=None,
        description="Language for audio briefing script generation, e.g., 'en', 'es', 'fr'",
    )
    llm_provider: str | None = Field(default=None, description="LLM provider for summarization and script composition")
    llm_model: str | None = Field(default=None, description="LLM model for summarization and script composition")
    persona_summarize: bool = Field(
        default=False,
        description="Enable per-item persona pre-summarization before composing the final audio script",
    )
    persona_id: str | None = Field(
        default=None,
        description="Persona identifier/style hint for per-item pre-summarization",
    )
    persona_provider: str | None = Field(
        default=None,
        description="Optional provider override for persona pre-summarization",
    )
    persona_model: str | None = Field(
        default=None,
        description="Optional model override for persona pre-summarization",
    )
    voice_map: dict[str, str] | None = Field(
        default=None,
        description="Voice marker to Kokoro voice ID mapping, e.g., {'HOST': 'af_bella', 'REPORTER': 'am_adam'}",
    )
    ingest_to_media_db: bool = Field(default=False, description="Ingest outputs into Media DB")
    tts_model: str | None = Field(default=None, description="TTS model id, e.g., 'kokoro', 'tts-1'")
    tts_voice: str | None = Field(default=None, description="TTS voice id, e.g., 'af_heart'")
    tts_speed: float | None = Field(default=None, ge=0.25, le=4.0, description="TTS speed override")
    retention_seconds: int | None = Field(None, ge=0, description="Optional custom retention in seconds (0 = no expiry)")
    temporary: bool | None = Field(False, description="Whether to use temporary retention defaults")
    deliveries: WatchlistOutputDeliveries | None = Field(
        default=None,
        description="Optional delivery configuration (email, chatbook). Overrides job defaults when provided.",
    )
    grouping: GroupingConfig | None = Field(
        default=None,
        description="Backend grouping configuration for items",
    )
    briefing_summary: BriefingSummaryConfig | None = Field(
        default=None,
        description="Briefing-level LLM summary configuration (always async)",
    )


class WatchlistOutput(BaseModel):
    id: int
    run_id: int
    job_id: int
    type: str
    format: str
    title: str | None = None
    content: str | None = None
    storage_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    media_item_id: int | None = None
    chatbook_path: str | None = None
    version: int
    expires_at: str | None = None
    expired: bool = False
    created_at: str


class WatchlistOutputsListResponse(BaseModel):
    items: list[WatchlistOutput]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class WatchlistTemplateSummary(BaseModel):
    name: str
    format: Literal["md", "html"]
    description: str | None = None
    updated_at: str
    version: int = 1
    history_count: int = 0
    composer_ast: dict[str, Any] | None = None
    composer_schema_version: str | None = None
    composer_sync_hash: str | None = None
    composer_sync_status: Literal["in_sync", "needs_repair", "recovered_from_code"] | None = None


class WatchlistTemplateDetail(WatchlistTemplateSummary):
    content: str
    available_versions: list[int] = Field(default_factory=list)


class WatchlistTemplateVersionSummary(BaseModel):
    version: int
    format: Literal["md", "html"]
    description: str | None = None
    updated_at: str
    is_current: bool = False


class WatchlistTemplateVersionsResponse(BaseModel):
    items: list[WatchlistTemplateVersionSummary]


class WatchlistTemplateListResponse(BaseModel):
    items: list[WatchlistTemplateSummary]


class WatchlistTemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    format: Literal["md", "html"] = "md"
    content: str = Field(..., description="Template content")
    description: str | None = Field(None, description="Optional human-readable description")
    overwrite: bool = Field(False, description="If false, creation fails when template already exists")
    composer_ast: dict[str, Any] | None = Field(
        None,
        description="Optional visual composer AST representation for template editing.",
    )
    composer_schema_version: str | None = Field(
        None,
        description="Optional semantic version for the composer AST schema.",
    )
    composer_sync_hash: str | None = Field(
        None,
        description="Optional sync checksum for Jinja content and composer AST parity.",
    )
    composer_sync_status: Literal["in_sync", "needs_repair", "recovered_from_code"] | None = Field(
        None,
        description="Optional parity status between content and composer AST.",
    )


class WatchlistTemplateValidationErrorDetail(BaseModel):
    error: Literal["template_validation_error"] = "template_validation_error"
    message: str = Field(..., description="Validation error message")


class WatchlistTemplateValidationErrorResponse(BaseModel):
    detail: WatchlistTemplateValidationErrorDetail


# --------------------
# OPML Import/Export
# --------------------
class SourcesImportItem(BaseModel):
    url: str
    name: str | None = None
    id: int | None = None
    status: Literal["created", "skipped", "error"]
    error: str | None = None


class SourcesImportResponse(BaseModel):
    items: list[SourcesImportItem]
    total: int
    created: int
    skipped: int
    errors: int


# --------------------
# Grouping & Briefing Summary
# --------------------
class GroupingRule(BaseModel):
    group_name: str
    match_field: Literal["tag", "source_id", "url", "title"] = "tag"
    match_pattern: str
    match_mode: Literal["exact", "contains", "regex"] = "exact"


class GroupingConfig(BaseModel):
    group_by: Literal["tag", "source", "topic", "custom"] = "tag"
    multi_tag_mode: Literal["primary", "duplicate"] = "primary"
    custom_rules: list[GroupingRule] | None = None
    ungrouped_label: str = "Uncategorized"
    sort_groups_by: Literal["name", "item_count"] = "name"
    topic_llm_provider: str | None = None
    topic_llm_model: str | None = None
    max_groups: int = Field(default=7, ge=2, le=20)


class BriefingSummaryConfig(BaseModel):
    enabled: bool = True
    prompt: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    per_group_summaries: bool = False
    per_group_prompt: str | None = None
    max_items_for_direct_summary: int = Field(default=30, ge=1, le=200)


# --------------------
# Template Preview & Validation
# --------------------
class TemplatePreviewRequest(BaseModel):
    content: str
    format: Literal["md", "html"] = "md"
    run_id: int


class TemplatePreviewResponse(BaseModel):
    rendered: str
    context_keys: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TemplateValidateRequest(BaseModel):
    content: str
    format: Literal["md", "html"] = "md"


class TemplateValidationErrorItem(BaseModel):
    line: int | None = None
    column: int | None = None
    message: str


class TemplateValidationResult(BaseModel):
    valid: bool
    errors: list[TemplateValidationErrorItem] = Field(default_factory=list)


# --------------------
# Template Composer Authoring (Manual Preview)
# --------------------
_TEMPLATE_COMPOSER_FLOW_MAX_TOTAL_CHARS = 120_000


class TemplateComposerSectionRequest(BaseModel):
    run_id: int = Field(..., ge=1)
    block_id: str = Field(..., min_length=1, max_length=128)
    prompt: str = Field(..., min_length=1, max_length=4000)
    input_scope: Literal["all_items", "top_items", "selected_items"] = "all_items"
    style: str | None = Field(default=None, max_length=128)
    length_target: Literal["short", "medium", "long"] = "medium"

    @field_validator("block_id", "prompt")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("style")
    @classmethod
    def _normalize_optional_style(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class TemplateComposerSectionResponse(BaseModel):
    block_id: str
    content: str
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class TemplateComposerFlowSection(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(default="", max_length=20000)

    @field_validator("id")
    @classmethod
    def _validate_non_blank_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class TemplateComposerFlowCheckRequest(BaseModel):
    run_id: int = Field(..., ge=1)
    mode: Literal["suggest_only", "auto_apply"] = "suggest_only"
    sections: list[TemplateComposerFlowSection] = Field(default_factory=list, max_length=128)

    @field_validator("sections")
    @classmethod
    def _validate_total_content_size(
        cls, value: list[TemplateComposerFlowSection]
    ) -> list[TemplateComposerFlowSection]:
        total_chars = sum(len(section.content) for section in value)
        if total_chars > _TEMPLATE_COMPOSER_FLOW_MAX_TOTAL_CHARS:
            raise ValueError(
                f"total section content exceeds maximum of {_TEMPLATE_COMPOSER_FLOW_MAX_TOTAL_CHARS} characters"
            )
        return value


class TemplateComposerFlowIssue(BaseModel):
    section_id: str | None = None
    severity: Literal["info", "warning"] = "info"
    message: str


class TemplateComposerFlowCheckResponse(BaseModel):
    mode: Literal["suggest_only", "auto_apply"]
    issues: list[TemplateComposerFlowIssue] = Field(default_factory=list)
    diff: str = ""
    sections: list[TemplateComposerFlowSection] = Field(default_factory=list)
