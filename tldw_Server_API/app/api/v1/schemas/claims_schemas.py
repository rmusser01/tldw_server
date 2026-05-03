from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    validate_offset_pagination_aliases,
)


def _default_offset_pagination_aliases(response):
    return validate_offset_pagination_aliases(response)


def _default_offset_pagination_navigation_aliases(response):
    pagination = getattr(response, "pagination", None)
    if pagination is None:
        return response
    for alias_name, metadata_name in (
        ("has_more", "has_more"),
        ("next_offset", "next_offset"),
    ):
        if not hasattr(response, alias_name):
            continue
        expected = getattr(pagination, metadata_name, None)
        actual = getattr(response, alias_name, None)
        if actual is None:
            setattr(response, alias_name, expected)
            continue
        if actual != expected:
            raise ValueError(
                f"{alias_name} alias mismatch: {alias_name}={actual} "
                f"pagination.{metadata_name}={expected}"
            )
    return response


class ClaimsSettingsResponse(BaseModel):
    """Current claims-related configuration values."""

    model_config = ConfigDict(extra="forbid")

    enable_ingestion_claims: bool = Field(..., description="Enable ingestion-time claim extraction.")
    claim_extractor_mode: str = Field(..., description="Default extractor mode for ingestion-time claims.")
    claims_max_per_chunk: int = Field(..., description="Maximum claims per chunk during ingestion.")
    claims_embed: bool = Field(..., description="Enable embedding of extracted claims.")
    claims_embed_model_id: str = Field(..., description="Model id for claim embeddings.")
    claims_cluster_method: str = Field(..., description="Clustering method for claim grouping.")
    claims_cluster_similarity_threshold: float = Field(
        ..., description="Cosine similarity threshold for embedding-based clustering."
    )
    claims_cluster_batch_size: int = Field(..., description="Batch size for loading claim embeddings.")
    claims_llm_provider: str = Field(..., description="LLM provider for claim extraction.")
    claims_llm_temperature: float = Field(..., description="LLM temperature for claim extraction.")
    claims_llm_model: str = Field(..., description="LLM model for claim extraction.")
    claims_json_parse_mode: Literal["lenient", "strict"] = Field(
        ...,
        description="JSON parsing strictness for LLM claim outputs.",
    )
    claims_prompt_validation_mode: Literal["off", "warning", "error"] = Field(
        ...,
        description="Startup/CI prompt preflight validation mode.",
    )
    claims_prompt_validation_strict: bool = Field(
        ...,
        description="Treat non-exact prompt alignment checks as strict failures.",
    )
    claims_alignment_mode: Literal["off", "exact", "fuzzy"] = Field(
        ...,
        description="Span alignment mode for extracted claims.",
    )
    claims_alignment_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fuzzy alignment threshold for extracted claim spans.",
    )
    claims_context_window_chars: int = Field(
        ...,
        ge=0,
        description="Trailing chars from previous chunk to include as extraction context.",
    )
    claims_extraction_passes: int = Field(
        ...,
        ge=1,
        description="Sequential extraction passes for llm/aps modes.",
    )
    claims_rebuild_enabled: bool = Field(..., description="Enable periodic claims rebuild worker.")
    claims_rebuild_interval_sec: int = Field(..., description="Claims rebuild loop interval in seconds.")
    claims_rebuild_policy: str = Field(..., description="Claims rebuild policy.")
    claims_stale_days: int = Field(..., description="Stale threshold for claims rebuild policy.")


class ClaimsSettingsUpdate(BaseModel):
    """Update payload for claims settings."""

    model_config = ConfigDict(extra="forbid")

    enable_ingestion_claims: bool | None = Field(default=None)
    claim_extractor_mode: str | None = Field(default=None)
    claims_max_per_chunk: int | None = Field(default=None, ge=1, le=100)
    claims_embed: bool | None = Field(default=None)
    claims_embed_model_id: str | None = Field(default=None)
    claims_cluster_method: str | None = Field(default=None)
    claims_cluster_similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    claims_cluster_batch_size: int | None = Field(default=None, ge=1, le=10000)
    claims_llm_provider: str | None = Field(default=None)
    claims_llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    claims_llm_model: str | None = Field(default=None)
    claims_json_parse_mode: Literal["lenient", "strict"] | None = Field(default=None)
    claims_prompt_validation_mode: Literal["off", "warning", "error"] | None = Field(default=None)
    claims_prompt_validation_strict: bool | None = Field(default=None)
    claims_alignment_mode: Literal["off", "exact", "fuzzy"] | None = Field(default=None)
    claims_alignment_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    claims_context_window_chars: int | None = Field(default=None, ge=0, le=20000)
    claims_extraction_passes: int | None = Field(default=None, ge=1, le=10)
    claims_rebuild_enabled: bool | None = Field(default=None)
    claims_rebuild_interval_sec: int | None = Field(default=None, ge=60, le=604800)
    claims_rebuild_policy: str | None = Field(default=None)
    claims_stale_days: int | None = Field(default=None, ge=1, le=3650)
    persist: bool | None = Field(default=None, description="Persist updates to config.txt.")


class ClaimsExtractorCatalogItem(BaseModel):
    """Claims extractor catalog entry."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    label: str
    description: str
    execution: str
    supports_languages: list[str] | None = None
    providers: list[str] | None = None
    auto_selectable: bool | None = None


class ClaimsExtractorCatalogResponse(BaseModel):
    """Claims extractor catalog response."""

    model_config = ConfigDict(extra="forbid")

    extractors: list[ClaimsExtractorCatalogItem] = Field(default_factory=list)
    default_mode: str
    auto_mode: str


class ClaimUpdateRequest(BaseModel):
    """Update fields for a claim entry."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str | None = Field(default=None)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor: str | None = Field(default=None)
    extractor_version: str | None = Field(default=None)
    deleted: bool | None = Field(default=None)


class ClaimsSearchResult(BaseModel):
    """Claims search result row."""

    model_config = ConfigDict(extra="forbid")

    id: int
    media_id: int
    chunk_index: int
    claim_text: str
    claim_cluster_id: int | None = None
    relevance_score: float | None = None


class ClaimsSearchClusterResult(BaseModel):
    """Clustered claims search result."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    canonical_claim_text: str | None = None
    representative_claim_id: int | None = None
    watchlist_count: int | None = None
    match_count: int
    top_claim: ClaimsSearchResult


class ClaimsSearchResponse(BaseModel):
    """Claims search response payload."""

    model_config = ConfigDict(extra="forbid")

    query: str
    group_by_cluster: bool
    total: int
    results: list[ClaimsSearchResult] = Field(default_factory=list)
    clusters: list[ClaimsSearchClusterResult] | None = None
    orphaned: list[ClaimsSearchResult] | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ClaimsClusterLinkCreate(BaseModel):
    """Create a cluster relationship link."""

    model_config = ConfigDict(extra="forbid")

    child_cluster_id: int
    relation_type: str | None = Field(default=None)


class ClaimsClusterLinkResponse(BaseModel):
    """Cluster relationship link response."""

    model_config = ConfigDict(extra="forbid")

    parent_cluster_id: int
    child_cluster_id: int
    relation_type: str | None = None
    created_at: str | None = None
    direction: str | None = None


class ClaimsMonitoringSettingsResponse(BaseModel):
    """Claims monitoring configuration."""

    model_config = ConfigDict(extra="forbid")

    id: int
    user_id: str
    threshold_ratio: float = Field(..., ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    slack_webhook_url: str | None = Field(default=None)
    webhook_url: str | None = Field(default=None)
    email_recipients: list[str] = Field(default_factory=list)
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None


class ClaimsMonitoringSettingsUpdate(BaseModel):
    """Update monitoring config."""

    model_config = ConfigDict(extra="forbid")

    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    slack_webhook_url: str | None = Field(default=None)
    webhook_url: str | None = Field(default=None)
    email_recipients: list[str] | None = Field(default=None)
    enabled: bool | None = Field(default=None)
    persist: bool | None = Field(default=None, description="Legacy no-op.")


class ClaimsAlertConfigResponse(BaseModel):
    """Alert configuration for claims monitoring."""

    model_config = ConfigDict(extra="forbid")

    id: int
    user_id: str
    name: str
    alert_type: str
    threshold_ratio: float | None = None
    baseline_ratio: float | None = None
    channels: dict[str, bool] = Field(default_factory=dict)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] = Field(default_factory=list)
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None


class ClaimsAlertConfigCreate(BaseModel):
    """Create alert configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    alert_type: str = Field(..., min_length=1)
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    channels: dict[str, bool] = Field(default_factory=dict)
    slack_webhook_url: str | None = Field(default=None)
    webhook_url: str | None = Field(default=None)
    email_recipients: list[str] | None = Field(default=None)
    enabled: bool | None = Field(default=None)


class ClaimsAlertConfigUpdate(BaseModel):
    """Update alert configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1)
    alert_type: str | None = Field(default=None, min_length=1)
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    channels: dict[str, bool] | None = Field(default=None)
    slack_webhook_url: str | None = Field(default=None)
    webhook_url: str | None = Field(default=None)
    email_recipients: list[str] | None = Field(default=None)
    enabled: bool | None = Field(default=None)


class ClaimNotificationResponse(BaseModel):
    """Notification payload for claims-related events."""

    model_config = ConfigDict(extra="forbid")

    id: int
    user_id: str
    kind: str
    target_user_id: str | None = None
    target_review_group: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: str | None = None
    delivered_at: str | None = None


class ClaimNotificationsAckRequest(BaseModel):
    """Mark notifications as delivered."""

    model_config = ConfigDict(extra="forbid")

    ids: list[int] = Field(..., min_length=1)


class ClaimNotificationsDigestResponse(BaseModel):
    """Digest response for claim notifications."""

    model_config = ConfigDict(extra="forbid")

    total: int
    counts_by_kind: dict[str, int] = Field(default_factory=dict)
    counts_by_target_user: dict[str, int] = Field(default_factory=dict)
    counts_by_review_group: dict[str, int] = Field(default_factory=dict)
    notifications: list[ClaimNotificationResponse] | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_navigation_aliases(self)


class ClaimReviewRequest(BaseModel):
    """Review update payload for a claim."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="New review status.")
    review_version: int = Field(..., ge=1)
    notes: str | None = Field(default=None)
    corrected_text: str | None = Field(default=None)
    reason_code: str | None = Field(default=None)
    reviewer_id: int | None = Field(default=None)
    review_group: str | None = Field(default=None)


class ClaimReviewBulkRequest(BaseModel):
    """Bulk review action payload."""

    model_config = ConfigDict(extra="forbid")

    claim_ids: list[int] = Field(..., min_length=1)
    status: str = Field(..., description="Review status for all claims.")
    notes: str | None = Field(default=None)
    reason_code: str | None = Field(default=None)
    reviewer_id: int | None = Field(default=None)
    review_group: str | None = Field(default=None)


class ClaimReviewRuleCreate(BaseModel):
    """Create review assignment rule."""

    model_config = ConfigDict(extra="forbid")

    priority: int = Field(default=0)
    predicate_json: dict[str, object] = Field(default_factory=dict)
    reviewer_id: int | None = Field(default=None)
    review_group: str | None = Field(default=None)
    active: bool | None = Field(default=None)


class ClaimReviewRuleUpdate(BaseModel):
    """Update review assignment rule."""

    model_config = ConfigDict(extra="forbid")

    priority: int | None = Field(default=None)
    predicate_json: dict[str, object] | None = Field(default=None)
    reviewer_id: int | None = Field(default=None)
    review_group: str | None = Field(default=None)
    active: bool | None = Field(default=None)


class ClaimsAnalyticsExportFilters(BaseModel):
    """Filters for claims analytics exports."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str | None = Field(default=None)
    event_type: str | None = Field(default=None)
    severity: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    model: str | None = Field(default=None)
    start_time: str | None = Field(default=None)
    end_time: str | None = Field(default=None)


class ClaimsAnalyticsExportPagination(BaseModel):
    """Pagination controls for claims analytics exports."""

    model_config = ConfigDict(extra="forbid")

    limit: int | None = Field(default=1000, ge=1, le=10000)
    offset: int | None = Field(default=0, ge=0)


class ClaimsAnalyticsExportRequest(BaseModel):
    """Export analytics payload."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["json", "csv"]
    filters: ClaimsAnalyticsExportFilters | None = Field(default=None)
    pagination: ClaimsAnalyticsExportPagination | None = Field(default=None)


class ClaimsAnalyticsExportResponse(BaseModel):
    """Claims analytics export handle."""

    model_config = ConfigDict(extra="forbid")

    export_id: str
    format: Literal["json", "csv"]
    status: str
    download_url: str | None = None
    created_at: str | None = None


class ClaimsAnalyticsExportPaginationMeta(BaseModel):
    """Pagination metadata for stored exports."""

    model_config = ConfigDict(extra="forbid")

    limit: int | None = None
    offset: int | None = None
    total: int | None = None


class ClaimsAnalyticsExportListItem(BaseModel):
    """Claims analytics export history item."""

    model_config = ConfigDict(extra="forbid")

    export_id: str
    format: Literal["json", "csv"]
    status: str
    download_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    filters: ClaimsAnalyticsExportFilters | None = None
    pagination: ClaimsAnalyticsExportPaginationMeta | None = None
    error_message: str | None = None


class ClaimsAnalyticsExportListResponse(BaseModel):
    """Claims analytics export history payload."""

    model_config = ConfigDict(extra="forbid")

    exports: list[ClaimsAnalyticsExportListItem] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ClaimsAnalyticsPerMediaCount(BaseModel):
    """Claims per media count."""

    model_config = ConfigDict(extra="forbid")

    media_id: int
    count: int


class ClaimsAnalyticsPerMediaStats(BaseModel):
    """Summary stats for claims per media."""

    model_config = ConfigDict(extra="forbid")

    mean: float | None = None
    p95: int | None = None
    max: int | None = None


class ClaimsAnalyticsReviewThroughputPoint(BaseModel):
    """Daily review throughput point."""

    model_config = ConfigDict(extra="forbid")

    date: str
    count: int


class ClaimsAnalyticsReviewThroughput(BaseModel):
    """Windowed review throughput summary."""

    model_config = ConfigDict(extra="forbid")

    window_days: int
    total: int
    daily: list[ClaimsAnalyticsReviewThroughputPoint] = Field(default_factory=list)


class ClaimsAnalyticsReviewStatusTrendPoint(BaseModel):
    """Daily review status trend point."""

    model_config = ConfigDict(extra="forbid")

    date: str
    total: int
    status_counts: dict[str, int] = Field(default_factory=dict)


class ClaimsAnalyticsReviewStatusTrends(BaseModel):
    """Windowed review status trend summary."""

    model_config = ConfigDict(extra="forbid")

    window_days: int
    daily: list[ClaimsAnalyticsReviewStatusTrendPoint] = Field(default_factory=list)


class ClaimsReviewExtractorMetricsDaily(BaseModel):
    """Daily review metrics grouped by extractor."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    user_id: str
    report_date: str
    extractor: str
    extractor_version: str
    total_reviewed: int
    approved_count: int
    rejected_count: int
    flagged_count: int
    reassigned_count: int
    edited_count: int
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class ClaimsReviewExtractorMetricsResponse(BaseModel):
    """Review extractor metrics response payload."""

    model_config = ConfigDict(extra="forbid")

    items: list[ClaimsReviewExtractorMetricsDaily] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ClaimsAnalyticsClusterSummary(BaseModel):
    """Summary view for a claim cluster."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    member_count: int
    watchlist_count: int
    canonical_claim_text: str | None = None
    updated_at: str | None = None


class ClaimsAnalyticsClusterHotspot(BaseModel):
    """Cluster hotspot summary."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    member_count: int
    issue_count: int
    issue_ratio: float | None = None
    watchlist_count: int
    canonical_claim_text: str | None = None
    updated_at: str | None = None


class ClaimsAnalyticsClusterStats(BaseModel):
    """Aggregate cluster statistics."""

    model_config = ConfigDict(extra="forbid")

    total_clusters: int
    clusters_with_members: int
    total_members: int
    avg_member_count: float | None = None
    p95_member_count: int | None = None
    max_member_count: int | None = None
    orphan_claims: int
    top_clusters: list[ClaimsAnalyticsClusterSummary] = Field(default_factory=list)
    hotspots: list[ClaimsAnalyticsClusterHotspot] = Field(default_factory=list)


class ClaimsAnalyticsUnsupportedRatios(BaseModel):
    """Unsupported ratio overview."""

    model_config = ConfigDict(extra="forbid")

    window_sec: int
    baseline_sec: int
    window_ratio: float | None = None
    baseline_ratio: float | None = None


class ClaimsAnalyticsProviderUsage(BaseModel):
    """Provider usage summary for claims LLM operations."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    operation: str
    requests: int
    errors: int
    total_tokens: int
    total_cost_usd: float
    latency_avg_ms: float | None = None
    latency_p95_ms: float | None = None


class ClaimsAnalyticsRebuildHealth(BaseModel):
    """Rebuild worker health summary."""

    model_config = ConfigDict(extra="forbid")

    status: str
    queue_length: int
    workers: int
    last_heartbeat_ts: float
    heartbeat_age_sec: float | None = None
    last_processed_ts: float | None = None
    last_failure: dict[str, object] | None = None
    stale: bool


class ClaimsAnalyticsDashboardResponse(BaseModel):
    """Dashboard-ready claims analytics payload."""

    model_config = ConfigDict(extra="forbid")

    total_claims: int
    status_counts: dict[str, int]
    avg_review_latency_sec: float | None = None
    p95_review_latency_sec: float | None = None
    review_backlog: int
    claims_per_media_top: list[ClaimsAnalyticsPerMediaCount] = Field(default_factory=list)
    claims_per_media_stats: ClaimsAnalyticsPerMediaStats
    review_throughput: ClaimsAnalyticsReviewThroughput
    review_status_trends: ClaimsAnalyticsReviewStatusTrends
    review_extractor_metrics: list[ClaimsReviewExtractorMetricsDaily] | None = None
    clusters: ClaimsAnalyticsClusterStats
    unsupported_ratios: ClaimsAnalyticsUnsupportedRatios
    provider_usage: list[ClaimsAnalyticsProviderUsage] = Field(default_factory=list)
    rebuild_health: ClaimsAnalyticsRebuildHealth | None = None


# =============================================================================
# FVA (Falsification-Verification Alignment) Schemas
# =============================================================================


class FVAConfigRequest(BaseModel):
    """Configuration options for FVA pipeline."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable FVA falsification checking.")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence threshold for triggering falsification."
    )
    contested_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Threshold ratio for CONTESTED status."
    )
    max_concurrent_falsifications: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent falsification operations."
    )
    timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=120.0, description="Timeout for falsification operations."
    )
    force_claim_types: list[str] | None = Field(
        default=None, description="Claim types that always trigger falsification."
    )
    max_budget_usd: float | None = Field(
        default=None, ge=0.0, description="Optional maximum budget in USD for this request."
    )


class FVAClaimInput(BaseModel):
    """A claim to verify through the FVA pipeline."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="The claim text to verify.")
    claim_type: str | None = Field(default=None, description="Optional claim type classification.")
    span_start: int | None = Field(default=None, ge=0, description="Start position in source text.")
    span_end: int | None = Field(default=None, ge=0, description="End position in source text.")


class FVAVerifyRequest(BaseModel):
    """Request for FVA-enhanced claim verification."""

    model_config = ConfigDict(extra="forbid")

    claims: list[FVAClaimInput] = Field(..., min_length=1, max_length=50, description="Claims to verify.")
    query: str = Field(..., min_length=1, description="Original query context for retrieval.")
    sources: list[str] | None = Field(
        default=None, description="Data sources to search (media_db, notes_db, etc.)."
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of documents to retrieve.")
    fva_config: FVAConfigRequest | None = Field(default=None, description="FVA pipeline configuration.")


class FVAEvidenceItem(BaseModel):
    """Evidence item for FVA verification results."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document ID.")
    snippet: str = Field(..., description="Evidence snippet text.")
    score: float = Field(..., description="Evidence relevance score.")
    stance: str | None = Field(default=None, description="Evidence stance: supports, contradicts, neutral.")
    confidence: float | None = Field(default=None, description="Stance assessment confidence.")


class FVAAdjudicationResult(BaseModel):
    """Adjudication details from FVA pipeline."""

    model_config = ConfigDict(extra="forbid")

    support_score: float = Field(..., description="Aggregate supporting evidence score.")
    contradict_score: float = Field(..., description="Aggregate contradicting evidence score.")
    contestation_score: float = Field(..., description="Evidence balance (0=one-sided, 1=balanced).")
    rationale: str = Field(..., description="Adjudication rationale explaining the verdict.")


class FVAClaimResult(BaseModel):
    """Verification result for a single claim from FVA pipeline."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(..., description="Original claim text.")
    claim_type: str | None = Field(default=None, description="Claim type classification.")
    original_status: str = Field(..., description="Status from initial verification.")
    final_status: str = Field(..., description="Status after FVA adjudication.")
    confidence: float = Field(..., description="Final verification confidence.")
    falsification_triggered: bool = Field(..., description="Whether falsification was triggered.")
    anti_context_found: int = Field(..., description="Number of anti-context documents found.")
    supporting_evidence: list[FVAEvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[FVAEvidenceItem] = Field(default_factory=list)
    adjudication: FVAAdjudicationResult | None = Field(default=None)
    rationale: str | None = Field(default=None, description="Verification rationale.")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds.")


class FVAVerifyResponse(BaseModel):
    """Response from FVA claim verification endpoint."""

    model_config = ConfigDict(extra="forbid")

    results: list[FVAClaimResult] = Field(default_factory=list)
    total_claims: int = Field(..., description="Total number of claims processed.")
    falsification_triggered_count: int = Field(..., description="Claims that triggered falsification.")
    status_changes: dict[str, int] = Field(
        default_factory=dict, description="Count of status transitions (e.g., 'verified->contested': 2)."
    )
    total_time_ms: float = Field(..., description="Total processing time in milliseconds.")
    budget_exhausted: bool = Field(default=False, description="Whether budget was exhausted.")


class FVASettingsResponse(BaseModel):
    """Current FVA pipeline settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(..., description="Whether FVA is enabled globally.")
    confidence_threshold: float = Field(..., description="Default confidence threshold.")
    contested_threshold: float = Field(..., description="Default contested threshold.")
    max_concurrent_falsifications: int = Field(..., description="Default max concurrent ops.")
    timeout_seconds: float = Field(..., description="Default timeout.")
    force_claim_types: list[str] = Field(default_factory=list)
    anti_context_cache_size: int = Field(default=0, description="Current anti-context cache size.")
