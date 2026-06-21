"""Pydantic schemas for the research discovery API."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_DISCOVERY_FILTER_KEYS = 50
MAX_DISCOVERY_FILTER_DEPTH = 6
MAX_DISCOVERY_FILTER_BYTES = 8192


class ResearchSourceCapabilitiesResponse(BaseModel):
    """Capabilities advertised by one research discovery source."""

    model_config = ConfigDict(from_attributes=True)

    searchable: bool
    full_text_resolvable: bool
    ingestable: bool
    requires_credentials: bool
    fallback_search_allowed: bool
    rate_limited: bool


class ResearchSourceResponse(BaseModel):
    """One source from the research discovery catalog."""

    model_config = ConfigDict(from_attributes=True)

    source_id: str
    display_name: str
    category: str
    subcategory: str | None
    content_types: list[str]
    access_level: str
    enabled: bool
    configured: bool
    default_discovery_mode: str
    fallback_enabled: bool
    fallback_configurable: bool = False
    priority: int
    provider_adapter: str | None
    site_hosts: list[str]
    trust_notes: str
    capabilities: ResearchSourceCapabilitiesResponse
    catalog_version: str


class ResearchSourceListResponse(BaseModel):
    """Research discovery source catalog response."""

    catalog_version: str
    sources: list[ResearchSourceResponse]


class ResearchDiscoverySearchRequest(BaseModel):
    """Request body for standalone research discovery search."""

    query: str = Field(..., min_length=1, max_length=1000)
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    categories: list[str] = Field(default_factory=list, max_length=20)
    per_source_limit: int = Field(default=5, ge=1, le=20)
    total_limit: int = Field(default=25, ge=1, le=100)
    fallback_policy: str = Field(default="disabled")
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("filters")
    @classmethod
    def _validate_filters(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject filters that would bloat snapshots or response config."""
        _validate_filter_shape(value)
        return value


class ResearchDiscoveryOACandidateResponse(BaseModel):
    """Open-access candidate returned for a discovery result."""

    model_config = ConfigDict(from_attributes=True)

    candidate_id: str
    candidate_type: str
    safe_url: str | None
    resolver_reference: str | None
    url_redacted: bool
    requires_reresolution: bool
    provider: str
    access_status: str | None
    license_hint: str | None
    content_type_hint: str | None
    rank: int
    confidence: float
    warnings: list[str]


class ResearchDiscoveryProvenanceResponse(BaseModel):
    """Source provenance attached to a discovery result."""

    model_config = ConfigDict(from_attributes=True)

    source_id: str
    provider: str
    discovery_mode: str
    provider_ids: dict[str, str]
    url: str | None
    source_rank: int | None
    status: str
    warnings: list[str]
    safe_metadata: dict[str, Any]
    adapter_version: str


class ResearchDiscoveryResultResponse(BaseModel):
    """Normalized research discovery result."""

    model_config = ConfigDict(from_attributes=True)

    result_id: str
    fingerprint: str
    primary_source_id: str
    primary_provider: str
    discovery_mode: str
    title: str
    authors: list[str]
    abstract: str | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    arxiv_id: str | None
    provider_ids: dict[str, str]
    canonical_url: str | None
    published_at: str | None
    updated_at: str | None
    source_category: str | None
    oa_candidates: list[ResearchDiscoveryOACandidateResponse]
    recommended_candidate_id: str | None
    ingest_eligible: bool
    dedupe_confidence: float
    ranking_signals: dict[str, Any]
    warnings: list[str]
    merged_provenance: list[ResearchDiscoveryProvenanceResponse]
    safe_metadata: dict[str, Any]
    adapter_version: str
    catalog_version: str


class ResearchDiscoverySourceStatusResponse(BaseModel):
    """Per-source execution status for a discovery search."""

    model_config = ConfigDict(from_attributes=True)

    source_id: str
    provider: str | None
    status: str
    message: str | None
    result_count: int
    elapsed_ms: float | None
    warnings: list[str]


class ResearchDiscoveryMetricsResponse(BaseModel):
    """Aggregate metrics for a discovery search."""

    model_config = ConfigDict(from_attributes=True)

    selected_source_count: int
    result_count: int
    deduped_result_count: int
    oa_candidate_count: int
    elapsed_ms: float | None


class ResearchDiscoverySearchResponse(BaseModel):
    """Standalone research discovery search response."""

    model_config = ConfigDict(from_attributes=True)

    discovery_id: str
    query: str
    results: list[ResearchDiscoveryResultResponse]
    source_statuses: list[ResearchDiscoverySourceStatusResponse]
    warnings: list[str]
    effective_config: dict[str, Any]
    catalog_version: str
    metrics: ResearchDiscoveryMetricsResponse


def _validate_filter_shape(value: dict[str, Any]) -> None:
    """Validate filter nesting, key count, and serialized size."""
    key_count = _count_filter_keys(value, depth=1)
    if key_count > MAX_DISCOVERY_FILTER_KEYS:
        raise ValueError("research_discovery_filters_too_many_keys")

    serialized = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > MAX_DISCOVERY_FILTER_BYTES:
        raise ValueError("research_discovery_filters_too_large")


def _count_filter_keys(value: Any, *, depth: int) -> int:
    """Count mapping keys while enforcing maximum filter depth."""
    if depth > MAX_DISCOVERY_FILTER_DEPTH:
        raise ValueError("research_discovery_filters_too_deep")
    if isinstance(value, dict):
        return len(value) + sum(_count_filter_keys(item, depth=depth + 1) for item in value.values())
    if isinstance(value, list):
        return sum(_count_filter_keys(item, depth=depth + 1) for item in value)
    return 0
