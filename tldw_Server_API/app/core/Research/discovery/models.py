"""Models for research discovery source catalog selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceCapabilities:
    searchable: bool
    full_text_resolvable: bool
    ingestable: bool
    requires_credentials: bool
    fallback_search_allowed: bool
    rate_limited: bool


@dataclass(frozen=True)
class ResearchSourceCatalogEntry:
    source_id: str
    display_name: str
    category: str
    subcategory: str | None
    content_types: tuple[str, ...]
    access_level: str
    enabled: bool
    configured: bool
    default_discovery_mode: str
    fallback_enabled: bool
    priority: int
    provider_adapter: str | None
    site_hosts: tuple[str, ...]
    trust_notes: str
    capabilities: SourceCapabilities
    catalog_version: str


@dataclass(frozen=True)
class SourceSelectionError:
    code: str
    message: str
    selected_count: int
    limit: int


@dataclass(frozen=True)
class SourceStatus:
    source_id: str
    provider: str | None
    status: str
    message: str | None
    result_count: int
    elapsed_ms: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryExecutionPolicy:
    per_source_timeout_seconds: float
    total_timeout_seconds: float
    max_concurrency: int


@dataclass(frozen=True)
class DiscoveryOACandidate:
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
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryProvenance:
    source_id: str
    provider: str
    discovery_mode: str
    provider_ids: dict[str, str]
    url: str | None
    source_rank: int | None
    status: str
    warnings: tuple[str, ...]
    safe_metadata: dict[str, Any]
    adapter_version: str


@dataclass(frozen=True)
class DiscoveryResult:
    result_id: str
    fingerprint: str
    primary_source_id: str
    primary_provider: str
    discovery_mode: str
    title: str
    authors: tuple[str, ...]
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
    oa_candidates: tuple[DiscoveryOACandidate, ...]
    recommended_candidate_id: str | None
    ingest_eligible: bool
    dedupe_confidence: float
    ranking_signals: dict[str, Any]
    warnings: tuple[str, ...]
    merged_provenance: tuple[DiscoveryProvenance, ...]
    safe_metadata: dict[str, Any]
    adapter_version: str
    catalog_version: str
