from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ScopeValue = str | None
DocumentType = Literal["markdown", "mdx", "text", "html", "other"]
RetrievalMode = Literal["metadata", "snippet", "section", "full", "chunk", "chunk_with_neighbors"]
SourceType = Literal["local_file", "local_directory", "url_page", "url_sitemap"]
SourceLinkStatus = Literal["active", "tombstoned", "failed"]
SyncMode = Literal["dry_run", "apply"]
StalePolicy = Literal["report", "tombstone"]
SyncItemStatus = Literal["created", "updated", "unchanged", "missing", "tombstoned", "failed", "skipped"]
SyncRunStatus = Literal["completed", "partial", "skipped", "denied", "failed"]
DiscoveryKind = Literal["auto", "sitemap", "page_links"]
DiscoveryMode = Literal["dry_run", "apply"]
DiscoveryApplyAction = Literal["register", "ingest", "register_and_ingest"]
DiscoveryCandidateStatus = Literal["accepted", "duplicate", "denied", "skipped", "ingested", "failed"]


@dataclass(frozen=True)
class AccessScope:
    owner_scope: ScopeValue = None
    profile_scope: ScopeValue = None


@dataclass(frozen=True)
class SearchFilters:
    collection: str | None = None
    keywords: tuple[str, ...] = ()
    source_type: str | None = None
    document_type: str | None = None
    uri_prefix: str | None = None
    package: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class SearchRequest:
    query: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    limit: int = 10
    offset: int = 0
    snippet_length: int = 300


@dataclass(frozen=True)
class ContextRequest:
    query: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    max_chunks: int = 8
    max_documents: int = 4
    max_characters: int = 12_000
    citation_style: str = "inline"


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    title: str
    document_type: str
    canonical_uri: str
    source_path: str | None
    source_url: str | None
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRecord:
    id: int
    source_type: SourceType
    canonical_uri: str
    display_name: str
    source_path: str | None
    source_url: str | None
    redacted_source_url: str | None
    sync_enabled: bool
    last_sync_status: str | None
    last_sync_started_at: str | None
    last_sync_completed_at: str | None
    last_error_code: str | None
    document_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncSourceRequest:
    source_id: int | None = None
    source_uri: str | None = None
    mode: SyncMode = "dry_run"
    max_documents: int | None = None
    max_pages: int | None = None
    stale_policy: StalePolicy = "report"
    force: bool = False


@dataclass(frozen=True)
class DiscoverSourceRequest:
    url: str
    kind: DiscoveryKind = "auto"
    mode: DiscoveryMode = "dry_run"
    apply_action: DiscoveryApplyAction | None = None
    max_pages: int | None = None
    max_depth: int | None = None
    collections: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    title: str | None = None
    include_seed: bool = False


@dataclass(frozen=True)
class SearchResult:
    document_id: int
    chunk_id: int
    title: str
    snippet: str
    score: float
    uri: str
    citation: str
    metadata: dict[str, Any] = field(default_factory=dict)
