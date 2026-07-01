from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ScopeValue = str | None
DocumentType = Literal["markdown", "mdx", "text", "html", "other"]
RetrievalMode = Literal["metadata", "snippet", "section", "full", "chunk", "chunk_with_neighbors"]


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
    filters: SearchFilters = SearchFilters()
    limit: int = 10
    offset: int = 0
    snippet_length: int = 300


@dataclass(frozen=True)
class ContextRequest:
    query: str
    filters: SearchFilters = SearchFilters()
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
class SearchResult:
    document_id: int
    chunk_id: int
    title: str
    snippet: str
    score: float
    uri: str
    citation: str
    metadata: dict[str, Any] = field(default_factory=dict)
