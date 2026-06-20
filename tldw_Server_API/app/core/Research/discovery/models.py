"""Models for research discovery source catalog selection."""

from __future__ import annotations

from dataclasses import dataclass


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
