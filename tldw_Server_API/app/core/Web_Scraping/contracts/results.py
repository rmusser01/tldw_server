"""Result contract objects for web scraping compatibility adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .errors import RuntimeFailure
from .statuses import WebScrapingStatus


def _freeze_value(value: Any) -> Any:
    """Recursively convert mutable containers into immutable equivalents."""
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable string-keyed mapping for result metadata."""
    return MappingProxyType({str(key): _freeze_value(item) for key, item in dict(value or {}).items()})


def _normalize_domains(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize comma-separated or sequence domain filters into a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        candidates = value.split(",")
    else:
        candidates = value
    return tuple(str(item).strip() for item in candidates if str(item).strip())


@dataclass(frozen=True, slots=True)
class PreflightAdvice:
    """Backend and method guidance derived from pre-scrape analysis."""

    backend: str | None = None
    method: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize advice notes to immutable strings."""
        object.__setattr__(self, "notes", tuple(str(note) for note in (self.notes or ())))


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Pre-scrape analyzer output plus normalized routing advice."""

    analysis: Mapping[str, Any] = field(default_factory=dict)
    advice: PreflightAdvice = field(default_factory=PreflightAdvice)
    status: WebScrapingStatus = WebScrapingStatus.OK
    failure: RuntimeFailure | None = None

    def __post_init__(self) -> None:
        """Freeze analysis metadata after dataclass initialization."""
        object.__setattr__(self, "analysis", _freeze_mapping(self.analysis))


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Normalized article extraction result before public conversion."""

    url: str
    title: str = "N/A"
    author: str = "N/A"
    date: str = "N/A"
    content: str = ""
    extraction_successful: bool = True
    error: str | None = None
    backend: str | None = None
    method: str | None = None
    preflight_analysis: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra_fields: Mapping[str, Any] = field(default_factory=dict)
    failure: RuntimeFailure | None = None

    def __post_init__(self) -> None:
        """Validate required URL fields and freeze metadata maps."""
        normalized_url = str(self.url or "").strip()
        if not normalized_url:
            raise ValueError("url is required")
        object.__setattr__(self, "url", normalized_url)
        if self.preflight_analysis is not None:
            object.__setattr__(self, "preflight_analysis", _freeze_mapping(self.preflight_analysis))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "extra_fields", _freeze_mapping(self.extra_fields))


@dataclass(frozen=True, slots=True)
class CrawlResult:
    """Normalized crawl result containing extracted page results."""

    base_url: str
    results: tuple[ExtractionResult, ...] = ()
    status: WebScrapingStatus = WebScrapingStatus.OK
    failure: RuntimeFailure | None = None

    def __post_init__(self) -> None:
        """Validate the crawl base URL and freeze result ordering."""
        normalized_url = str(self.base_url or "").strip()
        if not normalized_url:
            raise ValueError("base_url is required")
        object.__setattr__(self, "base_url", normalized_url)
        object.__setattr__(self, "results", tuple(self.results or ()))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Normalized individual web search result."""

    title: str
    url: str
    content: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize scalar fields and freeze result metadata."""
        object.__setattr__(self, "title", str(self.title or ""))
        object.__setattr__(self, "url", str(self.url or ""))
        object.__setattr__(self, "content", str(self.content or ""))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SearchResultsPayload:
    """Normalized web search response preserving legacy public fields."""

    search_engine: str
    search_query: str
    content_country: str = "US"
    search_lang: str = "en"
    output_lang: str = "en"
    result_count: int = 0
    date_range: str | None = None
    safesearch: str = "active"
    site_whitelist: str | Sequence[str] | None = None
    site_blacklist: str | Sequence[str] | None = None
    exactTerms: str | None = None
    excludeTerms: str | None = None
    filter: str | None = None
    geolocation: str | None = None
    search_result_language: str | None = None
    sort_results_by: str | None = None
    google_domain: str | None = None
    results: tuple[SearchResult | Mapping[str, Any], ...] = ()
    total_results_found: int = 0
    search_time: float = 0.0
    error: str | None = None
    warnings: tuple[str, ...] = ()
    processing_error: str | None = None
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize domain filters, search results, and extra metadata."""
        object.__setattr__(self, "site_whitelist", _normalize_domains(self.site_whitelist))
        object.__setattr__(self, "site_blacklist", _normalize_domains(self.site_blacklist))
        normalized_results: list[SearchResult | Mapping[str, Any]] = []
        for result in self.results or ():
            if isinstance(result, SearchResult):
                normalized_results.append(result)
            else:
                normalized_results.append(_freeze_mapping(result))
        object.__setattr__(self, "results", tuple(normalized_results))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in (self.warnings or ())))
        object.__setattr__(self, "extra_fields", _freeze_mapping(self.extra_fields))
