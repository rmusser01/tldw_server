from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in dict(value or {}).items()})


def _normalize_domains(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        candidates = value.split(",")
    else:
        candidates = value
    return tuple(str(item).strip() for item in candidates if str(item).strip())


@dataclass(frozen=True, slots=True)
class ScrapeContext:
    source: str = "web_scraping"
    user_id: str | int | None = None
    request_id: str | None = None
    policy_mode: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ScrapeRequest:
    url: str
    method: str = "auto"
    backend: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    custom_cookies: tuple[Mapping[str, Any], ...] = ()
    include_preflight: bool | None = None
    context: ScrapeContext = field(default_factory=ScrapeContext)

    def __post_init__(self) -> None:
        normalized_url = str(self.url or "").strip()
        if not normalized_url:
            raise ValueError("url is required")
        object.__setattr__(self, "url", normalized_url)
        object.__setattr__(self, "method", str(self.method or "auto"))
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
        cookies = tuple(_freeze_mapping(cookie) for cookie in (self.custom_cookies or ()))
        object.__setattr__(self, "custom_cookies", cookies)


@dataclass(frozen=True, slots=True)
class CrawlRequest:
    base_url: str
    max_pages: int | None = None
    max_depth: int | None = None
    include_external: bool = False
    context: ScrapeContext = field(default_factory=ScrapeContext)

    def __post_init__(self) -> None:
        normalized_url = str(self.base_url or "").strip()
        if not normalized_url:
            raise ValueError("base_url is required")
        object.__setattr__(self, "base_url", normalized_url)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    search_query: str
    search_engine: str = "google"
    content_country: str = "US"
    search_lang: str = "en"
    output_lang: str = "en"
    result_count: int = 10
    date_range: str | None = None
    safesearch: str | None = None
    site_blacklist: str | Sequence[str] | None = None
    exactTerms: str | None = None
    excludeTerms: str | None = None
    filter: str | None = None
    geolocation: str | None = None
    search_result_language: str | None = None
    sort_results_by: str | None = None
    search_params: Mapping[str, Any] = field(default_factory=dict)
    site_whitelist: str | Sequence[str] | None = None
    google_domain: str | None = None
    context: ScrapeContext = field(default_factory=ScrapeContext)

    def __post_init__(self) -> None:
        normalized_query = str(self.search_query or "").strip()
        if not normalized_query:
            raise ValueError("search_query is required")
        object.__setattr__(self, "search_query", normalized_query)
        object.__setattr__(self, "search_engine", str(self.search_engine or "google").strip() or "google")
        object.__setattr__(self, "site_blacklist", _normalize_domains(self.site_blacklist))
        object.__setattr__(self, "site_whitelist", _normalize_domains(self.site_whitelist))
        object.__setattr__(self, "search_params", _freeze_mapping(self.search_params))
