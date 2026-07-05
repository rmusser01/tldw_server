"""Conversion helpers between web scraping contracts and legacy payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import RuntimeFailure
from .requests import SearchRequest
from .results import ExtractionResult, PreflightResult, SearchResult, SearchResultsPayload


_ARTICLE_RESERVED_KEYS = {
    "url",
    "title",
    "author",
    "date",
    "content",
    "extraction_successful",
    "error",
    "backend",
    "method",
    "preflight_analysis",
    "policy_reason",
    "policy_mode",
    "policy_stage",
    "policy_source",
}

_SEARCH_RESERVED_KEYS = {
    "search_engine",
    "search_query",
    "content_country",
    "search_lang",
    "output_lang",
    "result_count",
    "date_range",
    "safesearch",
    "site_whitelist",
    "site_blacklist",
    "exactTerms",
    "excludeTerms",
    "filter",
    "geolocation",
    "search_result_language",
    "sort_results_by",
    "google_domain",
    "results",
    "total_results_found",
    "search_time",
    "error",
    "warnings",
    "processing_error",
}


def _to_plain(value: Any) -> Any:
    """Convert immutable contract containers back to plain JSON-like values."""
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_plain(item) for item in value]
    return value


def _merge_extra_fields(
    payload: dict[str, Any],
    extra_fields: Mapping[str, Any],
    reserved_keys: set[str],
) -> None:
    """Merge non-reserved extra fields into a public payload in place."""
    for key, value in _to_plain(extra_fields).items():
        if key in reserved_keys:
            continue
        payload[key] = value


def _domain_tuple_to_legacy_list(domains: tuple[str, ...]) -> list[str] | None:
    """Convert normalized domain filters to the legacy list-or-None shape."""
    values = [str(domain).strip() for domain in domains if str(domain).strip()]
    return values or None


def preflight_result_to_public_dict(result: PreflightResult) -> dict[str, Any]:
    """Return the public dictionary shape for preflight analyzer results."""
    return {
        "analysis": _to_plain(result.analysis),
        "advice": {
            "backend": result.advice.backend,
            "method": result.advice.method,
            "notes": list(result.advice.notes),
        },
    }


def extraction_result_to_public_dict(result: ExtractionResult) -> dict[str, Any]:
    """Return the legacy public dictionary shape for article extraction."""
    payload: dict[str, Any] = {
        "url": result.url,
        "title": result.title,
        "author": result.author,
        "date": result.date,
        "content": result.content,
        "extraction_successful": result.extraction_successful,
    }
    if result.error is not None:
        payload["error"] = result.error
    if result.backend is not None:
        payload["backend"] = result.backend
    if result.method is not None:
        payload["method"] = result.method
    if result.preflight_analysis is not None:
        payload["preflight_analysis"] = _to_plain(result.preflight_analysis)
    if result.failure is not None:
        payload["error"] = result.failure.public_message
        payload.update(result.failure.as_policy_fields())
    _merge_extra_fields(payload, result.extra_fields, _ARTICLE_RESERVED_KEYS)
    return payload


def article_failure_to_public_dict(url: str, failure: RuntimeFailure) -> dict[str, Any]:
    """Return the public failure payload used by article extraction."""
    payload: dict[str, Any] = {
        "url": str(url),
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": failure.public_message,
    }
    payload.update(failure.as_policy_fields())
    return payload


def enhanced_failure_to_public_dict(url: str, failure: RuntimeFailure) -> dict[str, Any]:
    """Return the public failure payload used by enhanced scraping."""
    payload: dict[str, Any] = {
        "url": str(url),
        "error": failure.public_message,
        "extraction_successful": False,
    }
    payload.update(failure.as_policy_fields())
    return payload


def search_result_to_public_dict(result: SearchResult | Mapping[str, Any]) -> dict[str, Any]:
    """Return the public dictionary shape for one search result."""
    if isinstance(result, SearchResult):
        return {
            "title": result.title,
            "url": result.url,
            "content": result.content,
            "metadata": _to_plain(result.metadata),
        }
    return _to_plain(result)


def search_results_to_public_dict(payload: SearchResultsPayload) -> dict[str, Any]:
    """Return the legacy public dictionary shape for search responses."""
    public: dict[str, Any] = {
        "search_engine": payload.search_engine,
        "search_query": payload.search_query,
        "content_country": payload.content_country,
        "search_lang": payload.search_lang,
        "output_lang": payload.output_lang,
        "result_count": payload.result_count,
        "date_range": payload.date_range,
        "safesearch": payload.safesearch,
        "site_whitelist": list(payload.site_whitelist),
        "site_blacklist": list(payload.site_blacklist),
        "exactTerms": payload.exactTerms,
        "excludeTerms": payload.excludeTerms,
        "filter": payload.filter,
        "geolocation": payload.geolocation,
        "search_result_language": payload.search_result_language,
        "sort_results_by": payload.sort_results_by,
        "google_domain": payload.google_domain,
        "results": [search_result_to_public_dict(result) for result in payload.results],
        "total_results_found": payload.total_results_found,
        "search_time": payload.search_time,
        "error": payload.error,
        "warnings": list(payload.warnings),
        "processing_error": payload.processing_error,
    }
    _merge_extra_fields(public, payload.extra_fields, _SEARCH_RESERVED_KEYS)
    return public


def search_request_to_legacy_kwargs(request: SearchRequest) -> dict[str, Any]:
    """Return keyword arguments accepted by the legacy search implementation."""
    return {
        "search_engine": request.search_engine,
        "search_query": request.search_query,
        "content_country": request.content_country,
        "search_lang": request.search_lang,
        "output_lang": request.output_lang,
        "result_count": request.result_count,
        "date_range": request.date_range,
        "safesearch": request.safesearch,
        "site_blacklist": _domain_tuple_to_legacy_list(request.site_blacklist),
        "exactTerms": request.exactTerms,
        "excludeTerms": request.excludeTerms,
        "filter": request.filter,
        "geolocation": request.geolocation,
        "search_result_language": request.search_result_language,
        "sort_results_by": request.sort_results_by,
        "search_params": _to_plain(request.search_params),
        "site_whitelist": _domain_tuple_to_legacy_list(request.site_whitelist),
        "google_domain": request.google_domain,
    }
