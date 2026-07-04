from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import RuntimeFailure
from .results import ExtractionResult, PreflightResult, SearchResult, SearchResultsPayload


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_plain(item) for item in value]
    return value


def preflight_result_to_public_dict(result: PreflightResult) -> dict[str, Any]:
    return {
        "analysis": _to_plain(result.analysis),
        "advice": {
            "backend": result.advice.backend,
            "method": result.advice.method,
            "notes": list(result.advice.notes),
        },
    }


def extraction_result_to_public_dict(result: ExtractionResult) -> dict[str, Any]:
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
    return payload


def article_failure_to_public_dict(url: str, failure: RuntimeFailure) -> dict[str, Any]:
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
    payload: dict[str, Any] = {
        "url": str(url),
        "error": failure.public_message,
        "extraction_successful": False,
    }
    payload.update(failure.as_policy_fields())
    return payload


def search_result_to_public_dict(result: SearchResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, SearchResult):
        return {
            "title": result.title,
            "url": result.url,
            "content": result.content,
            "metadata": _to_plain(result.metadata),
        }
    return _to_plain(result)


def search_results_to_public_dict(payload: SearchResultsPayload) -> dict[str, Any]:
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
    public.update(_to_plain(payload.extra_fields))
    return public
