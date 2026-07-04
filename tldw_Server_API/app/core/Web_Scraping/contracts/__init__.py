from __future__ import annotations

from .conversion import (
    article_failure_to_public_dict,
    enhanced_failure_to_public_dict,
    extraction_result_to_public_dict,
    preflight_result_to_public_dict,
    search_result_to_public_dict,
    search_results_to_public_dict,
)
from .errors import RuntimeFailure
from .requests import CrawlRequest, ScrapeContext, ScrapeRequest, SearchRequest
from .results import (
    CrawlResult,
    ExtractionResult,
    PreflightAdvice,
    PreflightResult,
    SearchResult,
    SearchResultsPayload,
)
from .statuses import WebScrapingStatus

__all__ = [
    "CrawlRequest",
    "CrawlResult",
    "ExtractionResult",
    "PreflightAdvice",
    "PreflightResult",
    "RuntimeFailure",
    "ScrapeContext",
    "ScrapeRequest",
    "SearchRequest",
    "SearchResult",
    "SearchResultsPayload",
    "WebScrapingStatus",
    "article_failure_to_public_dict",
    "enhanced_failure_to_public_dict",
    "extraction_result_to_public_dict",
    "preflight_result_to_public_dict",
    "search_result_to_public_dict",
    "search_results_to_public_dict",
]
