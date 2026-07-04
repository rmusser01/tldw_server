from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts import (
    CrawlRequest,
    CrawlResult,
    ExtractionResult,
    PreflightAdvice,
    PreflightResult,
    RuntimeFailure,
    ScrapeContext,
    ScrapeRequest,
    SearchRequest,
    SearchResult,
    SearchResultsPayload,
    WebScrapingStatus,
    article_failure_to_public_dict,
    enhanced_failure_to_public_dict,
    extraction_result_to_public_dict,
    preflight_result_to_public_dict,
    search_result_to_public_dict,
    search_request_to_legacy_kwargs,
    search_results_to_public_dict,
)


INITIALIZED_WEBSEARCH_KEYS = {
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


@pytest.mark.unit
def test_web_scraping_status_values_match_refactor_design() -> None:
    assert [status.value for status in WebScrapingStatus] == [
        "ok",
        "blocked",
        "policy_denied",
        "timeout",
        "budget_exhausted",
        "external_tool_disabled",
        "unavailable",
        "error",
    ]


@pytest.mark.unit
def test_runtime_failure_exposes_sanitized_public_message_and_policy_fields() -> None:
    failure = RuntimeFailure(
        status=WebScrapingStatus.POLICY_DENIED,
        public_message="Blocked by outbound policy",
        reason="robots_unreachable",
        mode="strict",
        stage="pre_fetch",
        source="article_extract",
    )

    assert failure.public_message == "Blocked by outbound policy"
    assert failure.as_policy_fields() == {
        "policy_reason": "robots_unreachable",
        "policy_mode": "strict",
        "policy_stage": "pre_fetch",
        "policy_source": "article_extract",
    }


@pytest.mark.unit
def test_contract_package_import_boundary_stays_stdlib_only() -> None:
    allowed_roots = {
        "__future__",
        "collections",
        "copy",
        "dataclasses",
        "enum",
        "types",
        "typing",
    }
    allowed_relative_modules = {
        "conversion",
        "errors",
        "requests",
        "results",
        "statuses",
    }
    contracts_dir = Path("tldw_Server_API/app/core/Web_Scraping/contracts")

    for path in contracts_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                assert roots <= allowed_roots, (path, roots - allowed_roots)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                root = (node.module or "").split(".", 1)[0]
                assert root in allowed_roots, (path, root)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 1, (path, node.level, node.module)
                if node.module:
                    root = node.module.split(".", 1)[0]
                    assert root in allowed_relative_modules, (path, root)
                else:
                    aliases = {alias.name.split(".", 1)[0] for alias in node.names}
                    assert aliases <= allowed_relative_modules, (path, aliases - allowed_relative_modules)


@pytest.mark.unit
def test_scrape_request_normalizes_and_copies_mutable_inputs() -> None:
    headers = {"User-Agent": "test-agent"}
    cookies = [{"name": "session", "value": "redacted"}]
    metadata = {"trace": {"id": "abc"}}
    request = ScrapeRequest(
        url=" https://example.com/article ",
        method="auto",
        backend="trafilatura",
        headers=headers,
        custom_cookies=cookies,
        context=ScrapeContext(source="unit-test", metadata=metadata),
    )
    headers["User-Agent"] = "mutated"
    cookies[0]["value"] = "mutated"
    metadata["trace"]["id"] = "mutated"

    assert request.url == "https://example.com/article"
    assert request.headers["User-Agent"] == "test-agent"
    assert request.custom_cookies[0]["value"] == "redacted"
    assert request.context.metadata["trace"]["id"] == "abc"
    assert isinstance(request.context.metadata, MappingProxyType)


@pytest.mark.unit
def test_requests_reject_empty_required_urls_and_queries() -> None:
    with pytest.raises(ValueError, match="url is required"):
        ScrapeRequest(url=" ")
    with pytest.raises(ValueError, match="base_url is required"):
        CrawlRequest(base_url="")
    with pytest.raises(ValueError, match="search_query is required"):
        SearchRequest(search_query=" ")


@pytest.mark.unit
def test_search_request_preserves_current_perform_websearch_surface() -> None:
    search_params = {"nested": {"value": "kept"}}
    request = SearchRequest(
        search_engine="google",
        search_query=" query ",
        content_country="FR",
        search_lang="fr",
        output_lang="en",
        result_count=3,
        date_range="w",
        safesearch="off",
        site_blacklist=["blocked.example"],
        exactTerms="exact",
        excludeTerms="exclude",
        filter="1",
        geolocation="Paris",
        search_result_language="fr",
        sort_results_by="date",
        search_params=search_params,
        site_whitelist="example.com, docs.example",
        google_domain="google.fr",
    )
    search_params["nested"]["value"] = "mutated"

    assert request.search_query == "query"
    assert request.search_engine == "google"
    assert request.content_country == "FR"
    assert request.search_lang == "fr"
    assert request.output_lang == "en"
    assert request.result_count == 3
    assert request.date_range == "w"
    assert request.safesearch == "off"
    assert request.site_blacklist == ("blocked.example",)
    assert request.site_whitelist == ("example.com", "docs.example")
    assert request.exactTerms == "exact"
    assert request.excludeTerms == "exclude"
    assert request.filter == "1"
    assert request.geolocation == "Paris"
    assert request.search_result_language == "fr"
    assert request.sort_results_by == "date"
    assert request.search_params["nested"]["value"] == "kept"
    assert request.google_domain == "google.fr"


@pytest.mark.unit
def test_result_contracts_copy_preflight_and_metadata() -> None:
    analysis = {
        "results": {"js": {"status": "success", "js_required": True}},
        "score": {"level": "medium"},
        "recommendations": {"actions": ["use_browser"]},
    }
    metadata = {"provider": {"name": "brave"}}
    preflight = PreflightResult(
        analysis=analysis,
        advice=PreflightAdvice(backend="curl", method="playwright", notes=["js_required", "tls_active"]),
    )
    search_result = SearchResult(
        title="Result",
        url="https://example.com/result",
        content="Summary",
        metadata=metadata,
    )
    analysis["results"]["js"]["status"] = "mutated"
    metadata["provider"]["name"] = "mutated"

    assert preflight.analysis["results"]["js"]["status"] == "success"
    assert preflight.advice.notes == ("js_required", "tls_active")
    assert search_result.metadata["provider"]["name"] == "brave"


@pytest.mark.unit
def test_result_contract_defaults_are_compatible_with_public_shapes() -> None:
    extraction = ExtractionResult(url="https://example.com", content="Article")
    crawl = CrawlResult(base_url="https://example.com", results=(extraction,))
    search = SearchResultsPayload(search_engine="duckduckgo", search_query="query")

    assert extraction.extraction_successful is True
    assert extraction.title == "N/A"
    assert crawl.results == (extraction,)
    assert search.results == ()
    assert search.safesearch == "active"
    assert search.site_whitelist == ()
    assert search.site_blacklist == ()
    assert search.warnings == ()
    assert search.processing_error is None


@pytest.mark.unit
def test_preflight_result_converts_to_public_attachment_only() -> None:
    failure = RuntimeFailure(status=WebScrapingStatus.ERROR, public_message="internal")
    result = PreflightResult(
        status=WebScrapingStatus.ERROR,
        analysis={
            "results": {"js": {"status": "success", "js_required": True}},
            "score": {"level": "medium"},
            "recommendations": {"actions": ["use_browser"]},
        },
        advice=PreflightAdvice(backend="curl", method="playwright", notes=("js_required", "tls_active")),
        failure=failure,
    )

    assert preflight_result_to_public_dict(result) == {
        "analysis": {
            "results": {"js": {"status": "success", "js_required": True}},
            "score": {"level": "medium"},
            "recommendations": {"actions": ["use_browser"]},
        },
        "advice": {
            "backend": "curl",
            "method": "playwright",
            "notes": ["js_required", "tls_active"],
        },
    }


@pytest.mark.unit
def test_extraction_result_converts_to_existing_public_article_shape() -> None:
    result = ExtractionResult(
        url="https://example.com/article",
        title="Example",
        author="Author",
        date="2026-07-04",
        content="Body",
        backend="trafilatura",
        method="auto",
        preflight_analysis={"analysis": {"results": {"js": {"status": "success"}}}, "advice": {"notes": []}},
    )

    assert extraction_result_to_public_dict(result) == {
        "url": "https://example.com/article",
        "title": "Example",
        "author": "Author",
        "date": "2026-07-04",
        "content": "Body",
        "extraction_successful": True,
        "backend": "trafilatura",
        "method": "auto",
        "preflight_analysis": {
            "analysis": {"results": {"js": {"status": "success"}}},
            "advice": {"notes": []},
        },
    }


@pytest.mark.unit
def test_extraction_result_preserves_legacy_diagnostic_fields() -> None:
    result = ExtractionResult(
        url="https://example.com/article",
        content="Body",
        extra_fields={
            "extraction_trace": [{"strategy": "trafilatura", "status": "success"}],
            "extraction_strategy": "trafilatura",
            "extraction_strategy_order": ["trafilatura", "readability"],
        },
    )

    public = extraction_result_to_public_dict(result)

    assert public["extraction_trace"] == [{"strategy": "trafilatura", "status": "success"}]
    assert public["extraction_strategy"] == "trafilatura"
    assert public["extraction_strategy_order"] == ["trafilatura", "readability"]


@pytest.mark.unit
def test_extraction_result_extra_fields_cannot_replace_canonical_article_fields() -> None:
    result = ExtractionResult(
        url="https://example.com/article",
        title="Canonical title",
        content="Body",
        extra_fields={
            "title": "provider override",
            "extraction_trace": [{"strategy": "trafilatura", "status": "success"}],
        },
    )

    public = extraction_result_to_public_dict(result)

    assert public["title"] == "Canonical title"
    assert public["extraction_trace"] == [{"strategy": "trafilatura", "status": "success"}]


@pytest.mark.unit
def test_failure_converters_keep_article_and_enhanced_shapes_separate() -> None:
    robots_failure = RuntimeFailure(
        status=WebScrapingStatus.POLICY_DENIED,
        public_message="Blocked by outbound policy",
        reason="robots_unreachable",
        mode="strict",
        stage="pre_fetch",
        source="article_extract",
    )
    deny_failure = RuntimeFailure(
        status=WebScrapingStatus.POLICY_DENIED,
        public_message="Egress denied: deny_test",
        reason="deny_test",
        mode="compat",
        stage="pre_fetch",
        source="enhanced_scrape",
        backend="curl",
        provider="internal",
    )
    enhanced_robots_failure = RuntimeFailure(
        status=WebScrapingStatus.POLICY_DENIED,
        public_message="Blocked by outbound policy",
        reason="robots_unreachable",
        mode="strict",
        stage="pre_fetch",
        source="enhanced_scrape",
        backend="curl",
        provider="internal",
    )

    assert article_failure_to_public_dict("https://example.com/blocked", robots_failure) == {
        "url": "https://example.com/blocked",
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "Blocked by outbound policy",
        "policy_reason": "robots_unreachable",
        "policy_mode": "strict",
        "policy_stage": "pre_fetch",
        "policy_source": "article_extract",
    }
    assert enhanced_failure_to_public_dict("https://example.com/blocked", deny_failure) == {
        "url": "https://example.com/blocked",
        "error": "Egress denied: deny_test",
        "extraction_successful": False,
        "policy_reason": "deny_test",
        "policy_mode": "compat",
        "policy_stage": "pre_fetch",
        "policy_source": "enhanced_scrape",
    }
    assert enhanced_failure_to_public_dict("https://example.com/blocked", enhanced_robots_failure) == {
        "url": "https://example.com/blocked",
        "error": "Blocked by outbound policy",
        "extraction_successful": False,
        "policy_reason": "robots_unreachable",
        "policy_mode": "strict",
        "policy_stage": "pre_fetch",
        "policy_source": "enhanced_scrape",
    }


@pytest.mark.unit
def test_search_result_converts_to_processed_provider_item_shape() -> None:
    result = SearchResult(
        title="Result title",
        url="https://example.com/result",
        content="Result summary",
        metadata={
            "date_published": "2026-07-04",
            "source": "Example",
            "relevance_score": 0.9,
            "snippet": "Result summary",
            "provider_extra": "kept",
        },
    )

    assert search_result_to_public_dict(result) == {
        "title": "Result title",
        "url": "https://example.com/result",
        "content": "Result summary",
        "metadata": {
            "date_published": "2026-07-04",
            "source": "Example",
            "relevance_score": 0.9,
            "snippet": "Result summary",
            "provider_extra": "kept",
        },
    }


@pytest.mark.unit
def test_search_results_payload_converts_to_current_initialized_shape() -> None:
    payload = SearchResultsPayload(
        search_engine="duckduckgo",
        search_query="capital of france",
        content_country="FR",
        search_lang="fr",
        output_lang="en",
        result_count=1,
        date_range="w",
        safesearch="off",
        site_whitelist=("example.com",),
        site_blacklist=("blocked.example",),
        exactTerms="capital",
        excludeTerms="exclude",
        filter="1",
        geolocation="Paris",
        search_result_language="fr",
        sort_results_by="date",
        google_domain="google.fr",
        results=(
            SearchResult(
                title="Paris",
                url="https://example.com/paris",
                content="Paris summary",
                metadata={"snippet": "Paris summary"},
            ),
        ),
        total_results_found=1,
        search_time=0.25,
        warnings=("partial",),
    )

    public = search_results_to_public_dict(payload)

    assert set(public) == INITIALIZED_WEBSEARCH_KEYS
    assert public["site_whitelist"] == ["example.com"]
    assert public["site_blacklist"] == ["blocked.example"]
    assert public["results"] == [
        {
            "title": "Paris",
            "url": "https://example.com/paris",
            "content": "Paris summary",
            "metadata": {"snippet": "Paris summary"},
        }
    ]
    assert public["warnings"] == ["partial"]


@pytest.mark.unit
def test_search_results_payload_preserves_top_level_provider_extras_when_supplied() -> None:
    payload = SearchResultsPayload(
        search_engine="brave",
        search_query="query",
        extra_fields={"city": "Paris", "state": "Ile-de-France", "more_results_available": True},
    )

    public = search_results_to_public_dict(payload)

    assert public["city"] == "Paris"
    assert public["state"] == "Ile-de-France"
    assert public["more_results_available"] is True


@pytest.mark.unit
def test_search_results_payload_extra_fields_cannot_replace_canonical_keys() -> None:
    payload = SearchResultsPayload(
        search_engine="brave",
        search_query="query",
        results=(SearchResult(title="Canonical", url="https://example.com"),),
        warnings=("canonical warning",),
        extra_fields={
            "results": [{"title": "provider override"}],
            "warnings": ["provider warning"],
            "error": "provider error",
            "city": "Paris",
        },
    )

    public = search_results_to_public_dict(payload)

    assert public["results"] == [
        {
            "title": "Canonical",
            "url": "https://example.com",
            "content": "",
            "metadata": {},
        }
    ]
    assert public["warnings"] == ["canonical warning"]
    assert public["error"] is None
    assert public["city"] == "Paris"


@pytest.mark.unit
def test_search_results_payload_normalizes_string_domain_filters_without_character_split() -> None:
    payload = SearchResultsPayload(
        search_engine="google",
        search_query="query",
        site_whitelist="example.com, docs.example",
        site_blacklist="blocked.example",
    )

    public = search_results_to_public_dict(payload)

    assert payload.site_whitelist == ("example.com", "docs.example")
    assert payload.site_blacklist == ("blocked.example",)
    assert public["site_whitelist"] == ["example.com", "docs.example"]
    assert public["site_blacklist"] == ["blocked.example"]


@pytest.mark.unit
def test_search_request_converts_domain_filters_to_legacy_list_kwargs() -> None:
    request = SearchRequest(
        search_engine="google",
        search_query="query",
        site_whitelist="example.com, docs.example",
        site_blacklist=["blocked.example"],
    )

    legacy_kwargs = search_request_to_legacy_kwargs(request)

    assert legacy_kwargs["site_whitelist"] == ["example.com", "docs.example"]
    assert legacy_kwargs["site_blacklist"] == ["blocked.example"]
    assert isinstance(legacy_kwargs["site_whitelist"], list)
    assert isinstance(legacy_kwargs["site_blacklist"], list)
