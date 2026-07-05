from __future__ import annotations

import importlib
import json
import types
from pathlib import Path
from typing import Any

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_JSON = REPO_ROOT / "Docs/Design/web_scraping_refactor_import_inventory.json"
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


def _article_test_config(*, preflight: bool = False) -> dict[str, Any]:
    return {
        "web_scraper": {
            "web_scraper_preflight_analyzers": preflight,
            "web_scraper_preflight_include_results": preflight,
            "web_scraper_preflight_scan_depth": "default",
            "web_scraper_preflight_timeout_s": 0,
            "web_scraper_respect_robots": True,
            "web_scraper_retry_count": 1,
            "web_scraper_retry_timeout": 1,
            "web_scraper_stealth_playwright": False,
        }
    }


def _force_article_default_plan(monkeypatch: pytest.MonkeyPatch, article: Any) -> None:
    def raise_rules(_path: str) -> list[Any]:
        raise FileNotFoundError("rules disabled for no-network test")

    monkeypatch.setattr(article.ScraperRouter, "load_rules_from_yaml", raise_rules)


def _successful_article(url: str, content: str = "Example content") -> dict[str, Any]:
    return {
        "url": url,
        "title": "Example Title",
        "author": "Example Author",
        "date": "2026-07-04",
        "content": content,
        "extraction_successful": True,
    }


def _sample_analysis(*, js_required: bool = False, tls_active: bool = False) -> dict[str, Any]:
    return {
        "results": {
            "js": {"status": "success", "js_required": js_required},
            "tls": {"status": "active" if tls_active else "inactive"},
        },
        "score": {"level": "medium"},
        "recommendations": {"actions": ["use_browser"]},
    }


def test_inventory_recorded_web_scraping_imports_remain_resolvable() -> None:
    inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    records_by_import: dict[tuple[str, str | None], dict[str, Any]] = {}
    for records in inventory["records"].values():
        for record in records:
            records_by_import.setdefault((record["module"], record.get("imported_name")), record)

    for record in records_by_import.values():
        module_name = record["module"]
        imported_name = record.get("imported_name")
        module = importlib.import_module(module_name)
        if imported_name is None:
            continue
        if hasattr(module, imported_name):
            continue
        try:
            importlib.import_module(f"{module_name}.{imported_name}")
        except (ImportError, AttributeError) as exc:
            pytest.fail(f"Inventory import no longer resolves: {record!r}; error={exc!r}")


@pytest.mark.parametrize(
    ("reason", "mode", "expected_error"),
    [
        ("robots_unreachable", "strict", "Blocked by outbound policy"),
        ("deny_test", "compat", "Egress denied: deny_test"),
    ],
)
async def test_scrape_article_policy_denial_keeps_public_blocked_shape(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    mode: str,
    expected_error: str,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import WebOutboundPolicyDecision

    _force_article_default_plan(monkeypatch, article)
    monkeypatch.setattr(article, "load_and_log_configs", lambda: _article_test_config())

    async def deny(_url: str, **kwargs: Any) -> WebOutboundPolicyDecision:
        return WebOutboundPolicyDecision(
            allowed=False,
            mode=mode,  # type: ignore[arg-type]
            reason=reason,
            stage=str(kwargs.get("stage", "pre_fetch")),
            source=str(kwargs.get("source", "article_extract")),
        )

    monkeypatch.setattr(article, "decide_web_outbound_policy", deny)

    result = await article.scrape_article("https://example.com/blocked")

    assert result == {
        "url": "https://example.com/blocked",
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": expected_error,
        "policy_reason": reason,
        "policy_mode": mode,
        "policy_stage": "pre_fetch",
        "policy_source": "article_extract",
    }


@pytest.mark.parametrize(
    ("reason", "mode", "expected_error"),
    [
        ("robots_unreachable", "strict", "Blocked by outbound policy"),
        ("deny_test", "compat", "Egress denied: deny_test"),
    ],
)
async def test_enhanced_scrape_policy_denial_keeps_public_blocked_shape(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    mode: str,
    expected_error: str,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import WebOutboundPolicyDecision

    scraper = enhanced.EnhancedWebScraper(config={})
    monkeypatch.setattr(scraper.rate_limiter, "acquire", lambda: _noop_async())
    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda _url: (_enhanced_plan(), "auto", ""))

    async def deny(_url: str, **kwargs: Any) -> WebOutboundPolicyDecision:
        return WebOutboundPolicyDecision(
            allowed=False,
            mode=mode,  # type: ignore[arg-type]
            reason=reason,
            stage=str(kwargs.get("stage", "pre_fetch")),
            source=str(kwargs.get("source", "enhanced_scrape")),
        )

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", deny)

    result = await scraper.scrape_article("https://example.com/blocked")

    assert result == {
        "url": "https://example.com/blocked",
        "error": expected_error,
        "extraction_successful": False,
        "policy_reason": reason,
        "policy_mode": mode,
        "policy_stage": "pre_fetch",
        "policy_source": "enhanced_scrape",
    }


async def _noop_async() -> None:
    return None


def _enhanced_plan() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        backend="auto",
        handler="",
        ua_profile="chrome_120_win",
        extra_headers={},
        cookies={},
        respect_robots=True,
        impersonate=None,
        proxies=None,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
    )


async def test_article_preflight_tls_advice_is_attached_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping import scraper_analyzers
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import WebOutboundPolicyDecision

    analysis = _sample_analysis(tls_active=True)
    _force_article_default_plan(monkeypatch, article)
    monkeypatch.setattr(article, "load_and_log_configs", lambda: _article_test_config(preflight=True))
    monkeypatch.setattr(scraper_analyzers, "run_analysis", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(article, "convert_html_to_markdown", lambda content: content)

    async def allow(_url: str, **kwargs: Any) -> WebOutboundPolicyDecision:
        return WebOutboundPolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage=str(kwargs.get("stage", "pre_fetch")),
            source=str(kwargs.get("source", "article_extract")),
        )

    class Response:
        status = 200
        text = "<html><body>Article</body></html>"
        headers: dict[str, str] = {}
        backend = "curl"

    monkeypatch.setattr(article, "decide_web_outbound_policy", allow)
    monkeypatch.setattr(article, "_fetch_with_curl", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(article, "http_fetch", lambda *_args, **_kwargs: pytest.fail("http_fetch should not run"))
    monkeypatch.setattr(
        article,
        "extract_article_with_pipeline",
        lambda _html, url, **_kwargs: _successful_article(url),
    )

    result = await article.scrape_article("https://example.com/article")

    assert result["preflight_analysis"] == {
        "analysis": analysis,
        "advice": {"backend": "curl", "method": "auto", "notes": ["tls_active"]},
    }


async def test_article_preflight_js_advice_is_attached_without_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping import scraper_analyzers
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import WebOutboundPolicyDecision

    analysis = _sample_analysis(js_required=True)
    _force_article_default_plan(monkeypatch, article)
    monkeypatch.setattr(article, "load_and_log_configs", lambda: _article_test_config(preflight=True))
    monkeypatch.setattr(scraper_analyzers, "run_analysis", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(article, "async_playwright", lambda: _FakePlaywright())
    monkeypatch.setattr(article, "convert_html_to_markdown", lambda content: content)

    async def allow(_url: str, **kwargs: Any) -> WebOutboundPolicyDecision:
        return WebOutboundPolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage=str(kwargs.get("stage", "pre_fetch")),
            source=str(kwargs.get("source", "article_extract")),
        )

    monkeypatch.setattr(article, "decide_web_outbound_policy", allow)
    monkeypatch.setattr(article, "_fetch_with_curl", lambda *_args, **_kwargs: pytest.fail("curl should not run"))
    monkeypatch.setattr(article, "http_fetch", lambda *_args, **_kwargs: pytest.fail("http_fetch should not run"))
    monkeypatch.setattr(
        article,
        "extract_article_with_pipeline",
        lambda _html, url, **_kwargs: _successful_article(url, content="Rendered content"),
    )

    result = await article.scrape_article("https://example.com/spa")

    assert result["preflight_analysis"] == {
        "analysis": analysis,
        "advice": {"backend": "auto", "method": "playwright", "notes": ["js_required"]},
    }


class _FakePlaywright:
    chromium: "_FakeChromium"

    def __init__(self) -> None:
        self.chromium = _FakeChromium()

    async def __aenter__(self) -> "_FakePlaywright":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakeChromium:
    async def launch(self, **_kwargs: Any) -> "_FakeBrowser":
        return _FakeBrowser()


class _FakeBrowser:
    async def new_context(self, **_kwargs: Any) -> "_FakeContext":
        return _FakeContext()

    async def close(self) -> None:
        return None


class _FakeContext:
    async def add_cookies(self, _cookies: list[dict[str, Any]]) -> None:
        return None

    async def new_page(self) -> "_FakePage":
        return _FakePage()


class _FakePage:
    async def goto(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def wait_for_timeout(self, _timeout: int) -> None:
        return None

    async def wait_for_load_state(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def content(self) -> str:
        return "<html><body>Rendered content</body></html>"


async def test_enhanced_preflight_js_and_tls_advice_is_attached_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import WebOutboundPolicyDecision

    analysis = _sample_analysis(js_required=True, tls_active=True)
    scraper = enhanced.EnhancedWebScraper(
        config={
            "web_scraper_preflight_analyzers": True,
            "web_scraper_preflight_include_results": True,
        }
    )
    monkeypatch.setattr(scraper.rate_limiter, "acquire", lambda: _noop_async())
    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda _url: (_enhanced_plan(), "auto", ""))
    monkeypatch.setattr(scraper, "_run_preflight_analysis", lambda _url: _return_async(analysis))
    monkeypatch.setattr(
        scraper,
        "_scrape_with_playwright",
        lambda url, *_args, **_kwargs: _return_async(_successful_article(url, content="Rendered content")),
    )

    async def allow(_url: str, **kwargs: Any) -> WebOutboundPolicyDecision:
        return WebOutboundPolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage=str(kwargs.get("stage", "pre_fetch")),
            source=str(kwargs.get("source", "enhanced_scrape")),
        )

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", allow)

    result = await scraper.scrape_article("https://example.com/spa")

    assert result["preflight_analysis"] == {
        "analysis": analysis,
        "advice": {"backend": "curl", "method": "playwright", "notes": ["js_required", "tls_active"]},
    }


async def _return_async(value: Any) -> Any:
    return value


def test_extract_article_with_pipeline_keeps_public_dict_shape() -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    result = article.extract_article_with_pipeline(
        "<html><title>Ignored</title><body><p>Body</p></body></html>",
        "https://example.com/article",
        strategy_order=["trafilatura"],
        fallback_extractor=lambda _html, url: _successful_article(url),
    )

    assert result["url"] == "https://example.com/article"
    assert result["title"] == "Example Title"
    assert result["author"] == "Example Author"
    assert result["date"] == "2026-07-04"
    assert result["content"] == "Example content"
    assert result["extraction_successful"] is True
    assert "extraction_trace" in result


def test_enhanced_scraping_job_to_dict_contract() -> None:
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import JobStatus, ScrapingJob

    job = ScrapingJob(job_id="job-1", url="https://example.com", method="auto")
    job.status = JobStatus.COMPLETED
    job.result = {"ok": True}

    payload = job.to_dict()

    assert payload["job_id"] == "job-1"
    assert payload["url"] == "https://example.com"
    assert payload["method"] == "auto"
    assert payload["priority"] == 2
    assert payload["status"] == "COMPLETED"
    assert payload["result"] == {"ok": True}
    assert payload["cancel_requested"] is False


def test_websearch_initialized_results_contract_and_include_domains_fallback() -> None:
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as websearch

    payload = websearch.initialize_web_search_results_dict(
        {
            "engine": "duckduckgo",
            "content_country": "FR",
            "search_lang": "fr",
            "output_lang": "en",
            "date_range": "w",
            "include_domains": ["example.com"],
            "site_blacklist": ["blocked.example"],
            "safesearch": "off",
            "exactTerms": "capital",
            "excludeTerms": "exclude",
            "filter": "1",
            "geolocation": "Paris",
            "search_result_language": "fr",
            "sort_results_by": "date",
            "google_domain": "google.fr",
        }
    )

    assert set(payload) == INITIALIZED_WEBSEARCH_KEYS
    assert payload["search_engine"] == "duckduckgo"
    assert payload["search_query"] == ""
    assert payload["content_country"] == "FR"
    assert payload["search_lang"] == "fr"
    assert payload["output_lang"] == "en"
    assert payload["date_range"] == "w"
    assert payload["safesearch"] == "off"
    assert payload["site_whitelist"] == ["example.com"]
    assert payload["site_blacklist"] == ["blocked.example"]
    assert payload["results"] == []
    assert payload["warnings"] == []
    assert payload["processing_error"] is None


def test_websearch_processed_provider_result_shape_preserves_metadata_and_extras() -> None:
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as websearch

    payload = websearch.process_web_search_results(
        {
            "query": {
                "original": "query",
                "country": "US",
                "city": "Paris",
                "state": "Ile-de-France",
                "more_results_available": True,
            },
            "web": {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "description": "Result summary",
                        "page_age": "2026-07-04",
                        "profile": {"name": "Example"},
                        "language": "en",
                        "family_friendly": True,
                        "type": "web",
                        "subtype": "article",
                        "thumbnail": {"src": "https://example.com/thumb.jpg"},
                    }
                ]
            },
        },
        "brave",
    )

    assert payload["city"] == "Paris"
    assert payload["state"] == "Ile-de-France"
    assert payload["more_results_available"] is True
    assert payload["results"] == [
        {
            "title": "Result",
            "url": "https://example.com/result",
            "content": "Result summary",
            "metadata": {
                "date_published": "2026-07-04",
                "author": None,
                "source": "Example",
                "language": "en",
                "relevance_score": None,
                "snippet": "Result summary",
                "family_friendly": True,
                "type": "web",
                "subtype": "article",
                "thumbnail": "https://example.com/thumb.jpg",
            },
        }
    ]
