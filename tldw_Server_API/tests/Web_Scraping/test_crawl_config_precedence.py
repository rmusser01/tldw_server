from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib as extractor_mod
import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as enhanced_core_mod
import tldw_Server_API.app.services.web_scraping_service as legacy_svc_mod
import tldw_Server_API.app.services.enhanced_web_scraping_service as enhanced_svc_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ScrapeMethod
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import (
    EnhancedWebScraper,
    ScrapingJob,
    ScrapingJobQueue,
)
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


def _cfg(
    *,
    max_pages: int = 42,
    strategy: str = "default",
    include_external: bool = True,
    score_threshold: float = 0.33,
) -> dict[str, object]:
    return {
        "web_scraper": {
            "web_crawl_max_pages": max_pages,
            "web_crawl_strategy": strategy,
            "web_crawl_include_external": include_external,
            "web_crawl_score_threshold": score_threshold,
        }
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_uses_config_defaults_when_request_values_omitted(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True
    captured: dict[str, object] = {}

    async def fake_scrape_recursive(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"method": "Recursive Scraping", "articles": []}

    async def fake_store_ephemeral(result, task_id, user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg())
    monkeypatch.setattr(svc, "_scrape_recursive", fake_scrape_recursive)
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    result = await svc.process_web_scraping_task(
        scrape_method="Recursive Scraping",
        url_input="https://example.com",
        max_pages=None,
        max_depth=2,
        mode="ephemeral",
        crawl_strategy=None,
        include_external=None,
        score_threshold=None,
    )

    args = captured.get("args")
    kwargs = captured.get("kwargs")
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[1] == 42
    assert kwargs.get("crawl_strategy") == "default"
    assert kwargs.get("include_external") is True
    assert kwargs.get("score_threshold") == pytest.approx(0.33, abs=1e-9)

    crawl_cfg = result.get("crawl_config", {})
    assert crawl_cfg.get("max_pages_source") == "config_default"
    assert crawl_cfg.get("requested_max_pages") is None
    assert crawl_cfg.get("effective_max_pages") == 42
    assert crawl_cfg.get("strategy_source") == "config_default"
    assert crawl_cfg.get("include_external_source") == "config_default"
    assert crawl_cfg.get("score_threshold_source") == "config_default"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_request_values_override_config(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True
    captured: dict[str, object] = {}

    async def fake_scrape_recursive(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"method": "Recursive Scraping", "articles": []}

    async def fake_store_ephemeral(result, task_id, user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg(max_pages=99, strategy="default"))
    monkeypatch.setattr(svc, "_scrape_recursive", fake_scrape_recursive)
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    result = await svc.process_web_scraping_task(
        scrape_method="Recursive Scraping",
        url_input="https://example.com",
        max_pages=7,
        max_depth=2,
        mode="ephemeral",
        crawl_strategy="best_first",
        include_external=False,
        score_threshold=0.15,
    )

    args = captured.get("args")
    kwargs = captured.get("kwargs")
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[1] == 7
    assert kwargs.get("crawl_strategy") == "best_first"
    assert kwargs.get("include_external") is False
    assert kwargs.get("score_threshold") == pytest.approx(0.15, abs=1e-9)

    crawl_cfg = result.get("crawl_config", {})
    assert crawl_cfg.get("max_pages_source") == "request"
    assert crawl_cfg.get("requested_max_pages") == 7
    assert crawl_cfg.get("effective_max_pages") == 7
    assert crawl_cfg.get("strategy_source") == "request"
    assert crawl_cfg.get("include_external_source") == "request"
    assert crawl_cfg.get("score_threshold_source") == "request"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_sanitizes_unexpected_runtime_error(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True

    async def raise_scraper_error(*args, **kwargs):
        raise RuntimeError("enhanced scraper leaked /private/web/cache/token")

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg())
    monkeypatch.setattr(svc, "_scrape_by_url_level", raise_scraper_error)

    with pytest.raises(HTTPException) as exc_info:
        await svc.process_web_scraping_task(
            scrape_method="URL Level",
            url_input="https://example.com",
            url_level=2,
            mode="ephemeral",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Enhanced web scraping task failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_max_pages_100_is_treated_as_request_override(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True
    captured: dict[str, object] = {}

    async def fake_scrape_recursive(*args, **kwargs):
        captured["args"] = args
        return {"method": "Recursive Scraping", "articles": []}

    async def fake_store_ephemeral(result, task_id, user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg(max_pages=12))
    monkeypatch.setattr(svc, "_scrape_recursive", fake_scrape_recursive)
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    result = await svc.process_web_scraping_task(
        scrape_method="Recursive Scraping",
        url_input="https://example.com",
        max_pages=100,
        max_depth=2,
        mode="ephemeral",
    )

    args = captured.get("args")
    assert isinstance(args, tuple)
    assert args[1] == 100
    crawl_cfg = result.get("crawl_config", {})
    assert crawl_cfg.get("max_pages_source") == "request"
    assert crawl_cfg.get("effective_max_pages") == 100


@pytest.mark.unit
def test_process_web_scraping_endpoint_omitted_max_pages_forwards_none(
    client_user_only, monkeypatch
):
    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    captured = {}

    async def fake_process_web_scraping_task(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "results": []}

    monkeypatch.setattr(
        media_mod,
        "process_web_scraping_task",
        fake_process_web_scraping_task,
        raising=True,
    )

    payload = {
        "scrape_method": "Recursive Scraping",
        "url_input": "https://example.com",
        "max_depth": 2,
        "mode": "ephemeral",
    }
    response = client_user_only.post("/api/v1/media/process-web-scraping", json=payload)
    assert response.status_code == 200
    assert captured.get("max_pages") is None


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scrape_method", "analysis_intent", "expected_call"),
    [
        ("Individual URLs", False, "multiple"),
        ("Sitemap", True, "sitemap"),
    ],
)
async def test_enhanced_service_propagates_analysis_intent_to_scraper(
    monkeypatch,
    scrape_method,
    analysis_intent,
    expected_call,
):
    captured: dict[str, dict[str, object]] = {}

    class FakeScraper:
        async def scrape_multiple(self, _urls, **kwargs):
            captured["multiple"] = kwargs
            return []

        async def scrape_sitemap(self, _url, **kwargs):
            captured["sitemap"] = kwargs
            return []

    svc = WebScrapingService()
    svc._initialized = True
    svc.scraper = FakeScraper()

    async def fake_store_ephemeral(result, _task_id, _user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg())
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    await svc.process_web_scraping_task(
        scrape_method=scrape_method,
        url_input="https://example.com/sitemap.xml" if scrape_method == "Sitemap" else "https://example.com/article",
        max_pages=5,
        summarize_checkbox=analysis_intent,
        mode="ephemeral",
    )

    assert captured[expected_call]["allow_llm_extraction"] is analysis_intent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scraping_job_queue_propagates_analysis_intent_to_article_scrape():
    captured: dict[str, object] = {}

    class FakeScraper:
        async def scrape_article(self, _url, _method, **kwargs):
            captured.update(kwargs)
            return {"extraction_successful": True}

    queue = ScrapingJobQueue(parent_scraper=FakeScraper())
    job = ScrapingJob(
        job_id="job-analysis-disabled",
        url="https://example.com/article",
        method="trafilatura",
        metadata={"allow_llm_extraction": False},
    )

    await queue._execute_job(job)

    assert captured["allow_llm_extraction"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sitemap_crawl_retains_analysis_intent_for_scrape_multiple(monkeypatch):
    captured: dict[str, object] = {}
    scraper = EnhancedWebScraper(config={})

    async def fake_policy(url, **kwargs):
        return enhanced_core_mod.WebOutboundPolicyDecision(
            allowed=True,
            mode="strict",
            reason="allowed",
            stage=kwargs["stage"],
            source=kwargs["source"],
        )

    class DummyResponse:
        text = "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'><url><loc>https://example.com/a</loc></url></urlset>"

        async def aclose(self):
            return None

    async def fake_afetch(**_kwargs):
        return DummyResponse()

    async def fake_scrape_multiple(_urls, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(enhanced_core_mod, "decide_web_outbound_policy", fake_policy)
    monkeypatch.setattr(enhanced_core_mod, "afetch", fake_afetch)
    monkeypatch.setattr(scraper, "scrape_multiple", fake_scrape_multiple)

    await scraper.scrape_sitemap(
        "https://example.com/sitemap.xml",
        allow_llm_extraction=False,
    )

    assert captured["allow_llm_extraction"] is False


@pytest.mark.unit
@pytest.mark.parametrize("analysis_intent", [False, True])
def test_url_level_scraper_explicitly_forwards_analysis_intent(monkeypatch, analysis_intent):
    captured: list[object] = []

    def fake_scrape_article_blocking(_url, *, allow_llm_extraction=None):
        captured.append(allow_llm_extraction)
        return {"url": _url, "extraction_successful": True}

    monkeypatch.setattr(extractor_mod, "collect_internal_links", lambda _url: {"https://example.com/article"})
    monkeypatch.setattr(extractor_mod, "scrape_article_blocking", fake_scrape_article_blocking)

    extractor_mod.scrape_by_url_level(
        "https://example.com",
        1,
        allow_llm_extraction=analysis_intent,
    )

    assert captured == [analysis_intent]


@pytest.mark.unit
@pytest.mark.parametrize("analysis_intent", [False, True])
def test_sitemap_scraper_explicitly_forwards_analysis_intent(monkeypatch, analysis_intent):
    captured: list[object] = []
    sitemap = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/article</loc></url>
    </urlset>
    """

    def fake_scrape_article_blocking(_url, *, allow_llm_extraction=None):
        captured.append(allow_llm_extraction)
        return {"url": _url, "extraction_successful": True}

    monkeypatch.setattr(
        extractor_mod,
        "decide_web_outbound_policy_sync",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr(
        extractor_mod,
        "http_fetch",
        lambda **_kwargs: {"status": 200, "text": sitemap},
    )
    monkeypatch.setattr(extractor_mod, "scrape_article_blocking", fake_scrape_article_blocking)

    extractor_mod.scrape_from_sitemap(
        "https://example.com/sitemap.xml",
        allow_llm_extraction=analysis_intent,
    )

    assert captured == [analysis_intent]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["trafilatura", "playwright", "beautifulsoup"])
async def test_async_scrapers_offload_extraction_pipeline(monkeypatch, method):
    scraper = EnhancedWebScraper(config={})
    offload_calls: list[dict[str, object]] = []

    async def fake_to_thread(func, *args, **kwargs):
        offload_calls.append({"func": func, "args": args, "kwargs": kwargs})
        return func(*args, **kwargs)

    async def fake_fetch_html(*_args, **_kwargs):
        return "<html><body>article</body></html>", "httpx", 0.01

    def fake_extract(html, url, **kwargs):
        return {
            "url": url,
            "title": "Article",
            "content": html,
            "extraction_successful": True,
            "allow_llm_extraction": kwargs["allow_llm_extraction"],
        }

    class FakePage:
        async def goto(self, *_args, **_kwargs):
            return None

        async def wait_for_load_state(self, *_args, **_kwargs):
            return None

        async def content(self):
            return "<html><body>article</body></html>"

        async def close(self):
            return None

    class FakeContext:
        async def set_extra_http_headers(self, _headers):
            return None

        async def add_cookies(self, _cookies):
            return None

        async def new_page(self):
            return FakePage()

        async def close(self):
            return None

    class FakeBrowser:
        async def new_context(self, **_kwargs):
            return FakeContext()

    monkeypatch.setattr(enhanced_core_mod.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(scraper, "_fetch_html", fake_fetch_html)
    monkeypatch.setattr(scraper, "_extract_from_html_with_pipeline", fake_extract)
    monkeypatch.setattr(scraper, "_apply_dedup", lambda _url, data: data)
    scraper._browser = FakeBrowser()

    scrape_method = getattr(scraper, f"_scrape_with_{method}")
    result = await scrape_method(
        "https://example.com/article",
        allow_llm_extraction=False,
    )

    assert result["allow_llm_extraction"] is False
    assert len(offload_calls) == 1
    assert offload_calls[0]["func"] is fake_extract


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scrape_method", "analysis_intent"),
    [
        (ScrapeMethod.INDIVIDUAL, False),
        (ScrapeMethod.INDIVIDUAL, True),
        (ScrapeMethod.SITEMAP, False),
        (ScrapeMethod.SITEMAP, True),
    ],
)
async def test_friendly_ingest_propagates_analysis_intent_to_direct_scrapers(
    monkeypatch,
    scrape_method,
    analysis_intent,
):
    captured: dict[str, object] = {}

    async def fake_scrape_article(_url, *, custom_cookies=None, allow_llm_extraction=None):
        captured["allow_llm_extraction"] = allow_llm_extraction
        return {
            "url": "https://example.com/article",
            "content": "article",
            "extraction_successful": True,
        }

    def fake_scrape_from_sitemap(_url, *, allow_llm_extraction=None):
        captured["allow_llm_extraction"] = allow_llm_extraction
        return [
            {
                "url": "https://example.com/article",
                "content": "article",
                "extraction_successful": True,
            }
        ]

    monkeypatch.setattr(legacy_svc_mod, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(legacy_svc_mod, "scrape_from_sitemap", fake_scrape_from_sitemap)

    request = SimpleNamespace(
        scrape_method=scrape_method,
        urls=["https://example.com/sitemap.xml"],
        titles=[],
        authors=[],
        keywords=[],
        perform_analysis=analysis_intent,
        api_name=None,
        use_cookies=False,
    )
    usage_log = SimpleNamespace(log_event=lambda *_args, **_kwargs: None)

    await legacy_svc_mod.ingest_web_content_orchestrate(
        request,
        SimpleNamespace(client_id="test-user"),
        usage_log,
    )

    assert captured["allow_llm_extraction"] is analysis_intent
