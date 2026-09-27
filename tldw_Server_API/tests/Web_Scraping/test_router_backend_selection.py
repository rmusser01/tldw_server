import types
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_scrape_article_backend_playwright_skips_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
        ArticleLimits,
        ArticlePlan,
        DirectBrowserProfile,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
    from tldw_Server_API.app.core.Web_Scraping.runtime import (
        PolicyDecision,
        RuntimeRequestContext,
    )

    def fake_handler(html: str, url: str) -> dict[str, object]:
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    class FailFetchClient:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def fetch(self, request: object) -> object:
            self.requests.append(request)
            raise AssertionError("fetch client should not be called for playwright backend")

    def extract(html: str, url: str, **kwargs: Any) -> dict[str, object]:
        handler = kwargs["handler"]
        assert handler is not None
        return handler(html, url)

    class Browser:
        async def acquire(self, *_args: Any, **_kwargs: Any) -> str:
            return "<html><body>ok</body></html>"

    class Executor:
        async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    fetch_client = FailFetchClient()

    async def evaluate_target(url: str, **_kwargs: Any) -> PreflightTarget:
        return PreflightTarget(
            url=url,
            decision=PolicyDecision(True, "test", "allowed", "pre_fetch", "article_extract"),
            request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        )

    plan = ArticlePlan(
        url="https://example.com/path",
        domain="example.com",
        backend="playwright",
        handler="tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
        headers={"User-Agent": "test-agent"},
        browser=DirectBrowserProfile("test-agent", (), 1, 1_000, False, 0),
        limits=ArticleLimits(4_096, 8_192),
    )

    dependencies = canonical.ArticleDependencies(
        load_config=lambda: {"web_scraper": {"web_scraper_preflight_analyzers": False}},
        resolve_plan=lambda _url, _config: plan,
        evaluate_target=evaluate_target,
        run_preflight=lambda *_args, **_kwargs: None,
        apply_preflight_advice=lambda result, **kwargs: (kwargs["backend"], kwargs["method"], result),
        fetch_client=fetch_client,
        browser=Browser(),
        executor=Executor(),
        extract=extract,
        build_preflight_context=lambda *_args, **_kwargs: object(),
        preflight_options=canonical.preflight_facade.PreflightOptions.from_mapping,
        public_preflight_payload=lambda *_args, **_kwargs: None,
        resolve_handler=lambda _path: fake_handler,
        js_required=lambda *_args, **_kwargs: False,
        convert_content=lambda content: content,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        clock=lambda: 0.0,
        log=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = await ael.scrape_article("https://example.com/path")
    assert result["title"] == "handled"
    assert fetch_client.requests == []


@pytest.mark.asyncio
async def test_scrape_article_backend_curl_uses_curl(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
        ArticleLimits,
        ArticlePlan,
        DirectBrowserProfile,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
    from tldw_Server_API.app.core.Web_Scraping.runtime import (
        FetchResponse,
        PolicyDecision,
        RuntimeRequestContext,
    )

    def fake_handler(html: str, url: str) -> dict[str, object]:
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    class FetchClient:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        def fetch(self, request: Any) -> FetchResponse:
            self.requests.append(request)
            return FetchResponse(request.url, 200, {}, "<html><body>ok</body></html>", request.backend)

    class Browser:
        async def acquire(self, *_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("browser fallback is not expected")

    class Executor:
        async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    def extract(html: str, url: str, **kwargs: Any) -> dict[str, object]:
        handler = kwargs["handler"]
        assert handler is not None
        return handler(html, url)

    async def evaluate_target(url: str, **_kwargs: Any) -> PreflightTarget:
        return PreflightTarget(
            url=url,
            decision=PolicyDecision(True, "test", "allowed", "pre_fetch", "article_extract"),
            request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        )

    fetch_client = FetchClient()
    plan = ArticlePlan(
        url="https://example.com/path",
        domain="example.com",
        backend="curl",
        handler="tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
        headers={"User-Agent": "test-agent"},
        browser=DirectBrowserProfile("test-agent", (), 1, 1_000, False, 0),
        limits=ArticleLimits(4_096, 8_192),
    )
    dependencies = canonical.ArticleDependencies(
        load_config=lambda: {"web_scraper": {"web_scraper_preflight_analyzers": False}},
        resolve_plan=lambda _url, _config: plan,
        evaluate_target=evaluate_target,
        run_preflight=lambda *_args, **_kwargs: None,
        apply_preflight_advice=lambda result, **kwargs: (kwargs["backend"], kwargs["method"], result),
        fetch_client=fetch_client,
        browser=Browser(),
        executor=Executor(),
        extract=extract,
        build_preflight_context=lambda *_args, **_kwargs: object(),
        preflight_options=canonical.preflight_facade.PreflightOptions.from_mapping,
        public_preflight_payload=lambda *_args, **_kwargs: None,
        resolve_handler=lambda _path: fake_handler,
        js_required=lambda *_args, **_kwargs: False,
        convert_content=lambda content: content,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        clock=lambda: 0.0,
        log=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(canonical, "_build_default_dependencies", lambda _cookies: dependencies)

    result = await ael.scrape_article("https://example.com/path")
    assert fetch_client.requests[0].backend == "curl"
    assert result["content"] == "handled-content"


@pytest.mark.asyncio
async def test_enhanced_scraper_router_backend_playwright(monkeypatch):
    from tldw_Server_API.app.core.Security import egress as egress_module
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url: types.SimpleNamespace(allowed=True),
    )

    async def allow_robots(*args, **kwargs):
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        allow_robots,
    )

    rules = {
        "domains": {
            "example.com": {
                "backend": "playwright",
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
            }
        }
    }
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.scraper_router.ScraperRouter.load_rules_from_yaml",
        lambda path: rules,
    )

    scraper = EnhancedWebScraper(config={"custom_scrapers_yaml_path": "unused"})

    async def fake_playwright(*args, **kwargs):
        return {"url": "https://example.com/path", "extraction_successful": True, "method": "playwright"}

    async def fail_trafilatura(*args, **kwargs):
        raise AssertionError("trafilatura path should not be used for playwright backend")

    monkeypatch.setattr(scraper, "_scrape_with_playwright", fake_playwright)
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fail_trafilatura)

    result = await scraper.scrape_article("https://example.com/path")
    assert result.get("method") == "playwright"
