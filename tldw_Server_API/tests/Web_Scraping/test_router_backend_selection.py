import types
import pytest


@pytest.mark.asyncio
async def test_scrape_article_backend_playwright_skips_httpx(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url: types.SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr(ael, "load_and_log_configs", lambda: {"web_scraper": {}})
    async def allow_robots(*args, **kwargs):
        return True

    monkeypatch.setattr(ael, "is_allowed_by_robots_async", allow_robots)
    monkeypatch.setattr(ael, "_js_required", lambda *args, **kwargs: False)

    rules = {
        "domains": {
            "example.com": {
                "backend": "playwright",
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
            }
        }
    }
    monkeypatch.setattr(ael.ScraperRouter, "load_rules_from_yaml", lambda path: rules)

    def fake_handler(html, url):
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(ael, "resolve_handler", lambda _: fake_handler)

    def fail_http_fetch(*args, **kwargs):
        raise AssertionError("http_fetch should not be called for playwright backend")

    monkeypatch.setattr(ael, "http_fetch", fail_http_fetch)

    class DummyPage:
        async def goto(self, *args, **kwargs):
            return None

        async def wait_for_timeout(self, *args, **kwargs):
            return None

        async def wait_for_load_state(self, *args, **kwargs):
            return None

        async def content(self):
            return "<html><body>ok</body></html>"

        async def close(self):
            return None

    class DummyContext:
        async def add_cookies(self, *args, **kwargs):
            return None

        async def new_page(self):
            return DummyPage()

        async def close(self):
            return None

    class DummyBrowser:
        async def new_context(self, *args, **kwargs):
            return DummyContext()

        async def close(self):
            return None

    class DummyChromium:
        async def launch(self, *args, **kwargs):
            return DummyBrowser()

    class DummyPlaywright:
        chromium = DummyChromium()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(ael, "async_playwright", lambda: DummyPlaywright())

    result = await ael.scrape_article("https://example.com/path")
    assert result["title"] == "handled"


@pytest.mark.asyncio
async def test_scrape_article_backend_curl_uses_curl(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
    from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url: types.SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr(ael, "load_and_log_configs", lambda: {"web_scraper": {}})
    async def allow_robots(*args, **kwargs):
        return True

    monkeypatch.setattr(ael, "is_allowed_by_robots_async", allow_robots)
    monkeypatch.setattr(ael, "_js_required", lambda *args, **kwargs: False)

    rules = {
        "domains": {
            "example.com": {
                "backend": "curl",
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
            }
        }
    }
    monkeypatch.setattr(ael.ScraperRouter, "load_rules_from_yaml", lambda path: rules)

    def fake_handler(html, url):
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(ael, "resolve_handler", lambda _: fake_handler)

    class FakeFetchClient:
        def __init__(self):
            self.requests = []

        def fetch(self, request):
            self.requests.append(request)
            return FetchResponse(
                url=request.url,
                status=200,
                text="<html><body>ok</body></html>",
                headers={},
                backend=request.backend,
            )

    fetch_client = FakeFetchClient()
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)

    result = await ael.scrape_article("https://example.com/path")
    assert fetch_client.requests[0].backend == "curl"
    assert result["content"] == "handled-content"


@pytest.mark.asyncio
async def test_enhanced_scraper_router_backend_playwright(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper
    from tldw_Server_API.app.core.Security import egress as egress_module

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
