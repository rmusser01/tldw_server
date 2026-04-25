import asyncio
from types import SimpleNamespace
import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import (
    CookieManager,
    EnhancedWebScraper,
)


pytestmark = pytest.mark.asyncio


async def test_playwright_guard_fallback(monkeypatch):
    scraper = EnhancedWebScraper()

    async def fake_traf(url, custom_cookies=None, user_agent=None, custom_headers=None, **kwargs):  # noqa: ARG002
        return {"url": url, "title": "t", "author": "a", "date": "", "content": "c", "extraction_successful": True, "method": "trafilatura"}

    from tldw_Server_API.app.core.Security import egress as egress_module
    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url: SimpleNamespace(allowed=True),
    )
    async def allow_robots(*args, **kwargs):
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        allow_robots,
    )

    # Ensure browser is None and trafilatura path is used
    scraper._browser = None
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_traf)

    result = await scraper.scrape_article("https://example.com/x", method="playwright")
    assert result["extraction_successful"] is True
    assert result.get("method") == "trafilatura"


async def test_playwright_guard_strict_policy_blocks_before_navigation(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as scraper_mod
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
        WebOutboundPolicyDecision,
    )

    scraper = EnhancedWebScraper()

    async def fake_policy(*args, **kwargs):
        return WebOutboundPolicyDecision(
            allowed=False,
            mode="strict",
            reason="robots_unreachable",
            stage="pre_fetch",
            source="enhanced_scrape",
        )

    monkeypatch.setattr(scraper_mod, "decide_web_outbound_policy", fake_policy, raising=False)

    async def fail_traf(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("trafilatura should not run when outbound policy blocks")

    async def fail_playwright(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("playwright should not run when outbound policy blocks")

    scraper._browser = object()
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fail_traf)
    monkeypatch.setattr(scraper, "_scrape_with_playwright", fail_playwright)

    result = await scraper.scrape_article("https://example.com/x", method="playwright")

    assert result["extraction_successful"] is False
    assert result["error"] == "Blocked by outbound policy"
    assert result["policy_reason"] == "robots_unreachable"


async def test_cookie_manager_accepts_name_value(tmp_path):
    manager = CookieManager(storage_path=tmp_path / "cookies.json")
    manager.add_cookies("example.com", [{"name": "foo", "value": "bar"}])
    scraper = EnhancedWebScraper(config={})
    scraper.cookie_manager = manager
    cookies = scraper._build_cookie_map(
        "https://example.com",
        custom_cookies=[{"name": "baz", "value": "qux"}],
    )
    assert cookies == {"foo": "bar", "baz": "qux"}
