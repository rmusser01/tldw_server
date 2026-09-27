from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import (
    CookieManager,
    EnhancedWebScraper,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision

pytestmark = pytest.mark.asyncio


async def test_playwright_guard_fallback(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as scraper_mod
    from tldw_Server_API.app.core.Web_Scraping import filters as filters_mod

    scraper = EnhancedWebScraper()
    checker_decision = AsyncMock(
        return_value=PolicyDecision(
            allowed=True,
            mode="strict",
            reason="allowed",
            stage="pre_fetch",
            source="enhanced_scrape",
        )
    )
    robots_fetch = Mock(side_effect=AssertionError("robots HTTP fetch must stay offline"))
    monkeypatch.setattr(filters_mod, "http_fetch", robots_fetch)

    async def fake_traf(url, custom_cookies=None, user_agent=None, custom_headers=None, **kwargs):  # noqa: ARG002
        return {
            "url": url,
            "title": "t",
            "author": "a",
            "date": "",
            "content": "c",
            "extraction_successful": True,
            "method": "trafilatura",
        }

    # Ensure browser is None and trafilatura path is used
    scraper._browser = None
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_traf)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(type(scraper_mod._ENHANCED_POLICY_CHECKER), "decide", checker_decision)
        result = await scraper.scrape_article("https://example.com/x", method="playwright")
        checker_decision.assert_awaited_once()

    assert "decide" not in scraper_mod._ENHANCED_POLICY_CHECKER.__dict__
    robots_fetch.assert_not_called()
    assert result["extraction_successful"] is True
    assert result.get("method") == "trafilatura"


async def test_playwright_guard_strict_policy_blocks_before_navigation(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as scraper_mod
    from tldw_Server_API.app.core.Web_Scraping import filters as filters_mod

    scraper = EnhancedWebScraper()

    checker_decision = AsyncMock(
        return_value=PolicyDecision(
            allowed=False,
            mode="strict",
            reason="robots_unreachable",
            stage="pre_fetch",
            source="enhanced_scrape",
        )
    )
    robots_fetch = Mock(side_effect=AssertionError("robots HTTP fetch must stay offline"))
    monkeypatch.setattr(filters_mod, "http_fetch", robots_fetch)

    async def fail_traf(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("trafilatura should not run when outbound policy blocks")

    async def fail_playwright(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("playwright should not run when outbound policy blocks")

    scraper._browser = object()
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fail_traf)
    monkeypatch.setattr(scraper, "_scrape_with_playwright", fail_playwright)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(type(scraper_mod._ENHANCED_POLICY_CHECKER), "decide", checker_decision)
        result = await scraper.scrape_article("https://example.com/x", method="playwright")
        checker_decision.assert_awaited_once()

    assert "decide" not in scraper_mod._ENHANCED_POLICY_CHECKER.__dict__
    robots_fetch.assert_not_called()
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
