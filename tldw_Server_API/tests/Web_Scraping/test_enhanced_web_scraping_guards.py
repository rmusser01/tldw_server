from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as ews


def test_fetch_html_curl_routes_through_http_client_fetch(monkeypatch):
    calls: dict[str, object] = {}

    def fake_fetch(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "text": "<html>ok</html>",
            "url": url,
            "backend": "curl",
        }

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    scraper = ews.EnhancedWebScraper(config={})
    html = scraper._fetch_html_curl(
        "https://example.com/article",
        headers={"X-Test": "true"},
        cookies={"session": "abc"},
        timeout=5.0,
        impersonate="chrome120",
        proxies=None,
    )

    assert html == "<html>ok</html>"
    assert calls["url"] == "https://example.com/article"
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["backend"] == "curl"
    assert kwargs["follow_redirects"] is True
    assert kwargs["headers"]["X-Test"] == "true"
    assert kwargs["cookies"] == {"session": "abc"}


def _build_scraper(monkeypatch):
    scraper = ews.EnhancedWebScraper(config={})

    async def _acquire():
        return None

    scraper.rate_limiter.acquire = _acquire

    plan = SimpleNamespace(
        respect_robots=True,
        ua_profile="chrome_120_win",
        extra_headers={},
        cookies={},
        impersonate=None,
        proxies=None,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
        backend="auto",
    )

    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda url: (plan, "httpx", ""))
    monkeypatch.setattr(scraper, "_run_preflight_analysis", AsyncMock(return_value=None))
    monkeypatch.setattr(scraper, "_apply_preflight_advice", lambda *args: ("httpx", "trafilatura", []))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        lambda url: SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr(ews, "increment_counter", lambda *args, **kwargs: None)

    return scraper


@pytest.mark.asyncio
async def test_scrape_article_allows_when_robots_check_errors(monkeypatch):
    scraper = _build_scraper(monkeypatch)
    fake_scrape = AsyncMock(
        return_value={
            "url": "https://example.com/article",
            "content": "ok",
            "extraction_successful": True,
        }
    )
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_scrape)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        AsyncMock(side_effect=RuntimeError("robots unavailable")),
    )

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is True
    fake_scrape.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrape_article_blocks_when_robots_disallows(monkeypatch):
    scraper = _build_scraper(monkeypatch)
    fake_scrape = AsyncMock()
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_scrape)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        AsyncMock(return_value=False),
    )

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is False
    assert result["error"] == "Blocked by robots policy"
    fake_scrape.assert_not_awaited()
