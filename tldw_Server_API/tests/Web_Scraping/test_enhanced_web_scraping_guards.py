from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as ews
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext

HTTP_BACKEND = "httpx"


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

    assert html == "<html>ok</html>"  # nosec B101
    assert calls["url"] == "https://example.com/article"  # nosec B101
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)  # nosec B101
    assert kwargs["backend"] == "curl"  # nosec B101
    assert kwargs["follow_redirects"] is True  # nosec B101
    assert kwargs["headers"]["X-Test"] == "true"  # nosec B101
    assert kwargs["cookies"] == {"session": "abc"}  # nosec B101


def test_fetch_html_curl_rejects_non_terminal_responses(monkeypatch):
    def fake_fetch(url, **kwargs):
        return {
            "status": 302,
            "headers": {"Location": "https://example.com/final"},
            "text": "",
            "url": url,
            "backend": "curl",
        }

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    scraper = ews.EnhancedWebScraper(config={})

    with pytest.raises(ValueError, match="terminal 2xx"):
        scraper._fetch_html_curl(
            "https://example.com/article",
            headers={"X-Test": "true"},
            cookies={"session": "abc"},
            timeout=5.0,
            impersonate="chrome120",
            proxies=None,
        )


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

    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda url: (plan, HTTP_BACKEND, ""))
    monkeypatch.setattr(ews, "preflight_facade", preflight_facade, raising=False)
    monkeypatch.setattr(
        preflight_facade,
        "evaluate_target",
        AsyncMock(
            return_value=PreflightTarget(
                url="https://example.com/article",
                decision=PolicyDecision(
                    allowed=True,
                    mode="compat",
                    reason="allowed",
                    stage="pre_fetch",
                    source="enhanced_scrape",
                ),
                request_context=RuntimeRequestContext(source="enhanced_scrape", stage="pre_fetch"),
            )
        ),
    )
    monkeypatch.setattr(ews, "_ENHANCED_POLICY_CHECKER", object(), raising=False)

    async def _deny_legacy_policy(*_args, **_kwargs):
        return ews.WebOutboundPolicyDecision(
            allowed=False,
            mode="strict",
            reason="deny_legacy_path",
            stage="pre_fetch",
            source="enhanced_scrape",
        )

    monkeypatch.setattr(ews, "decide_web_outbound_policy", _deny_legacy_policy)
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

    assert result["extraction_successful"] is True  # nosec B101
    fake_scrape.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrape_article_blocks_when_robots_disallows(monkeypatch):
    scraper = _build_scraper(monkeypatch)
    fake_scrape = AsyncMock()
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_scrape)

    monkeypatch.setattr(
        preflight_facade,
        "evaluate_target",
        AsyncMock(
            return_value=PreflightTarget(
                url="https://example.com/article",
                decision=PolicyDecision(
                    allowed=False,
                    mode="strict",
                    reason="robots_disallowed",
                    stage="pre_fetch",
                    source="enhanced_scrape",
                ),
                request_context=RuntimeRequestContext(source="enhanced_scrape", stage="pre_fetch"),
            )
        ),
    )

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is False  # nosec B101
    assert result["error"] == "Blocked by outbound policy"  # nosec B101
    assert result["policy_reason"] == "robots_disallowed"  # nosec B101
    fake_scrape.assert_not_awaited()
