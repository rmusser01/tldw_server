import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as ews
from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext

HTTP_BACKEND = "httpx"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_scrape_with_trafilatura", "_scrape_with_beautifulsoup"])
@pytest.mark.parametrize("classification", ["timeout", None])
async def test_scrape_handles_normalized_http_client_failure(monkeypatch, method, classification):
    monkeypatch.setattr(ews, "afetch", AsyncMock(side_effect=NetworkError("", classification=classification)))
    scraper = ews.EnhancedWebScraper(config={})
    result = await getattr(scraper, method)("https://example.com/article")
    assert result["extraction_successful"] is False
    assert result.get("error_code") == ("extraction_timeout" if classification else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), httpx.ReadTimeout(""), PlaywrightTimeoutError("")])
async def test_scrape_preserves_typed_timeout_without_an_exception_message(monkeypatch, error):
    scraper = ews.EnhancedWebScraper(config={})
    monkeypatch.setattr(scraper, "_fetch_html", AsyncMock(side_effect=error))
    result = await scraper._scrape_with_trafilatura("https://example.com/article")
    assert result["error_code"] == "extraction_timeout"
    assert result["extraction_successful"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_browser_timeout_is_classified_but_cancellation_propagates_with_cleanup(monkeypatch, cancelled):
    error = asyncio.CancelledError() if cancelled else PlaywrightTimeoutError("")
    page = SimpleNamespace(goto=AsyncMock(side_effect=error), close=AsyncMock())
    context = SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
    scraper = ews.EnhancedWebScraper(config={})
    scraper._browser = SimpleNamespace(new_context=AsyncMock(return_value=context))
    monkeypatch.setattr(ews, "resolve_browser_transport_decision", lambda *args, **kwargs: SimpleNamespace(allowed=True))

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await scraper._scrape_with_playwright("https://example.com/article")
    else:
        result = await scraper._scrape_with_playwright("https://example.com/article")
        assert result["error_code"] == "extraction_timeout"
        assert result["extraction_successful"] is False
    page.close.assert_awaited_once()
    context.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [302, 403, 429, 500])
async def test_http_fetch_rejects_non_success_body_before_extraction(monkeypatch, status_code):
    response = SimpleNamespace(
        status_code=status_code,
        text="Please respect our robot policy when crawling us.",
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(ews, "afetch", AsyncMock(return_value=response))
    scraper = ews.EnhancedWebScraper(config={})

    with pytest.raises(ValueError, match=str(status_code)):
        await scraper._fetch_html(
            "https://en.wikipedia.org/wiki/Playwright_(software)",
            headers={}, cookies=None, backend="httpx", impersonate=None, proxies=None,
        )
    response.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_fetch_retains_successful_article_body(monkeypatch):
    response = SimpleNamespace(status_code=200, text="Article body", aclose=AsyncMock())
    monkeypatch.setattr(ews, "afetch", AsyncMock(return_value=response))
    scraper = ews.EnhancedWebScraper(config={})
    html, backend, _ = await scraper._fetch_html(
        "https://example.com/article",
        headers={}, cookies=None, backend="httpx", impersonate=None, proxies=None,
    )
    assert (html, backend) == ("Article body", "httpx")


@pytest.mark.asyncio
async def test_remote_refusal_does_not_trigger_another_transport(monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", lambda *_args, **_kwargs: {"status": 403})
    fallback = AsyncMock(return_value=SimpleNamespace(status_code=200, text="fallback", aclose=AsyncMock()))
    monkeypatch.setattr(ews, "afetch", fallback)
    scraper = ews.EnhancedWebScraper(config={})
    with pytest.raises(ValueError, match="403"):
        await scraper._fetch_html(
            "https://en.wikipedia.org/wiki/Playwright_(software)",
            headers={}, cookies=None, backend="curl", impersonate=None, proxies=None,
        )
    fallback.assert_not_awaited()


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
