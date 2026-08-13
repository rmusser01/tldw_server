import dataclasses
import types
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.contracts import PreflightResult
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)


class FakeArticlePolicyChecker:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision

    async def decide(
        self,
        _url: str,
        *,
        respect_robots: bool,
        user_agent: str,
        context: object,
        config: Mapping[str, Any],
    ) -> PolicyDecision:
        return self.decision


class FakeArticleFetchClient:
    def __init__(self, response: FetchResponse | BaseException) -> None:
        self.response = response
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _allowed_article_target(url: str) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )


def _install_article_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: Mapping[str, Any],
    target: PreflightTarget,
    fetch_client: Any,
    extract: Any,
    backend: str = "auto",
    run_preflight: Any | None = None,
    browser_acquire: Any | None = None,
) -> Any:
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
        ArticleLimits,
        ArticlePlan,
        DirectBrowserProfile,
    )

    async def acquire(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("browser should not run")

    async def evaluate_target(*_args: Any, **_kwargs: Any) -> PreflightTarget:
        return target

    default_dependencies = canonical._build_default_dependencies

    def build_dependencies(cookies: Any) -> Any:
        plan = ArticlePlan(
            url=target.url,
            domain="example.com",
            backend=backend,
            browser=DirectBrowserProfile("test", tuple(cookies), 1, 1_000, False, 0),
            limits=ArticleLimits(),
        )
        dependencies = default_dependencies(cookies)
        return dataclasses.replace(
            dependencies,
            load_config=lambda: config,
            resolve_plan=lambda _url, _config: plan,
            evaluate_target=evaluate_target,
            run_preflight=run_preflight or dependencies.run_preflight,
            fetch_client=fetch_client,
            browser=types.SimpleNamespace(acquire=browser_acquire or acquire),
            extract=extract,
            js_required=lambda *_args, **_kwargs: False,
        )

    monkeypatch.setattr(canonical, "_build_default_dependencies", build_dependencies)
    return canonical


def _enhanced_target(url: str, *, allowed: bool, reason: str) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=PolicyDecision(
            allowed=allowed,
            mode="compat",
            reason=reason,
            stage="pre_fetch",
            source="enhanced_scrape",
        ),
        request_context=RuntimeRequestContext(source="enhanced_scrape", stage="pre_fetch"),
    )


@pytest.mark.asyncio
async def test_egress_denied_scrape_article(monkeypatch):
    # Deny egress
    from tldw_Server_API.app.core.Security import egress as eg
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael

    pol = types.SimpleNamespace(allowed=False, reason="deny_test")
    monkeypatch.setattr(eg, "evaluate_url_policy", lambda url: pol)

    result = await ael.scrape_article("https://example.com")
    assert result["extraction_successful"] is False
    assert "Egress denied" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_egress_denied_enhanced_scraper(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    monkeypatch.setattr(enhanced, "preflight_facade", preflight_facade, raising=False)
    monkeypatch.setattr(
        preflight_facade,
        "evaluate_target",
        AsyncMock(return_value=_enhanced_target("https://example.com", allowed=False, reason="deny_test")),
    )

    async def deny_legacy(*_args, **_kwargs):
        return enhanced.WebOutboundPolicyDecision(
            allowed=False,
            mode="compat",
            reason="deny_test",
            stage="pre_fetch",
            source="enhanced_scrape",
        )

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", deny_legacy)
    scraper = enhanced.EnhancedWebScraper()
    result = await scraper.scrape_article("https://example.com")
    assert result["extraction_successful"] is False
    assert "Egress denied" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_article_scrape_blocks_on_shared_policy_before_network(monkeypatch):
    url = "https://example.com/blocked"
    target = PreflightTarget(
        url=url,
        decision=PolicyDecision(
            allowed=False,
            mode="strict",
            reason="robots_unreachable",
            stage="pre_fetch",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )
    fetch_client = FakeArticleFetchClient(AssertionError("fetch should not run"))
    canonical = _install_article_dependencies(
        monkeypatch,
        config={"web_scraper": {"web_scraper_preflight_analyzers": False}},
        target=target,
        fetch_client=fetch_client,
        extract=lambda *_args, **_kwargs: pytest.fail("extraction should not run"),
    )

    result = await canonical.scrape_article(url)

    assert result["extraction_successful"] is False
    assert result["error"] == "Blocked by outbound policy"
    assert result["policy_reason"] == "robots_unreachable"


def test_scrape_article_blocking_sanitizes_policy_evaluation_error(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael

    def fail_policy(*args, **kwargs):
        raise RuntimeError("secret-token")

    monkeypatch.setattr(
        ael,
        "decide_web_outbound_policy_sync",
        fail_policy,
        raising=False,
    )

    result = ael.scrape_article_blocking("https://example.com/private")

    assert result["extraction_successful"] is False
    assert result["error"] == "Outbound policy evaluation failed"


@pytest.mark.asyncio
async def test_extract_links_text_vs_html(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

    scraper = EnhancedWebScraper()

    # Dummy response to avoid real network
    class DummyResp:
        text = "<html><body><a href='/a'>A</a></body></html>"

        async def aclose(self):
            return None

    async def fake_afetch(*args, **kwargs):  # noqa: ANN001, ARG001
        return DummyResp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping.afetch",
        fake_afetch,
    )

    # Case 1: plain text content should trigger a fetch and return links
    links_text = await scraper._extract_links("https://example.com", "plain text no html")
    assert "/a" in links_text

    # Case 2: HTML content provided should be parsed directly
    links_html = await scraper._extract_links("https://example.com", "<a href='https://example.com/b'>B</a>")
    assert "https://example.com/b" in links_html


@pytest.mark.asyncio
async def test_sitemap_scrape_blocks_on_async_policy_before_fetch(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as ews
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

    scraper = EnhancedWebScraper(config={"web_scraper_respect_robots": True})
    calls = {"async": 0, "sync": 0}

    async def fake_policy(url, *, respect_robots, user_agent, source, stage, config, robots_filter=None):
        calls["async"] += 1
        assert url == "https://example.com/sitemap.xml"
        assert respect_robots is True
        assert user_agent == ews.DEFAULT_USER_AGENT
        assert source == "enhanced_sitemap"
        assert stage == "pre_fetch"
        assert config == {"web_scraper": {"web_scraper_respect_robots": True}}
        return ews.WebOutboundPolicyDecision(
            allowed=False,
            mode="strict",
            reason="robots_unreachable",
            stage=stage,
            source=source,
        )

    def fake_sync_policy(*args, **kwargs):
        calls["sync"] += 1
        return ews.WebOutboundPolicyDecision(
            allowed=True,
            mode="strict",
            reason="allowed",
            stage="pre_fetch",
            source="enhanced_sitemap",
        )

    class DummyResp:
        text = "<urlset></urlset>"

        async def aclose(self):
            return None

    async def fake_afetch(*args, **kwargs):
        return DummyResp()

    monkeypatch.setattr(ews, "decide_web_outbound_policy", fake_policy, raising=False)
    monkeypatch.setattr(ews, "decide_web_outbound_policy_sync", fake_sync_policy, raising=False)
    monkeypatch.setattr(ews, "afetch", fake_afetch, raising=False)

    results = await scraper.scrape_sitemap("https://example.com/sitemap.xml")

    assert results == []
    assert calls == {"async": 1, "sync": 0}


def test_provider_missing_keys_google(monkeypatch):
    # Ensure Google provider raises ValueError when API key/engine id missing
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_cfg():
        return {
            "search_engines": {
                "google_search_api_key": "",
                "google_search_engine_id": "",
                "google_search_api_url": "https://customsearch.googleapis.com/customsearch/v1",
                "google_simp_trad_chinese": "1",
                "limit_google_search_to_country": False,
            },
        }

    monkeypatch.setattr(ws, "get_loaded_config", fake_cfg)

    with pytest.raises(ValueError):
        ws.search_web_google("query")


def test_provider_missing_keys_kagi(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_cfg():
        return {"search_engines": {"kagi_search_api_key": ""}}

    monkeypatch.setattr(ws, "get_loaded_config", fake_cfg)

    with pytest.raises(ValueError):
        ws.search_web_kagi("query", 5)


def test_preflight_advice_prefers_playwright_for_js():
    analysis = {"results": {"js": {"status": "success", "js_required": True}}}
    backend, method, result = preflight_facade.apply_preflight_advice(
        PreflightResult(analysis=analysis),
        backend="httpx",
        method="auto",
        backend_setting="auto",
    )
    assert method == "playwright"
    assert result is not None
    assert "js_required" in result.advice.notes


def test_preflight_advice_prefers_curl_for_tls_when_auto():
    analysis = {"results": {"tls": {"status": "active"}}}
    backend, method, result = preflight_facade.apply_preflight_advice(
        PreflightResult(analysis=analysis),
        backend="httpx",
        method="auto",
        backend_setting="auto",
    )
    assert backend == "curl"
    assert result is not None
    assert "tls_active" in result.advice.notes


def test_preflight_advice_respects_explicit_backend():
    analysis = {"results": {"tls": {"status": "active"}}}
    backend, method, result = preflight_facade.apply_preflight_advice(
        PreflightResult(analysis=analysis),
        backend="httpx",
        method="auto",
        backend_setting="httpx",
    )
    assert backend == "httpx"
    assert result is not None
    assert "tls_active" not in result.advice.notes


def test_scoring_includes_fingerprint_and_integrity():
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.scoring.scoring_engine import (
        calculate_difficulty_score,
    )

    results = {
        "fingerprint": {
            "status": "success",
            "detected_services": ["DataDome"],
            "canvas_fingerprinting_signal": True,
            "behavioral_listeners_detected": ["mousemove"],
        },
        "integrity": {
            "status": "success",
            "modified_functions": {"HTMLCanvasElement.prototype.toDataURL": "patched"},
        },
    }

    score = calculate_difficulty_score(results)
    assert score["score"] >= 4


def test_recommendations_include_fingerprint_and_integrity_guidance():
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.recommendations.recommender import (
        generate_recommendations,
    )

    results = {
        "fingerprint": {
            "status": "success",
            "detected_services": ["DataDome"],
            "canvas_fingerprinting_signal": True,
            "behavioral_listeners_detected": ["mousemove"],
        },
        "integrity": {
            "status": "success",
            "modified_functions": {"Date.now": "patched"},
        },
    }

    recs = generate_recommendations(results)
    strategy = " ".join(recs.get("strategy", []))
    tools = " ".join(recs.get("tools", []))

    assert "bot detection" in strategy.lower()
    assert "canvas" in strategy.lower()
    assert "timing" in strategy.lower()
    assert "undetected" in tools.lower() or "playwright-stealth" in tools.lower()


@pytest.mark.asyncio
async def test_article_preflight_prefers_playwright_for_js(monkeypatch):
    playwright_used = {"used": False}

    def fake_extract(*args, **kwargs):
        return {"extraction_successful": True, "content": "ok", "title": "t"}

    fetch_client = FakeArticleFetchClient(AssertionError("fetch should not run"))

    async def acquire(*_args, **_kwargs):
        playwright_used["used"] = True
        return "<html><body><article>hello</article></body></html>"

    canonical = _install_article_dependencies(
        monkeypatch,
        config={
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
                "web_scraper_default_backend": "auto",
            }
        },
        target=_allowed_article_target("https://example.com"),
        fetch_client=fetch_client,
        extract=fake_extract,
        run_preflight=AsyncMock(
            return_value=PreflightResult(analysis={"results": {"js": {"status": "success", "js_required": True}}})
        ),
        browser_acquire=acquire,
    )

    result = await canonical.scrape_article("https://example.com")
    assert result.get("extraction_successful") is True
    assert playwright_used["used"] is True
    assert fetch_client.requests == []


@pytest.mark.asyncio
async def test_article_preflight_prefers_curl_for_tls(monkeypatch):
    fetch_client = FakeArticleFetchClient(
        FetchResponse(
            url="https://example.com",
            status=200,
            text="<html><body>ok</body></html>",
            backend="curl",
        )
    )

    def fake_extract(*args, **kwargs):
        return {"extraction_successful": True, "content": "ok", "title": "t"}

    canonical = _install_article_dependencies(
        monkeypatch,
        config={
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
                "web_scraper_default_backend": "auto",
            }
        },
        target=_allowed_article_target("https://example.com"),
        fetch_client=fetch_client,
        extract=fake_extract,
        run_preflight=AsyncMock(return_value=PreflightResult(analysis={"results": {"tls": {"status": "active"}}})),
    )

    result = await canonical.scrape_article("https://example.com")
    assert result.get("extraction_successful") is True
    assert fetch_client.requests[0].backend == "curl"
