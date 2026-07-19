from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts import PreflightResult
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchRequest, FetchResponse, PolicyDecision
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers import runner
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.recommendations.recommender import (
    generate_recommendations,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.scoring.scoring_engine import (
    calculate_difficulty_score,
)

EXPECTED_SIGNATURES = {
    "check_robots_txt": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_tls_fingerprint": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_js_rendering": "(url: 'str') -> 'dict[str, Any]'",
    "detect_honeypots": "(url: 'str', scan_depth: 'ScanDepth' = 'default') -> 'dict[str, Any]'",
    "detect_captcha": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_fingerprinting": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_function_integrity": "(url: 'str') -> 'dict[str, Any]'",
    "profile_rate_limits": "(url: 'str', crawl_delay: 'float | None', impersonate: 'bool' = False) -> 'dict[str, Any]'",
    "detect_waf": "(url: 'str', find_all: 'bool' = False) -> 'dict[str, Any]'",
}


ANALYSIS_RESULTS = {
    "js": {"status": "success", "js_required": True, "is_spa": True},
    "tls": {"status": "active"},
    "captcha": {
        "status": "success",
        "captcha_detected": True,
        "trigger_condition": "on page load",
    },
    "behavioral": {"status": "success", "honeypot_detected": True},
    "rate_limit": {"status": "success", "results": {"requests_sent": 3, "blocking_code": 429}},
    "waf": {"status": "success", "wafs": [("DataDome", None)]},
    "fingerprint": {
        "status": "success",
        "detected_services": ["DataDome"],
        "canvas_fingerprinting_signal": True,
        "behavioral_listeners_detected": ["mousemove"],
    },
    "integrity": {
        "status": "success",
        "modified_functions": {
            "HTMLCanvasElement.prototype.toDataURL": "patched",
            "Date.now": "patched",
        },
    },
}

EXPECTED_SCORE = {"score": 10, "label": "Very Hard"}

EXPECTED_RECOMMENDATIONS = {
    "tools": [
        "A CAPTCHA solving service (e.g. 2Captcha, Anti-Captcha).",
        "A headless browser such as Playwright or Selenium for JavaScript rendering.",
        "A library with browser impersonation (e.g. curl_cffi) or a full headless browser.",
        "A pool of high-quality rotating proxies (residential or mobile).",
        "An anti-detection browser automation library (e.g. playwright-stealth, undetected-chromedriver).",
    ],
    "strategy": [
        "Add delays between requests (3-5 seconds) and rotate request headers.",
        "Add random delays and jitter between actions to appear more human.",
        "Avoid interacting with invisible elements; drive the page like a human.",
        "Canvas fingerprinting detected. Use automation with built-in evasion (not basic requests).",
        "Integrate the CAPTCHA solver when challenges appear.",
        "Site modifies canvas functions (strong fingerprinting). Avoid basic automation.",
        "Site monitors timing patterns. Vary your request timing to look less robotic.",
        "Site monitors user behavior (mouse, keyboard, scroll). Simulate realistic interaction.",
        "Site uses advanced bot detection (DataDome). Use playwright-stealth or undetected-chromedriver.",
        "Standard Python HTTP clients are blocked; impersonate a real browser.",
        "Use a modern, non-generic User-Agent and align headers with real browsers.",
        "Wait for dynamic content to load before extracting data.",
    ],
}


@pytest.mark.unit
def test_scoring_and_recommendation_shims_reexport_canonical_callables() -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight.recommendations.recommender import (
        generate_recommendations as canonical_recommend,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.scoring.scoring_engine import (
        calculate_difficulty_score as canonical_score,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.recommendations.recommender import (
        generate_recommendations as legacy_recommend,
    )
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.scoring.scoring_engine import (
        calculate_difficulty_score as legacy_score,
    )

    assert legacy_score is canonical_score  # nosec B101
    assert legacy_recommend is canonical_recommend  # nosec B101


class FakePolicyChecker:
    def __init__(self, decision: PolicyDecision | None = None) -> None:
        self.decision = decision or _policy_decision(allowed=True)
        self.calls = 0

    async def decide(self, *_args: Any, **_kwargs: Any) -> PolicyDecision:
        self.calls += 1
        return self.decision


class FakeFetchClient:
    def __init__(self, responses: list[FetchResponse]) -> None:
        self.responses = responses
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class _NoopPreflightContext:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def _policy_decision(*, allowed: bool) -> PolicyDecision:
    return PolicyDecision(
        allowed=allowed,
        mode="compat",
        reason="allowed" if allowed else "robots_disallowed",
        stage="pre_fetch",
        source="characterization",
    )


def _article_response() -> FetchResponse:
    return FetchResponse(
        url="https://example.com/article",
        status=200,
        headers={"Content-Type": "text/html"},
        text="<html><body><article>article</article></body></html>",
        backend="httpx",
    )


def _configure_article_consumer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_results: bool | None,
    backend: str = "auto",
) -> Any:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article_extractor

    web_scraper_config: dict[str, Any] = {
        "web_scraper_preflight_analyzers": True,
        "web_scraper_preflight_timeout_s": 0,
        "web_scraper_respect_robots": True,
    }
    if include_results is not None:
        web_scraper_config["web_scraper_preflight_include_results"] = include_results

    monkeypatch.setattr(
        article_extractor,
        "load_and_log_configs",
        lambda: {"web_scraper": web_scraper_config},
    )
    monkeypatch.setattr(article_extractor, "_js_required", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        article_extractor.ScraperRouter,
        "load_rules_from_yaml",
        lambda _path: {
            "domains": {
                "example.com": {
                    "backend": backend,
                    "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
                }
            }
        },
    )
    monkeypatch.setattr(article_extractor, "resolve_handler", lambda _path: lambda *_args: {})
    monkeypatch.setattr(article_extractor, "observe_histogram", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(article_extractor, "increment_counter", lambda *_args, **_kwargs: None)
    return article_extractor


def _enhanced_plan(*, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
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
        backend=backend,
        handler="",
    )


def _configure_enhanced_consumer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_results: bool | None,
    backend: str = "auto",
) -> tuple[Any, Any]:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    config: dict[str, Any] = {"web_scraper_preflight_analyzers": True}
    if include_results is not None:
        config["web_scraper_preflight_include_results"] = include_results
    scraper = enhanced.EnhancedWebScraper(config=config)

    async def acquire() -> None:
        return None

    scraper.rate_limiter.acquire = acquire
    monkeypatch.setattr(
        scraper,
        "_resolve_scrape_plan",
        lambda _url: (_enhanced_plan(backend=backend), backend, ""),
    )
    monkeypatch.setattr(enhanced, "increment_counter", lambda *_args, **_kwargs: None)
    return enhanced, scraper


def test_analyzer_and_public_entry_point_signatures_match_current_inventory() -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article_extractor
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

    assert {
        name: str(inspect.signature(getattr(runner, name))) for name in EXPECTED_SIGNATURES
    } == EXPECTED_SIGNATURES  # nosec B101
    analyzer_entry_points = {name: getattr(runner, name) for name in EXPECTED_SIGNATURES}
    assert {  # nosec B101
        name: inspect.iscoroutinefunction(entry_point) for name, entry_point in analyzer_entry_points.items()
    } == {
        "check_robots_txt": False,
        "analyze_tls_fingerprint": True,
        "analyze_js_rendering": False,
        "detect_honeypots": False,
        "detect_captcha": False,
        "analyze_fingerprinting": False,
        "analyze_function_integrity": False,
        "profile_rate_limits": True,
        "detect_waf": False,
    }
    public_entry_points = {
        "gather_analysis": runner.gather_analysis,
        "run_analysis": runner.run_analysis,
        "article_scrape_article": article_extractor.scrape_article,
        "enhanced_scrape_article": EnhancedWebScraper.scrape_article,
    }
    assert {  # nosec B101
        name: str(inspect.signature(entry_point)) for name, entry_point in public_entry_points.items()
    } == {
        "gather_analysis": "(url: 'str', *, find_all: 'bool' = False, impersonate: 'bool' = False, scan_depth: 'ScanDepth | None' = None) -> 'AnalysisOutput'",
        "run_analysis": "(url: 'str', *, find_all: 'bool' = False, impersonate: 'bool' = False, scan_depth: 'ScanDepth | None' = None) -> 'AnalysisOutput'",
        "article_scrape_article": "(url: str, custom_cookies: list[dict[str, Any]] | None = None, *, allow_llm_extraction: bool = True) -> dict[str, typing.Any]",
        "enhanced_scrape_article": "(self, url: str, method: str = 'auto', custom_cookies: list[dict[str, Any]] | None = None, user_agent: str | None = None, custom_headers: dict[str, str] | None = None, allow_llm_extraction: bool = True) -> dict[str, typing.Any]",
    }
    assert {  # nosec B101
        name: inspect.iscoroutinefunction(entry_point) for name, entry_point in public_entry_points.items()
    } == {
        "gather_analysis": True,
        "run_analysis": False,
        "article_scrape_article": True,
        "enhanced_scrape_article": True,
    }


@pytest.mark.asyncio
async def test_gather_analysis_preserves_order_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight import runner as canonical_runner

    events: list[str] = []
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    context = _NoopPreflightContext()

    def analyzer_result(name: str, payload: dict[str, Any]):
        async def call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            events.append(name)
            calls.append((name, _args, _kwargs))
            return payload

        return call

    monkeypatch.setattr(
        canonical_runner,
        "_check_robots_txt",
        analyzer_result("robots", {"status": "success", "crawl_delay": 2.5}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_tls_fingerprint",
        analyzer_result("tls", {"status": "inactive"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_js_rendering",
        analyzer_result(
            "js",
            {
                "status": "success",
                "js_required": False,
                "is_spa": False,
                "content_difference_%": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_detect_honeypots",
        analyzer_result("behavioral", {"status": "success", "honeypot_detected": False}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_detect_captcha",
        analyzer_result("captcha", {"status": "success", "captcha_detected": False}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_fingerprinting",
        analyzer_result("fingerprint", {"status": "success", "detected_services": []}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_function_integrity",
        analyzer_result("integrity", {"status": "success", "modified_functions": {}}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_profile_rate_limits",
        analyzer_result("rate_limit", {"status": "success", "results": {"requests_sent": 12}}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_detect_waf",
        analyzer_result("waf", {"status": "success", "wafs": []}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_default_policy_checker",
        lambda: FakePolicyChecker(),
    )
    monkeypatch.setattr(
        canonical_runner,
        "build_execution_context",
        lambda *_args, **_kwargs: context,
    )

    result = await runner.gather_analysis(
        "https://example.com",
        find_all=True,
        impersonate=True,
        scan_depth="deep",
    )

    assert events == [  # nosec B101
        "robots",
        "tls",
        "js",
        "behavioral",
        "captcha",
        "fingerprint",
        "integrity",
        "rate_limit",
        "waf",
    ]
    assert list(result) == ["results", "score", "recommendations"]  # nosec B101
    assert list(result["results"]) == events  # nosec B101
    assert calls == [  # nosec B101
        ("robots", ("https://example.com", context), {}),
        ("tls", ("https://example.com", context), {}),
        ("js", ("https://example.com", context), {}),
        ("behavioral", ("https://example.com", context, "deep"), {}),
        ("captcha", ("https://example.com", context), {}),
        ("fingerprint", ("https://example.com", context), {}),
        ("integrity", ("https://example.com", context), {}),
        ("rate_limit", ("https://example.com", context, 2.5, True), {}),
        ("waf", ("https://example.com", context, True, None), {}),
    ]
    assert result == {  # nosec B101
        "results": {
            "robots": {"status": "success", "crawl_delay": 2.5},
            "tls": {"status": "inactive"},
            "js": {
                "status": "success",
                "js_required": False,
                "is_spa": False,
                "content_difference_%": 0.0,
            },
            "behavioral": {"status": "success", "honeypot_detected": False},
            "captcha": {"status": "success", "captcha_detected": False},
            "fingerprint": {"status": "success", "detected_services": []},
            "integrity": {"status": "success", "modified_functions": {}},
            "rate_limit": {"status": "success", "results": {"requests_sent": 12}},
            "waf": {"status": "success", "wafs": []},
        },
        "score": {"score": 0, "label": "Easy"},
        "recommendations": {
            "tools": ["Standard HTTP clients (requests, aiohttp) should be sufficient."],
            "strategy": ["A simple, direct scraping approach is likely to work."],
        },
    }
    assert context.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_gather_analysis_isolates_middle_analyzer_failure_and_runs_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight import runner as canonical_runner

    events: list[str] = []
    context = _NoopPreflightContext()

    def analyzer_result(name: str, payload: dict[str, Any]):
        async def call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            events.append(name)
            return payload

        return call

    def fail_captcha(*_args: Any, **_kwargs: Any) -> Awaitable[dict[str, Any]]:
        events.append("captcha")
        raise RuntimeError("credential-bearing analyzer setup failure")

    monkeypatch.setattr(
        canonical_runner,
        "_check_robots_txt",
        analyzer_result("robots", {"status": "success", "crawl_delay": None}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_tls_fingerprint",
        analyzer_result("tls", {"status": "inactive"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_js_rendering",
        analyzer_result("js", {"status": "success"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_detect_honeypots",
        analyzer_result("behavioral", {"status": "success"}),
    )
    monkeypatch.setattr(canonical_runner, "_detect_captcha", fail_captcha)
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_fingerprinting",
        analyzer_result("fingerprint", {"status": "success"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_analyze_function_integrity",
        analyzer_result("integrity", {"status": "success"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_profile_rate_limits",
        analyzer_result("rate_limit", {"status": "success"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_detect_waf",
        analyzer_result("waf", {"status": "success"}),
    )
    monkeypatch.setattr(
        canonical_runner,
        "_default_policy_checker",
        lambda: FakePolicyChecker(),
    )
    monkeypatch.setattr(
        canonical_runner,
        "build_execution_context",
        lambda *_args, **_kwargs: context,
    )

    result = await runner.gather_analysis("https://example.com")

    assert events == [  # nosec B101
        "robots",
        "tls",
        "js",
        "behavioral",
        "captcha",
        "fingerprint",
        "integrity",
        "rate_limit",
        "waf",
    ]
    assert result["results"]["captcha"] == {  # nosec B101
        "status": "error",
        "message": "Captcha detection failed.",
        "error_code": "analyzer_error",
    }
    assert context.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_scoring_recommendations_and_article_payload_are_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article_extractor

    score = calculate_difficulty_score(ANALYSIS_RESULTS)
    recommendations = generate_recommendations(ANALYSIS_RESULTS)

    assert score == EXPECTED_SCORE  # nosec B101
    assert recommendations == EXPECTED_RECOMMENDATIONS  # nosec B101

    analysis = {"results": ANALYSIS_RESULTS, "score": score, "recommendations": recommendations}
    preflight_result = PreflightResult(analysis=analysis)
    public_analysis = {
        **analysis,
        "results": {
            **ANALYSIS_RESULTS,
            "waf": {"status": "success", "wafs": [["DataDome", None]]},
        },
    }
    expected_payload = {
        "analysis": public_analysis,
        "advice": {
            "backend": "curl",
            "method": "playwright",
            "notes": ["js_required", "tls_active"],
        },
    }

    monkeypatch.setattr(
        article_extractor,
        "load_and_log_configs",
        lambda: {
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
                "web_scraper_preflight_include_results": True,
                "web_scraper_preflight_timeout_s": 0,
                "web_scraper_default_backend": "auto",
                "web_scraper_respect_robots": True,
            }
        },
    )
    monkeypatch.setattr(article_extractor, "_ARTICLE_POLICY_CHECKER", FakePolicyChecker())
    monkeypatch.setattr(article_extractor, "_js_required", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        article_extractor.ScraperRouter,
        "load_rules_from_yaml",
        lambda _path: {
            "domains": {
                "example.com": {
                    "backend": "auto",
                    "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
                }
            }
        },
    )
    run_preflight = AsyncMock(return_value=preflight_result)
    monkeypatch.setattr(article_extractor.preflight_facade, "run_preflight", run_preflight)
    monkeypatch.setattr(
        article_extractor,
        "extract_article_with_pipeline",
        lambda *_args, **_kwargs: {"extraction_successful": True, "content": "article"},
    )

    class FakePage:
        async def goto(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def wait_for_load_state(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def content(self) -> str:
            return "<html><body><article>article</article></body></html>"

    class FakeContext:
        async def add_cookies(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            return None

    class FakeBrowser:
        async def new_context(self, *_args: Any, **_kwargs: Any) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch(self, *_args: Any, **_kwargs: Any) -> FakeBrowser:
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

    class FakePlaywrightContext:
        async def __aenter__(self) -> FakePlaywright:
            return FakePlaywright()

        async def __aexit__(self, *_args: Any) -> bool:
            return False

    monkeypatch.setattr(article_extractor, "async_playwright", FakePlaywrightContext)
    article_result = await article_extractor.scrape_article("https://example.com/article")

    assert article_result["preflight_analysis"] == expected_payload  # nosec B101
    run_preflight.assert_awaited_once()


@pytest.mark.asyncio
async def test_enhanced_consumer_payload_is_current(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    score = calculate_difficulty_score(ANALYSIS_RESULTS)
    recommendations = generate_recommendations(ANALYSIS_RESULTS)
    analysis = {"results": ANALYSIS_RESULTS, "score": score, "recommendations": recommendations}
    expected_payload = {
        "analysis": analysis,
        "advice": {
            "backend": "curl",
            "method": "playwright",
            "notes": ["js_required", "tls_active"],
        },
    }

    scraper = enhanced.EnhancedWebScraper(
        config={
            "web_scraper_preflight_analyzers": True,
            "web_scraper_preflight_include_results": True,
        }
    )

    async def acquire() -> None:
        return None

    async def allow_policy(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(allowed=True)

    async def scrape_with_playwright(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"extraction_successful": True, "content": "enhanced"}

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
        handler="",
    )
    scraper.rate_limiter.acquire = acquire
    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda _url: (plan, "auto", ""))
    monkeypatch.setattr(scraper, "_run_preflight_analysis", lambda _url: _return(analysis))
    monkeypatch.setattr(scraper, "_scrape_with_playwright", scrape_with_playwright)
    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", allow_policy)
    monkeypatch.setattr(enhanced, "increment_counter", lambda *_args, **_kwargs: None)

    enhanced_result = await scraper.scrape_article("https://example.com/article")

    assert enhanced_result["preflight_analysis"] == expected_payload  # nosec B101


async def _return(value: dict[str, Any]) -> dict[str, Any]:
    return value


@pytest.mark.asyncio
async def test_article_policy_denial_prevents_preflight_and_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article_extractor = _configure_article_consumer(monkeypatch, include_results=True)
    policy_checker = FakePolicyChecker(_policy_decision(allowed=False))
    build_context = Mock(side_effect=AssertionError("preflight context created"))
    run_preflight = AsyncMock(side_effect=AssertionError("preflight ran"))
    fetch_client = FakeFetchClient([])
    extract = Mock(side_effect=AssertionError("extraction ran"))

    monkeypatch.setattr(article_extractor, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(article_extractor.preflight_facade, "build_execution_context", build_context)
    monkeypatch.setattr(article_extractor.preflight_facade, "run_preflight", run_preflight)
    monkeypatch.setattr(article_extractor, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(article_extractor, "extract_article_with_pipeline", extract)

    result = await article_extractor.scrape_article("https://example.com/article")

    assert policy_checker.calls == 1  # nosec B101
    assert result["extraction_successful"] is False  # nosec B101
    assert result["policy_reason"] == "robots_disallowed"  # nosec B101
    assert "preflight_analysis" not in result  # nosec B101
    build_context.assert_not_called()
    run_preflight.assert_not_awaited()
    assert fetch_client.requests == []  # nosec B101
    extract.assert_not_called()


@pytest.mark.asyncio
async def test_article_preflight_failure_is_advisory_and_preserves_http_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article_extractor = _configure_article_consumer(
        monkeypatch,
        include_results=True,
        backend="httpx",
    )
    fetch_client = FakeFetchClient([_article_response()])
    extraction_calls: list[object] = []

    def extract(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        extraction_calls.append(object())
        return {"extraction_successful": True, "content": "article"}

    run_preflight = AsyncMock(side_effect=RuntimeError("preflight failed"))
    monkeypatch.setattr(article_extractor, "_ARTICLE_POLICY_CHECKER", FakePolicyChecker())
    monkeypatch.setattr(article_extractor, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(article_extractor.preflight_facade, "run_preflight", run_preflight)
    monkeypatch.setattr(article_extractor, "extract_article_with_pipeline", extract)

    result = await article_extractor.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is True  # nosec B101
    assert fetch_client.requests[0].backend == "httpx"  # nosec B101
    assert len(extraction_calls) == 1  # nosec B101
    assert "preflight_analysis" not in result  # nosec B101
    run_preflight.assert_awaited_once()


@pytest.mark.asyncio
async def test_article_preflight_cancellation_propagates_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article_extractor = _configure_article_consumer(monkeypatch, include_results=True)
    fetch_client = FakeFetchClient([_article_response()])
    extraction_calls: list[object] = []

    def extract(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        extraction_calls.append(object())
        return {"extraction_successful": True, "content": "article"}

    run_preflight = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(article_extractor, "_ARTICLE_POLICY_CHECKER", FakePolicyChecker())
    monkeypatch.setattr(article_extractor, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(article_extractor.preflight_facade, "run_preflight", run_preflight)
    monkeypatch.setattr(article_extractor, "extract_article_with_pipeline", extract)

    with pytest.raises(asyncio.CancelledError):
        await article_extractor.scrape_article("https://example.com/article")

    run_preflight.assert_awaited_once()
    assert fetch_client.requests == []  # nosec B101
    assert extraction_calls == []  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("include_results", [False, None], ids=["false", "omitted"])
async def test_article_successful_advice_omits_payload_when_results_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
    include_results: bool | None,
) -> None:
    article_extractor = _configure_article_consumer(
        monkeypatch,
        include_results=include_results,
    )
    fetch_client = FakeFetchClient([_article_response()])
    run_preflight = AsyncMock(return_value=PreflightResult(analysis={"results": {"tls": {"status": "active"}}}))
    monkeypatch.setattr(article_extractor, "_ARTICLE_POLICY_CHECKER", FakePolicyChecker())
    monkeypatch.setattr(article_extractor, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(article_extractor.preflight_facade, "run_preflight", run_preflight)
    monkeypatch.setattr(
        article_extractor,
        "extract_article_with_pipeline",
        lambda *_args, **_kwargs: {"extraction_successful": True, "content": "article"},
    )

    result = await article_extractor.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is True  # nosec B101
    assert fetch_client.requests[0].backend == "curl"  # nosec B101
    assert "preflight_analysis" not in result  # nosec B101
    run_preflight.assert_awaited_once()


@pytest.mark.asyncio
async def test_enhanced_policy_denial_prevents_preflight_and_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enhanced, scraper = _configure_enhanced_consumer(monkeypatch, include_results=True)
    extraction_calls: list[object] = []

    async def deny_policy(*_args: Any, **_kwargs: Any) -> Any:
        return enhanced.WebOutboundPolicyDecision(
            allowed=False,
            mode="compat",
            reason="robots_disallowed",
            stage="pre_fetch",
            source="characterization",
        )

    async def preflight_should_not_run(_url: str) -> None:
        raise AssertionError("preflight ran")

    async def extraction_should_not_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        extraction_calls.append(object())
        return {"extraction_successful": True}

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", deny_policy)
    monkeypatch.setattr(scraper, "_run_preflight_analysis", preflight_should_not_run)
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", extraction_should_not_run)

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is False  # nosec B101
    assert result["policy_reason"] == "robots_disallowed"  # nosec B101
    assert extraction_calls == []  # nosec B101
    assert "preflight_analysis" not in result  # nosec B101


@pytest.mark.asyncio
async def test_enhanced_preflight_failure_is_advisory_and_preserves_method_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import scraper_analyzers

    enhanced, scraper = _configure_enhanced_consumer(
        monkeypatch,
        include_results=True,
        backend="httpx",
    )
    scrape_calls: list[dict[str, Any]] = []

    async def allow_policy(*_args: Any, **_kwargs: Any) -> Any:
        return enhanced.WebOutboundPolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="characterization",
        )

    def fail_analysis(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("preflight failed")

    async def scrape_with_beautifulsoup(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        scrape_calls.append(kwargs)
        return {"extraction_successful": True, "content": "enhanced"}

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", allow_policy)
    monkeypatch.setattr(scraper_analyzers, "run_analysis", fail_analysis)
    monkeypatch.setattr(scraper, "_scrape_with_beautifulsoup", scrape_with_beautifulsoup)

    result = await scraper.scrape_article("https://example.com/article", method="beautifulsoup")

    assert result["extraction_successful"] is True  # nosec B101
    assert len(scrape_calls) == 1  # nosec B101
    assert scrape_calls[0]["backend"] == "httpx"  # nosec B101
    assert "preflight_analysis" not in result  # nosec B101


@pytest.mark.asyncio
async def test_enhanced_preflight_cancellation_returns_error_without_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enhanced, scraper = _configure_enhanced_consumer(monkeypatch, include_results=True)
    extraction_calls: list[object] = []

    async def allow_policy(*_args: Any, **_kwargs: Any) -> Any:
        return enhanced.WebOutboundPolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="characterization",
        )

    async def cancel_preflight(_url: str) -> None:
        raise asyncio.CancelledError

    async def extraction_should_not_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        extraction_calls.append(object())
        return {"extraction_successful": True}

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", allow_policy)
    monkeypatch.setattr(scraper, "_run_preflight_analysis", cancel_preflight)
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", extraction_should_not_run)

    result = await scraper.scrape_article("https://example.com/article")

    assert result == {  # nosec B101
        "url": "https://example.com/article",
        "error": "",
        "extraction_successful": False,
    }
    assert extraction_calls == []  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("include_results", [False, None], ids=["false", "omitted"])
async def test_enhanced_successful_advice_omits_payload_when_results_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
    include_results: bool | None,
) -> None:
    enhanced, scraper = _configure_enhanced_consumer(
        monkeypatch,
        include_results=include_results,
    )
    scrape_calls: list[dict[str, Any]] = []

    async def allow_policy(*_args: Any, **_kwargs: Any) -> Any:
        return enhanced.WebOutboundPolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="characterization",
        )

    async def successful_preflight(_url: str) -> dict[str, Any]:
        return {"results": {"tls": {"status": "active"}}}

    async def scrape_with_trafilatura(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        scrape_calls.append(kwargs)
        return {"extraction_successful": True, "content": "enhanced"}

    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", allow_policy)
    monkeypatch.setattr(scraper, "_run_preflight_analysis", successful_preflight)
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", scrape_with_trafilatura)

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is True  # nosec B101
    assert scrape_calls[0]["backend"] == "curl"  # nosec B101
    assert "preflight_analysis" not in result  # nosec B101
