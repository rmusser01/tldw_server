from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers import runner
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.recommendations.recommender import (
    generate_recommendations,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.scoring.scoring_engine import (
    calculate_difficulty_score,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision

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


class FakePolicyChecker:
    async def decide(self, *_args: Any, **_kwargs: Any) -> PolicyDecision:
        return PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="characterization",
        )


def test_analyzer_signatures_match_current_inventory() -> None:
    assert {
        name: str(inspect.signature(getattr(runner, name))) for name in EXPECTED_SIGNATURES
    } == EXPECTED_SIGNATURES  # nosec B101


@pytest.mark.asyncio
async def test_gather_analysis_preserves_order_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def sync_result(name: str, payload: dict[str, Any]):
        def call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            events.append(name)
            return payload

        return call

    async def async_result(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        events.append(name)
        return payload

    monkeypatch.setattr(
        runner,
        "check_robots_txt",
        sync_result("robots", {"status": "success", "crawl_delay": 0.0}),
    )
    monkeypatch.setattr(
        runner,
        "analyze_tls_fingerprint",
        lambda _url: async_result("tls", {"status": "inactive"}),
    )
    monkeypatch.setattr(
        runner,
        "analyze_js_rendering",
        sync_result(
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
        runner,
        "detect_honeypots",
        sync_result("behavioral", {"status": "success", "honeypot_detected": False}),
    )
    monkeypatch.setattr(
        runner,
        "detect_captcha",
        sync_result("captcha", {"status": "success", "captcha_detected": False}),
    )
    monkeypatch.setattr(
        runner,
        "analyze_fingerprinting",
        sync_result("fingerprint", {"status": "success", "detected_services": []}),
    )
    monkeypatch.setattr(
        runner,
        "analyze_function_integrity",
        sync_result("integrity", {"status": "success", "modified_functions": {}}),
    )
    monkeypatch.setattr(
        runner,
        "profile_rate_limits",
        lambda *_args, **_kwargs: async_result(
            "rate_limit",
            {"status": "success", "results": {"requests_sent": 12}},
        ),
    )
    monkeypatch.setattr(
        runner,
        "detect_waf",
        sync_result("waf", {"status": "success", "wafs": []}),
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


@pytest.mark.asyncio
async def test_scoring_recommendations_and_article_payload_are_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article_extractor
    from tldw_Server_API.app.core.Web_Scraping import scraper_analyzers

    score = calculate_difficulty_score(ANALYSIS_RESULTS)
    recommendations = generate_recommendations(ANALYSIS_RESULTS)

    assert score == EXPECTED_SCORE  # nosec B101
    assert recommendations == EXPECTED_RECOMMENDATIONS  # nosec B101

    analysis = {"results": ANALYSIS_RESULTS, "score": score, "recommendations": recommendations}
    expected_payload = {
        "analysis": analysis,
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
    monkeypatch.setattr(scraper_analyzers, "run_analysis", lambda *_args, **_kwargs: analysis)
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
