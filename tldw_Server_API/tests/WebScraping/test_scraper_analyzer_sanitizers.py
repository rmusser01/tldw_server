import pytest

from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.analyzers import (
    behavioral_detector,
    captcha_detector,
    js_detector,
    rate_limit_profiler,
    robots_checker,
)


pytestmark = pytest.mark.unit


def test_captcha_detector_sanitizes_defensive_failures(monkeypatch):
    def fail_playwright():
        raise RuntimeError("captcha backend failed at /private/captcha.json")

    monkeypatch.setattr(captcha_detector, "sync_playwright", fail_playwright)

    result = captcha_detector.detect_captcha("https://example.com")

    assert result == {"status": "error", "message": "Captcha detection failed."}


def test_behavioral_detector_sanitizes_defensive_failures(monkeypatch):
    def fail_playwright():
        raise RuntimeError("behavior backend failed at /private/behavior.json")

    monkeypatch.setattr(behavioral_detector, "sync_playwright", fail_playwright)

    result = behavioral_detector.detect_honeypots("https://example.com")

    assert result == {"status": "error", "message": "Honeypot detection failed."}


def test_js_detector_sanitizes_defensive_failures(monkeypatch):
    def fail_session(*_args, **_kwargs):
        raise RuntimeError("js backend failed at /private/js.json")

    monkeypatch.setattr(js_detector, "CurlCffiSession", fail_session)
    monkeypatch.setattr(js_detector, "sync_playwright", lambda: None)

    result = js_detector.analyze_js_rendering("https://example.com")

    assert result == {"status": "error", "message": "JavaScript rendering analysis failed."}


def test_robots_checker_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("robots backend failed at /private/robots.txt")

    monkeypatch.setattr(robots_checker, "http_fetch", fail_fetch)

    result = robots_checker.check_robots_txt("https://example.com")

    assert result == {"status": "error", "message": "Robots.txt check failed."}


@pytest.mark.asyncio
async def test_rate_limit_profiler_sanitizes_defensive_failures(monkeypatch):
    async def fail_profiler(*_args, **_kwargs):
        raise RuntimeError("rate limit backend failed at /private/rate-limit.json")

    monkeypatch.setattr(rate_limit_profiler, "_run_rate_limit_profiler", fail_profiler)

    result = await rate_limit_profiler.profile_rate_limits("https://example.com", crawl_delay=0)

    assert result == {"status": "error", "message": "Rate limit profiling failed."}
