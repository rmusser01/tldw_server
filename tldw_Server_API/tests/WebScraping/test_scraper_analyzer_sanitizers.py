import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article_extractor
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.analyzers import (
    behavioral_detector,
    captcha_detector,
    js_detector,
    rate_limit_profiler,
    robots_checker,
)


pytestmark = pytest.mark.unit


_LEAKY_ERROR = "backend exploded at /tmp/secret-token with api_key=abc123"


def _assert_safe_text(value):
    text = str(value)
    assert "backend exploded" not in text
    assert "/tmp/secret-token" not in text
    assert "api_key" not in text.lower()


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


def test_article_pipeline_schema_import_failure_sanitizes_trace_detail(monkeypatch):
    original_import = __import__

    def fail_fetchers_import(name, *args, **kwargs):
        if name == "tldw_Server_API.app.core.Watchlists.fetchers":
            raise ImportError(_LEAKY_ERROR)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_fetchers_import)

    result = article_extractor.extract_article_with_pipeline(
        "<html><body><h1>Title</h1></body></html>",
        "https://example.com/post",
        strategy_order=["schema"],
        schema_rules={"title_xpath": "//h1"},
    )

    trace_entry = result["extraction_trace"][0]
    assert trace_entry["reason"] == "schema_import_error"
    _assert_safe_text(trace_entry)


def test_article_pipeline_schema_failure_sanitizes_trace_detail(monkeypatch):
    from tldw_Server_API.app.core.Watchlists import fetchers

    def fail_extract(*_args, **_kwargs):
        raise RuntimeError(_LEAKY_ERROR)

    monkeypatch.setattr(fetchers, "extract_schema_fields", fail_extract)

    result = article_extractor.extract_article_with_pipeline(
        """
        <html>
          <body>
            <article>
              <h1>Example Title</h1>
              <p>First paragraph.</p>
            </article>
          </body>
        </html>
        """,
        "https://example.com/post",
        strategy_order=["schema"],
        schema_rules={"title_xpath": "//article//h1", "content_xpath": "//article//p"},
    )

    trace_entry = result["extraction_trace"][0]
    assert trace_entry["reason"] == "schema_error"
    _assert_safe_text(trace_entry)


def test_article_pipeline_handler_failure_sanitizes_trace_detail():
    def fail_handler(*_args, **_kwargs):
        raise RuntimeError(_LEAKY_ERROR)

    result = article_extractor.extract_article_with_pipeline(
        "<html><body><p>Body</p></body></html>",
        "https://example.com/post",
        strategy_order=["schema"],
        handler=fail_handler,
    )

    trace_entry = result["extraction_trace"][0]
    assert trace_entry["reason"] == "handler_error"
    _assert_safe_text(trace_entry)


def test_article_pipeline_fallback_extractor_failure_sanitizes_trace_detail():
    def fail_extractor(*_args, **_kwargs):
        raise RuntimeError(_LEAKY_ERROR)

    result = article_extractor.extract_article_with_pipeline(
        "<html><body><p>Body</p></body></html>",
        "https://example.com/post",
        strategy_order=["trafilatura"],
        fallback_extractor=fail_extractor,
    )

    trace_entry = result["extraction_trace"][0]
    assert trace_entry["reason"] == "extractor_error"
    _assert_safe_text(trace_entry)
