from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article_extractor
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline

pytestmark = pytest.mark.unit


_LEAKY_ERROR = "backend exploded at /tmp/secret-token with api_key=abc123"


def _install_extraction_dependencies(monkeypatch, **overrides):
    dependencies = replace(pipeline.build_default_dependencies(), **overrides)
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)


def _assert_safe_text(value):
    text = str(value)
    assert "backend exploded" not in text
    assert "/tmp/secret-token" not in text
    assert "api_key" not in text.lower()


class _FailingPageManager:
    async def __aenter__(self):
        raise RuntimeError(_LEAKY_ERROR)

    async def __aexit__(self, *_args):
        return None


class _FailingBrowserProbe:
    def open_page(self, _options):
        return _FailingPageManager()


def _browser_context(**overrides):
    values = {
        "browser": _FailingBrowserProbe(),
        "browser_identity": lambda: {"User-Agent": "sanitizer-test"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_captcha_detector_sanitizes_defensive_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.captcha_detector import (
        _detect_captcha,
    )

    result = await _detect_captcha("https://example.com", _browser_context())

    assert result == {"status": "error", "message": "Captcha detection failed."}


@pytest.mark.asyncio
async def test_behavioral_detector_sanitizes_defensive_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.behavioral_detector import (
        _detect_honeypots,
    )

    result = await _detect_honeypots("https://example.com", _browser_context())

    assert result == {"status": "error", "message": "Honeypot detection failed."}


@pytest.mark.asyncio
async def test_js_detector_sanitizes_defensive_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.js_detector import (
        _analyze_js_rendering,
    )

    class FailingHttpProbe:
        async def get(self, _request):
            raise RuntimeError(_LEAKY_ERROR)

    result = await _analyze_js_rendering(
        "https://example.com",
        _browser_context(http=FailingHttpProbe()),
    )

    assert result == {"status": "error", "message": "JavaScript rendering analysis failed."}


@pytest.mark.asyncio
async def test_fingerprint_detector_sanitizes_defensive_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.fingerprint_analyzer import (
        _analyze_fingerprinting,
    )

    result = await _analyze_fingerprinting("https://example.com", _browser_context())

    assert result == {
        "status": "error",
        "message": "Fingerprint analysis failed.",
        "error_code": "analyzer_error",
        "detected_services": [],
        "canvas_fingerprinting_signal": False,
        "behavioral_listeners_detected": [],
    }


@pytest.mark.asyncio
async def test_integrity_detector_sanitizes_defensive_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.integrity_analyzer import (
        _analyze_function_integrity,
    )

    result = await _analyze_function_integrity("https://example.com", _browser_context())

    assert result == {
        "status": "error",
        "message": "Function integrity analysis failed.",
        "error_code": "analyzer_error",
        "modified_functions": {},
    }


@pytest.mark.asyncio
async def test_robots_checker_sanitizes_injected_probe_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.robots_checker import (
        _check_robots_txt,
    )

    class FailingHttpProbe:
        async def get(self, _request):
            raise RuntimeError("robots backend failed at /private/robots.txt")

    result = await _check_robots_txt(
        "https://example.com",
        SimpleNamespace(http=FailingHttpProbe()),
    )

    assert result == {
        "status": "error",
        "message": "Robots.txt check failed.",
        "error_code": "analyzer_error",
    }


@pytest.mark.asyncio
async def test_rate_limit_profiler_sanitizes_injected_probe_failures():
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.rate_limit_profiler import (
        _profile_rate_limits,
    )

    class FailingHttpProbe:
        async def get(self, _request):
            raise RuntimeError("rate limit backend failed at /private/rate-limit.json")

    async def unexpected_sleep(_delay):
        raise AssertionError("profiling must fail before sleeping")

    def browser_identity():
        return {"User-Agent": "sanitizer-test"}

    context = SimpleNamespace(
        http=FailingHttpProbe(),
        controls=SimpleNamespace(sleep=unexpected_sleep),
        browser_identity=browser_identity,
    )
    result = await _profile_rate_limits(
        "https://example.com",
        context,
        crawl_delay=0,
    )

    assert result == {
        "status": "error",
        "message": "Rate limit profiling failed.",
        "error_code": "analyzer_error",
    }


def test_article_pipeline_schema_import_failure_sanitizes_trace_detail(monkeypatch):
    def fail_extract(*_args, **_kwargs):
        raise ImportError(_LEAKY_ERROR)

    _install_extraction_dependencies(monkeypatch, extract_schema_fields=fail_extract)

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
    def fail_extract(*_args, **_kwargs):
        raise RuntimeError(_LEAKY_ERROR)

    _install_extraction_dependencies(monkeypatch, extract_schema_fields=fail_extract)

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


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [
        (ImportError(_LEAKY_ERROR), "schema_import_error"),
        (RuntimeError(_LEAKY_ERROR), "schema_error"),
    ],
    ids=["import", "generic"],
)
def test_article_pipeline_schema_validation_failure_is_sanitized_and_falls_back(
    monkeypatch,
    error,
    expected_reason,
):
    def fail_validation(*_args, **_kwargs):
        raise error

    _install_extraction_dependencies(monkeypatch, validate_selector_rules=fail_validation)

    result = article_extractor.extract_article_with_pipeline(
        "<html><body><h1>Title</h1></body></html>",
        "https://example.com/post",
        strategy_order=["schema", "trafilatura"],
        schema_rules={"title_xpath": "//h1"},
        fallback_extractor=lambda *_args, **_kwargs: {
            "url": "https://example.com/post",
            "extraction_successful": True,
            "title": "Fallback",
            "content": "Body",
        },
    )

    trace_entry = result["extraction_trace"][0]
    assert trace_entry["reason"] == expected_reason
    assert result["extraction_strategy"] == "trafilatura"
    assert result["extraction_trace"][1]["status"] == "success"
    _assert_safe_text(trace_entry)


def test_article_pipeline_schema_validation_failure_is_not_retried(monkeypatch):
    validation_calls = 0
    retry_delays = []
    retry_metrics = []

    def fail_validation(*_args, **_kwargs):
        nonlocal validation_calls
        validation_calls += 1
        raise RuntimeError(_LEAKY_ERROR)

    def record_counter(metric_name, value=1, labels=None):
        if metric_name == "extraction_retry_total":
            retry_metrics.append((value, labels))

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "2")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "10")
    monkeypatch.setenv("EXTRACTOR_RETRY_JITTER_MS", "0")
    _install_extraction_dependencies(
        monkeypatch,
        validate_selector_rules=fail_validation,
        sleep=retry_delays.append,
        increment_counter=record_counter,
    )

    result = article_extractor.extract_article_with_pipeline(
        "<html><body><h1 data-no-retry>Title</h1></body></html>",
        "https://example.com/schema-validation-no-retry",
        strategy_order=["schema"],
        schema_rules={"title_xpath": "//h1[@data-no-retry]"},
    )

    assert (validation_calls, retry_delays, retry_metrics) == (1, [], [])
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
