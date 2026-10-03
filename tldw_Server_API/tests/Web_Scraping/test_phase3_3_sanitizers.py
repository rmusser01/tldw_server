import pytest

import tldw_Server_API.app.core.http_client as http_client_module
from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws
from tldw_Server_API.app.core.WebSearch import Web_Search as legacy_ws


pytestmark = pytest.mark.unit


_LEAKY_ERROR = "backend exploded at /tmp/secret-token with api_key=abc123"


def _assert_safe_text(value: object) -> None:
    text = str(value)
    assert "backend exploded" not in text
    assert "/tmp/secret-token" not in text
    assert "api_key" not in text.lower()


class _LeakyContains(dict):
    def __contains__(self, _key):
        raise TypeError(_LEAKY_ERROR)


class _FakeLogger:
    def __init__(self):
        self.errors: list[str] = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def exception(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def opt(self, **_kwargs):
        return self


class _FakeLazyLogger(_FakeLogger):
    def __init__(self):
        super().__init__()
        self.debug_messages: list[str] = []
        self.lazy_calls = 0
        self._lazy = False

    def opt(self, *, lazy: bool = False, **_kwargs):
        self._lazy = lazy
        if lazy:
            self.lazy_calls += 1
        return self

    def debug(self, message, *args, **_kwargs):
        rendered_args = [arg() if self._lazy and callable(arg) else arg for arg in args]
        self.debug_messages.append(str(message).format(*rendered_args))
        self._lazy = False


def test_brave_smoke_debug_logs_are_redacted_and_lazy(monkeypatch):
    logger = _FakeLazyLogger()
    monkeypatch.setattr(ws, "logging", logger)
    monkeypatch.setattr(
        ws,
        "search_web_brave",
        lambda *_args, **_kwargs: {
            "query": {"original": "cake"},
            "web": {
                "results": [
                    {
                        "title": "Cake",
                        "url": "https://example.com/path?api_key=secret",
                        "description": "token=secret",
                    }
                ]
            },
        },
    )

    result = ws.perform_websearch("brave", "cake", "US", "en", "en", 10)

    rendered = "\n".join(logger.debug_messages)
    assert "api_key=secret" not in rendered
    assert "token=secret" not in rendered
    # Results themselves must pass through unredacted.
    rendered_results = "\n".join(
        str(item) for item in (result.get("results") or [])
    )
    assert "api_key=secret" in rendered_results
    assert "token=secret" in rendered_results


def test_duckduckgo_smoke_debug_logs_are_redacted(monkeypatch):
    logger = _FakeLazyLogger()
    monkeypatch.setattr(ws, "logging", logger)
    monkeypatch.setattr(
        ws,
        "search_web_duckduckgo",
        lambda *_args, **_kwargs: [
            {
                "title": "Duck",
                "href": "https://example.com/path?token=secret",
                "body": "api_key=secret",
            }
        ],
    )

    result = ws.perform_websearch("duckduckgo", "cake", "US", "en", "en", 10)

    rendered = "\n".join(logger.debug_messages)
    assert "token=secret" not in rendered
    assert "api_key=secret" not in rendered
    # Results themselves must pass through unredacted.
    rendered_results = "\n".join(
        str(item) for item in (result.get("results") or [])
    )
    assert "token=secret" in rendered_results
    assert "api_key=secret" in rendered_results


def test_google_parse_lazy_debug_logs_are_redacted(monkeypatch):
    """parse_google_results lazy-logs raw results; the lazy rendering must redact."""
    logger = _FakeLazyLogger()
    monkeypatch.setattr(ws, "logging", logger)
    monkeypatch.setattr(ws, "get_loaded_config", lambda: {
        "search_engines": {
            "google_search_api_url": "https://www.googleapis.com/customsearch/v1",
            "google_search_api_key": "test-key",
            "google_search_engine_id": "test-cx",
            "google_simp_trad_chinese": "1",
            "limit_google_search_to_country": False,
            "google_safe_search": "off",
        }
    })
    monkeypatch.setattr(ws, "_enforce_provider_outbound_policy", lambda *a, **k: None)

    def _fake_fetch_json(*, method, url, params, timeout):
        return {
            "items": [
                {
                    "title": "T",
                    "link": "https://example.com/path?api_key=secret",
                    "snippet": "token=secret",
                }
            ]
        }

    monkeypatch.setattr(http_client_module, "fetch_json", _fake_fetch_json)

    result = ws.perform_websearch("google", "cake", "US", "en", "en", 10)

    assert result.get("processing_error") is None
    assert logger.lazy_calls >= 1
    rendered = "\n".join(logger.debug_messages)
    assert "api_key=secret" not in rendered
    assert "token=secret" not in rendered
    assert "[REDACTED]" in rendered


@pytest.mark.parametrize(
    ("module", "parser_name", "payload", "expected_error"),
    [
        (ws, "parse_brave_results", _LeakyContains(), "Error processing Brave results"),
        (
            ws,
            "parse_duckduckgo_results",
            {"results": [{"title": "T", "href": "https://example.com/path", "body": "B"}]},
            "Error processing DuckDuckGo results",
        ),
        (ws, "parse_google_results", _LeakyContains(), "Error processing Google results"),
        (ws, "parse_kagi_results", _LeakyContains(), "Error processing Kagi results"),
        (
            ws,
            "parse_searx_results",
            {"results": [{"title": "T", "url": "https://example.com/path", "content": "B"}]},
            "Error processing Searx results",
        ),
        (
            ws,
            "parse_serper_results",
            {"organic": [{"title": "T", "link": "https://example.com/path", "snippet": "B"}]},
            "Error processing Serper results",
        ),
        (
            ws,
            "parse_tavily_results",
            {"results": [{"title": "T", "url": "https://example.com/path", "content": "B"}]},
            "Error processing Tavily results",
        ),
        (
            ws,
            "parse_exa_results",
            {"results": [{"title": "T", "url": "https://example.com/path", "text": "B"}]},
            "Error processing Exa results",
        ),
        (
            ws,
            "parse_firecrawl_results",
            {"data": [{"title": "T", "url": "https://example.com/path", "markdown": "B"}]},
            "Error processing Firecrawl results",
        ),
        (
            ws,
            "parse_4chan_results",
            {"results": [{"title": "T", "url": "https://example.com/path", "content": "B"}]},
            "Error processing 4chan results",
        ),
        (legacy_ws, "parse_bing_results", _LeakyContains(), "Error processing Bing results"),
        (legacy_ws, "parse_brave_results", _LeakyContains(), "Error processing Brave results"),
        (
            legacy_ws,
            "parse_duckduckgo_results",
            {"results": [{"title": "T", "href": "https://example.com/path", "body": "B"}]},
            "Error processing DuckDuckGo results",
        ),
        (legacy_ws, "parse_google_results", _LeakyContains(), "Error processing Google results"),
        (legacy_ws, "parse_kagi_results", _LeakyContains(), "Error processing Kagi results"),
    ],
)
def test_websearch_parsers_sanitize_processing_errors_and_logs(
    monkeypatch,
    module,
    parser_name,
    payload,
    expected_error,
):
    logger = _FakeLogger()
    monkeypatch.setattr(module, "logging", logger)

    if parser_name in {
        "parse_duckduckgo_results",
        "parse_searx_results",
        "parse_serper_results",
        "parse_tavily_results",
        "parse_exa_results",
        "parse_firecrawl_results",
        "parse_4chan_results",
    }:
        def fail_extract_domain(_url):
            raise TypeError(_LEAKY_ERROR)

        monkeypatch.setattr(module, "extract_domain", fail_extract_domain)

    output = {}

    getattr(module, parser_name)(payload, output)

    assert output["processing_error"] == expected_error
    _assert_safe_text(output["processing_error"])
    _assert_safe_text(logger.errors)


@pytest.mark.parametrize("module", [ws, legacy_ws])
def test_process_web_search_results_sanitizes_parser_failures(monkeypatch, module):
    logger = _FakeLogger()
    monkeypatch.setattr(module, "logging", logger)

    def fail_parser(*_args, **_kwargs):
        raise TypeError(_LEAKY_ERROR)

    monkeypatch.setattr(module, "parse_google_results", fail_parser)

    output = module.process_web_search_results({"items": []}, "google")

    assert output["processing_error"] == "Error processing search results"
    _assert_safe_text(output["processing_error"])
    _assert_safe_text(logger.errors)


@pytest.mark.parametrize("module", [ws, legacy_ws])
def test_process_web_search_results_preserves_invalid_engine_diagnostic(module):
    output = module.process_web_search_results({}, "not-a-provider")

    assert output["processing_error"] == "Error: Invalid Search Engine Name not-a-provider"
