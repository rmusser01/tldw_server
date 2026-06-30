import pytest

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

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


@pytest.mark.parametrize(
    ("helper_name", "provider_label"),
    [
        ("test_perform_websearch_google", "google"),
        ("test_perform_websearch_brave", "brave"),
        ("test_perform_websearch_ddg", "duckduckgo"),
        ("test_perform_websearch_kagi", "kagi"),
        ("test_perform_websearch_serper", "serper"),
        ("test_perform_websearch_tavily", "tavily"),
        ("test_perform_websearch_searx", "searx"),
        ("test_perform_websearch_yandex", "yandex"),
    ],
)
def test_provider_search_helpers_sanitize_stdout(monkeypatch, capsys, helper_name, provider_label):
    def fail_search(*_args, **_kwargs):
        raise RuntimeError(_LEAKY_ERROR)

    monkeypatch.setattr(ws, "perform_websearch", fail_search)

    getattr(ws, helper_name)()

    output = capsys.readouterr().out
    assert f"Error performing {provider_label} searches" in output
    _assert_safe_text(output)


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
