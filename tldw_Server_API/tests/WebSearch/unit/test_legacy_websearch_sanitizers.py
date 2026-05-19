import json

import pytest

from tldw_Server_API.app.core.WebSearch import Web_Search as web_search


pytestmark = pytest.mark.unit


def test_perform_websearch_sanitizes_legacy_provider_failures(monkeypatch):
    def fail_search(**_kwargs):
        raise RuntimeError("legacy provider token at /private/legacy-provider")

    monkeypatch.setattr(web_search, "search_web_bing", fail_search)

    result = web_search.perform_websearch(
        search_engine="bing",
        search_query="query",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=1,
    )

    assert result["processing_error"] == "Error performing web search"
    assert "legacy provider token" not in result["processing_error"]
    assert "/private/legacy-provider" not in result["processing_error"]


def test_search_web_searx_sanitizes_invalid_legacy_url_configuration(monkeypatch):
    def fail_urlparse(_url):
        raise ValueError("invalid legacy searx config at /private/searx.conf")

    monkeypatch.setattr(web_search, "urlparse", fail_urlparse)

    payload = web_search.search_web_searx("query", searx_url="https://searx.example")
    result = json.loads(payload)

    assert result["error"] == "Invalid URL configuration."
    assert "invalid legacy searx config" not in result["error"]
    assert "/private/searx.conf" not in result["error"]


def test_search_web_searx_sanitizes_legacy_fetch_errors(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("legacy searx token at /private/searx.key")

    monkeypatch.setattr(web_search.random, "uniform", lambda *_args: 0)
    monkeypatch.setattr(web_search.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(web_search, "fetch", fail_fetch)

    payload = web_search.search_web_searx("query", searx_url="https://searx.example")
    result = json.loads(payload)

    assert result["error"] == "There was an error searching for content."
    assert "legacy searx token" not in result["error"]
    assert "/private/searx.key" not in result["error"]


def test_search_web_tavily_sanitizes_legacy_fetch_errors(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("legacy tavily token at /private/tavily.key")

    monkeypatch.setattr(
        web_search,
        "loaded_config_data",
        {"search_engines": {"tavily_search_api_key": "test-key"}},
    )
    monkeypatch.setattr(web_search, "fetch", fail_fetch)

    result = web_search.search_web_tavily("query")

    assert result == "There was an error searching for content."
    assert "legacy tavily token" not in result
    assert "/private/tavily.key" not in result
