"""WebSearch provider tests.

Unit tests below verify the Google CSE request formatting that the old inline
`test_perform_websearch_google` FIXME flagged ("Fails. Need to fix arg
formatting"): every emitted parameter must be a documented Google Custom Search
parameter, `cr` must be in `countryXX` form, and unsupported parameters must
not be sent.

The external_api-marked smoke cases replace the manual smoke scripts that used
to live inside tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

import tldw_Server_API.app.core.http_client as http_client_module
from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as wsa

# Documented Google Custom Search JSON API query parameters.
GOOGLE_CSE_ALLOWED_PARAMS = {
    "q", "cx", "key", "num", "start", "c2coff", "cr", "dateRestrict",
    "exactTerms", "excludeTerms", "filter", "gl", "hl", "lr", "safe",
    "googlehost", "siteSearch", "siteSearchFilter",
}


class _FakeConfig(dict):
    pass


def _config_with_google() -> _FakeConfig:
    return _FakeConfig(
        search_engines={
            "google_search_api_url": "https://www.googleapis.com/customsearch/v1",
            "google_search_api_key": "test-key",
            "google_search_engine_id": "test-cx",
            "google_simp_trad_chinese": "1",
            "limit_google_search_to_country": False,
            "google_safe_search": "off",
        }
    )


@pytest.fixture()
def captured_google_params(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_fetch_json(*, method: str, url: str, params: dict[str, Any], timeout: float):
        captured.update(params)
        return {"items": [{"title": "t", "link": "https://example.com", "snippet": "s"}]}

    monkeypatch.setattr(wsa, "get_loaded_config", lambda: _config_with_google())
    monkeypatch.setattr(wsa, "_enforce_provider_outbound_policy", lambda *a, **k: None)
    monkeypatch.setattr(http_client_module, "fetch_json", _fake_fetch_json)
    return captured


@pytest.mark.unit
def test_google_advanced_args_use_only_documented_cse_params(captured_google_params):
    result = wsa.perform_websearch(
        "google",
        "What is the capital of France?",
        "US",
        "en",
        "en",
        10,
        date_range="y",
        safesearch="active",
        site_blacklist=["spam-site.com"],
    )

    assert result.get("processing_error") is None, result
    unknown = set(captured_google_params) - GOOGLE_CSE_ALLOWED_PARAMS
    assert not unknown, f"Google CSE does not accept: {sorted(unknown)}"


@pytest.mark.unit
def test_google_country_param_is_normalized_to_countryxx(captured_google_params):
    wsa.perform_websearch(
        "google", "What is the capital of France?", "US", "en", "en", 10
    )

    assert captured_google_params["cr"] == "countryUS"


@pytest.mark.unit
def test_google_country_param_passthrough_when_already_formatted(captured_google_params):
    wsa.perform_websearch(
        "google", "What is the capital of France?", "countryFR", "fr", "fr", 5
    )

    assert captured_google_params["cr"] == "countryFR"


@pytest.mark.unit
def test_google_sort_request_is_not_sent_as_unknown_param(captured_google_params):
    wsa.perform_websearch(
        "google", "query", "US", "en", "en", 10, sort_results_by="date"
    )

    assert "sort" not in captured_google_params


@pytest.mark.external_api
@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_API_TESTS", "0") != "1",
    reason="External API tests disabled. Set RUN_EXTERNAL_API_TESTS=1 to enable.",
)
@pytest.mark.parametrize(
    "engine,kwargs",
    [
        ("google", {}),
        ("google", {"date_range": "y", "safesearch": "active", "site_blacklist": ["spam-site.com"]}),
        ("duckduckgo", {}),
        ("duckduckgo", {"date_range": "y"}),
        ("brave", {}),
        ("kagi", {}),
        ("serper", {}),
        ("tavily", {}),
        ("searx", {}),
        ("yandex", {}),
    ],
)
def test_perform_websearch_provider_smoke(engine: str, kwargs: dict):
    """Manual smoke: run with -m external_api and configured provider keys."""
    result = wsa.perform_websearch(engine, "What is the capital of France?", "US", "en", "en", 10, **kwargs)
    assert result.get("processing_error") is None, result
    assert result.get("results") or result.get("items"), result
