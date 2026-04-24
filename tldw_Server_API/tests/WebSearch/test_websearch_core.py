from __future__ import annotations

from typing import Any, Dict
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as web_search


def test_aggregate_results_returns_structured_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat_api_call(**_: Any) -> str:
        return "Aggregated answer text."

    def fake_summarize(**_: Any) -> str:
        return "Chunk summary"

    monkeypatch.setattr(web_search, "summarize", fake_summarize)
    monkeypatch.setattr(web_search, "chat_api_call", fake_chat_api_call)

    relevant_results = {
        "0": {"content": "Summary content", "reasoning": "Contains the answer"}
    }

    result = web_search.aggregate_results(
        relevant_results=relevant_results,
        question="What is the capital of France?",
        sub_questions=[],
        api_endpoint="fake-llm",
    )

    assert result["text"] == "Aggregated answer text."
    assert result["confidence"] == pytest.approx(0.495, abs=1e-3)
    assert result["evidence"][0]["id"] == "0"
    assert result["chunks"]
    assert result["chunks"][0]["chunk_index"] == 1
    assert result["chunks"][0]["generated"] is True


def test_aggregate_results_handles_large_context(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_calls = {"count": 0}

    def fake_summarize(input_data: str, **_: Any) -> str:
        chunk_calls["count"] += 1
        return f"chunk-summary-{chunk_calls['count']}"

    def fake_chat_api_call(**_: Any) -> str:
        return "Final aggregated answer."

    monkeypatch.setattr(web_search, "summarize", fake_summarize)
    monkeypatch.setattr(web_search, "chat_api_call", fake_chat_api_call)

    large_snippet = "Paris is the capital of France. " * 300  # ~8400 characters
    relevant_results = {
        "0": {"content": large_snippet, "reasoning": "Contains capital info"},
        "1": {"content": large_snippet, "reasoning": "Additional confirmation"},
    }

    result = web_search.aggregate_results(
        relevant_results=relevant_results,
        question="What is the capital of France?",
        sub_questions=[],
        api_endpoint="fake-llm",
    )

    assert result["text"] == "Final aggregated answer."
    assert chunk_calls["count"] >= 2  # Expect at least two chunk summaries
    assert len(result["chunks"]) >= 2
    assert all("summary" in chunk for chunk in result["chunks"])
    assert result["confidence"] > 0.5


@pytest.mark.asyncio
async def test_search_result_relevance_filters_irrelevant_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, int] = {"relevance": 0}

    async def fake_scrape_article(url: str) -> Dict[str, Any]:
        return {"content": f"Full article for {url}"}

    def fake_summarize(**_: Any) -> str:
        return "Summarized content"

    responses = iter([
        "Selected Answer: True\nReasoning: Contains the requested information.",
        "Selected Answer: False\nReasoning: Off topic.",
    ])

    def fake_chat_api_call(**_: Any) -> str:
        calls["relevance"] += 1
        return next(responses)

    async def instant_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(web_search, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(web_search, "summarize", fake_summarize)
    monkeypatch.setattr(web_search, "chat_api_call", fake_chat_api_call)
    monkeypatch.setattr(web_search.asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(web_search, "get_loaded_config", lambda: {})

    search_results = [
        {"id": "keep", "url": "https://example.com/keep", "content": "Snippet keep"},
        {"id": "drop", "url": "https://example.com/drop", "content": "Snippet drop"},
    ]

    relevant = await web_search.search_result_relevance(
        search_results=search_results,
        original_question="What is the capital of France?",
        sub_questions=["capital of France"],
        api_endpoint="fake-llm",
    )

    assert "keep" in relevant
    assert relevant["keep"]["content"] == "Summarized content"
    assert "drop" not in relevant
    assert calls["relevance"] == 2


def test_search_web_brave_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response_payload = {"web": {"results": []}}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:

            return fake_response_payload

    def fake_get(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> DummyResponse:
        fake_get.last_request = {"url": url, "headers": headers, "params": params}  # type: ignore[attr-defined]
        return DummyResponse()

    # Patch the Brave wrapper seam instead of requests.get
    monkeypatch.setattr(web_search, "brave_http_get", fake_get)
    from tldw_Server_API.app.core.Security import egress as egress_module
    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url: SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {
            "search_engines": {
                "brave_search_ai_api_key": "ai-key",
                "brave_search_api_key": "web-key",
                "search_engine_country_code_brave": "GB",
                "search_result_max_per_query": 7,
            }
        },
    )

    result = web_search.search_web_brave(
        search_term="capital of france",
        country="FR",
        search_lang="fr",
        ui_lang="en",
        result_count=5,
        safesearch="active",
        date_range="w",
        site_blacklist=["example.com", "test.com"],
    )

    assert result == fake_response_payload

    captured = fake_get.last_request  # type: ignore[attr-defined]
    assert captured["headers"]["X-Subscription-Token"] == "ai-key"
    assert captured["params"]["source"] == "ai"
    assert captured["params"]["safeSearch"] == "Active"
    assert captured["params"]["country"] == "FR"
    assert captured["params"]["search_lang"] == "fr"
    assert captured["params"]["ui_lang"] == "en"
    assert captured["params"]["count"] == 5
    assert captured["params"]["freshness"] == "w"
    assert captured["params"]["exclude_sites"] == "example.com,test.com"


def test_search_web_brave_blocks_with_shared_policy_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
        WebOutboundPolicyDecision,
    )

    monkeypatch.setattr(
        web_search,
        "decide_web_outbound_policy_sync",
        lambda *args, **kwargs: WebOutboundPolicyDecision(
            allowed=False,
            mode="strict",
            reason="deny_test",
            stage=kwargs.get("stage", "provider_request"),
            source=kwargs.get("source", "websearch_brave"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        web_search,
        "brave_http_get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider HTTP call should not run when outbound policy blocks")
        ),
    )
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {"search_engines": {"brave_search_ai_api_key": "ai-key"}},
    )

    with pytest.raises(ValueError, match="Blocked by outbound policy: deny_test"):
        web_search.search_web_brave(
            search_term="capital of france",
            country="US",
            search_lang="en",
            ui_lang="en",
            result_count=5,
        )


def test_perform_websearch_duckduckgo_surfaces_policy_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
        WebOutboundPolicyDecision,
    )

    monkeypatch.setattr(
        web_search,
        "decide_web_outbound_policy_sync",
        lambda *args, **kwargs: WebOutboundPolicyDecision(
            allowed=False,
            mode="strict",
            reason="deny_test",
            stage=kwargs.get("stage", "provider_request"),
            source=kwargs.get("source", "websearch_duckduckgo_html"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        web_search,
        "fetch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider fetch should not run when outbound policy blocks")
        ),
    )

    result = web_search.perform_websearch(
        "duckduckgo",
        "capital of france",
        "US",
        "en",
        "en",
        5,
    )

    assert result["processing_error"] == "Error performing web search: Blocked by outbound policy: deny_test"


def test_generate_and_search_propagates_searx_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_search_web_searx(search_query: str, **kwargs: Any) -> Dict[str, Any]:
        captured["search_query"] = search_query
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setattr(web_search, "search_web_searx", fake_search_web_searx)

    result = web_search.generate_and_search(
        "test query",
        {
            "engine": "searx",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 1,
            "searx_url": "https://custom.searx.local",
            "searx_json_mode": True,
        },
    )

    assert "web_search_results_dict" in result
    assert captured.get("search_query") == "test query"
    assert captured.get("searx_url") == "https://custom.searx.local"
    assert captured.get("json_mode") is True


def test_analyze_question_accepts_json_array_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search,
        "_call_adapter_text",
        lambda **_kwargs: '["alpha", " beta ", "", "ALPHA"]',
    )
    monkeypatch.setattr(web_search, "get_loaded_config", lambda: {})

    result = web_search.analyze_question("main query", "openai")

    assert result["sub_questions"] == ["alpha", "beta"]
    assert result["search_queries"] == ["alpha", "beta"]


def test_analyze_question_accepts_json_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search,
        "_call_adapter_text",
        lambda **_kwargs: '{"sub_questions": ["one", "two"]}',
    )
    monkeypatch.setattr(web_search, "get_loaded_config", lambda: {})

    result = web_search.analyze_question("main query", "openai")

    assert result["sub_questions"] == ["one", "two"]
    assert result["search_queries"] == ["one", "two"]


def test_analyze_question_empty_response_falls_back_to_no_subquestions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_search, "_call_adapter_text", lambda **_kwargs: "{}")
    monkeypatch.setattr(web_search, "get_loaded_config", lambda: {})

    result = web_search.analyze_question("main query", "openai")

    assert result["sub_questions"] == []
    assert result["search_queries"] == []


def test_generate_and_search_dedupes_and_sanitizes_subqueries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_queries: list[str] = []

    def fake_analyze_question(question: str, _api_endpoint: str) -> Dict[str, Any]:
        return {
            "main_goal": question,
            "sub_questions": [
                "Main Query",
                " follow up ",
                "",
                None,
                {"query": "follow up"},
                "Deep dive",
            ],
            "search_queries": [],
            "analysis_prompt": None,
        }

    def fake_perform_websearch(**kwargs: Any) -> Dict[str, Any]:
        captured_queries.append(str(kwargs.get("search_query")))
        return {"results": [], "total_results_found": 0, "search_time": 0.0}

    monkeypatch.setattr(web_search, "analyze_question", fake_analyze_question)
    monkeypatch.setattr(web_search, "perform_websearch", fake_perform_websearch)
    monkeypatch.setattr(web_search.time, "sleep", lambda _seconds: None)

    result = web_search.generate_and_search(
        "Main Query",
        {
            "engine": "google",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 5,
            "subquery_generation": True,
            "subquery_generation_llm": "openai",
        },
    )

    assert captured_queries == ["Main Query", "follow up", "Deep dive"]
    assert result["sub_query_dict"]["sub_questions"] == ["follow up", "Deep dive"]
    assert result["sub_query_dict"]["search_queries"] == ["follow up", "Deep dive"]


def test_generate_and_search_surfaces_processing_errors_when_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search,
        "perform_websearch",
        lambda **_kwargs: {"processing_error": "provider failed"},
    )
    monkeypatch.setattr(web_search.time, "sleep", lambda _seconds: None)

    result = web_search.generate_and_search(
        "Main Query",
        {
            "engine": "google",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 5,
            "subquery_generation": False,
        },
    )

    web_results = result["web_search_results_dict"]
    assert web_results["results"] == []
    assert web_results.get("error") == "provider failed"
    assert web_results.get("warnings")


def test_perform_websearch_tavily_forwards_site_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_search_web_tavily(search_query: str, result_count: int = 10, site_whitelist=None, site_blacklist=None) -> Dict[str, Any]:
        captured["search_query"] = search_query
        captured["result_count"] = result_count
        captured["site_whitelist"] = site_whitelist
        captured["site_blacklist"] = site_blacklist
        return {"results": []}

    monkeypatch.setattr(web_search, "search_web_tavily", fake_search_web_tavily)

    result = web_search.perform_websearch(
        search_engine="tavily",
        search_query="test query",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=3,
        site_whitelist=["allowed.example"],
        site_blacklist=["blocked.example"],
    )

    assert captured.get("search_query") == "test query"
    assert captured.get("result_count") == 3
    assert captured.get("site_whitelist") == ["allowed.example"]
    assert captured.get("site_blacklist") == ["blocked.example"]
    assert result.get("search_engine") == "tavily"


def test_perform_websearch_google_forwards_google_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_search_web_google(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(web_search, "search_web_google", fake_search_web_google)
    monkeypatch.setattr(web_search, "process_web_search_results", lambda *_args, **_kwargs: {"search_engine": "google"})
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {
            "search_engines": {
                "google_search_api_key": "key",
                "google_search_engine_id": "engine-id",
            }
        },
    )

    result = web_search.perform_websearch(
        search_engine="google",
        search_query="capital of france",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=3,
        google_domain="google.de",
    )

    assert captured.get("google_domain") == "google.de"
    assert result.get("search_engine") == "google"


def test_perform_websearch_google_multi_blacklist_does_not_set_site_search(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_search_web_google(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(web_search, "search_web_google", fake_search_web_google)
    monkeypatch.setattr(web_search, "process_web_search_results", lambda *_args, **_kwargs: {"search_engine": "google"})
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {
            "search_engines": {
                "google_search_api_key": "key",
                "google_search_engine_id": "engine-id",
            }
        },
    )

    result = web_search.perform_websearch(
        search_engine="google",
        search_query="capital of france",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=3,
        site_blacklist="foo.example, bar.example",
    )

    assert captured.get("site_blacklist") == "foo.example, bar.example"
    assert "siteSearch" not in captured
    assert "siteSearchFilter" not in captured
    assert result.get("search_engine") == "google"


def test_perform_websearch_searx_maps_safesearch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_search_web_searx(search_query: str, **kwargs: Any) -> Dict[str, Any]:
        captured["search_query"] = search_query
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setattr(web_search, "search_web_searx", fake_search_web_searx)
    monkeypatch.setattr(web_search, "process_web_search_results", lambda *_args, **_kwargs: {"search_engine": "searx"})

    result = web_search.perform_websearch(
        search_engine="searx",
        search_query="capital of france",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=3,
        safesearch="strict",
    )

    assert captured.get("safesearch") == 2
    assert result.get("search_engine") == "searx"


def test_perform_websearch_serper_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_search_web_serper(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"organic": []}

    monkeypatch.setattr(web_search, "search_web_serper", fake_search_web_serper)

    result = web_search.perform_websearch(
        search_engine="serper",
        search_query="test query",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=5,
        date_range="w",
        safesearch="active",
        site_whitelist=["allowed.example"],
        site_blacklist=["blocked.example"],
        exactTerms="exact phrase",
        excludeTerms="omit this",
    )

    assert captured.get("search_query") == "test query"
    assert captured.get("result_count") == 5
    assert captured.get("content_country") == "US"
    assert captured.get("search_lang") == "en"
    assert captured.get("output_lang") == "en"
    assert captured.get("date_range") == "w"
    assert captured.get("safesearch") == "active"
    assert captured.get("site_whitelist") == ["allowed.example"]
    assert captured.get("site_blacklist") == ["blocked.example"]
    assert captured.get("exactTerms") == "exact phrase"
    assert captured.get("excludeTerms") == "omit this"
    assert result.get("search_engine") == "serper"


def test_perform_websearch_invalid_engine_returns_structured_error() -> None:
    result = web_search.perform_websearch(
        search_engine="notarealengine",
        search_query="test query",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=1,
    )

    assert isinstance(result, dict)
    assert "processing_error" in result
    assert "Invalid Search Engine Name notarealengine" in str(result["processing_error"])


def test_search_web_serper_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_fetch_json(*, method: str, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return {"organic": []}

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(http_client, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(egress_module, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {
            "search_engines": {
                "serper_search_api_key": "serper-key",
                "serper_search_api_url": "https://google.serper.dev/search",
            }
        },
    )

    result = web_search.search_web_serper(
        search_query="capital of france",
        result_count=7,
        content_country="FR",
        search_lang="fr",
        output_lang="fr",
        date_range="w",
        safesearch="active",
        site_whitelist=["allowed.example"],
        site_blacklist=["blocked.example"],
        exactTerms="exact phrase",
        excludeTerms="omit this",
    )

    assert result == {"organic": []}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://google.serper.dev/search"
    assert captured["headers"]["X-API-KEY"] == "serper-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["timeout"] == 20.0

    payload = captured["json"]
    assert payload["num"] == 7
    assert payload["gl"] == "fr"
    assert payload["hl"] == "fr"
    assert payload["safe"] == "active"
    assert payload["tbs"] == "qdr:w"
    assert "site:allowed.example" in payload["q"]
    assert "-site:blocked.example" in payload["q"]
    assert "\"exact phrase\"" in payload["q"]
    assert " -omit " in payload["q"]
    assert " -this" in payload["q"]


def test_search_web_google_forwards_google_domain_to_googlehost(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_fetch_json(*, method: str, url: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured["method"] = method
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return {"items": []}

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(http_client, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(egress_module, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {"search_engines": {"google_search_api_url": "https://example.com/customsearch/v1"}},
    )

    result = web_search.search_web_google(
        search_query="capital of france",
        google_search_api_key="google-key",
        google_search_engine_id="engine-id",
        c2coff="1",
        results_origin_country="countryUS",
        safesearch="off",
        google_domain="google.de",
    )

    assert result == {"items": []}
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.com/customsearch/v1"
    assert captured["timeout"] == 15.0
    assert captured["params"]["googlehost"] == "google.de"


def test_search_web_google_applies_multi_domain_blacklist_to_query(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_fetch_json(*, method: str, url: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured["method"] = method
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return {"items": []}

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(http_client, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(egress_module, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {"search_engines": {"google_search_api_url": "https://example.com/customsearch/v1"}},
    )

    result = web_search.search_web_google(
        search_query="capital of france",
        google_search_api_key="google-key",
        google_search_engine_id="engine-id",
        c2coff="1",
        results_origin_country="countryUS",
        safesearch="off",
        site_blacklist="foo.example, bar.example",
    )

    assert result == {"items": []}
    assert "-site:foo.example" in captured["params"]["q"]
    assert "-site:bar.example" in captured["params"]["q"]


def test_search_web_kagi_uses_correct_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_fetch_json(*, method: str, url: str, headers: Dict[str, Any], params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return {"data": []}

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(http_client, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(egress_module, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {"search_engines": {"kagi_search_api_key": "kagi-key"}},
    )

    result = web_search.search_web_kagi(query="capital of france", limit=4)

    assert result == {"data": []}
    assert captured["method"] == "GET"
    assert captured["url"] == "https://kagi.com/api/v0/search"
    assert "/search/search" not in captured["url"]
    assert captured["params"]["q"] == "capital of france"
    assert captured["params"]["limit"] == 4


def test_search_web_serper_uses_env_key_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_fetch_json(*, method: str, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return {"organic": []}

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(http_client, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(egress_module, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
    monkeypatch.setattr(
        web_search,
        "get_loaded_config",
        lambda: {
            "search_engines": {
                "serper_search_api_url": "https://google.serper.dev/search",
            }
        },
    )
    monkeypatch.delenv("SEARCH_ENGINE_API_KEY_SERPER", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "env-serper-key")

    result = web_search.search_web_serper(search_query="capital of france")

    assert result == {"organic": []}
    assert captured["method"] == "POST"
    assert captured["headers"]["X-API-KEY"] == "env-serper-key"
    assert captured["url"] == "https://google.serper.dev/search"


def test_generate_and_search_propagates_include_domains_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_perform_websearch(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"results": [], "total_results_found": 0, "search_time": 0.0}

    monkeypatch.setattr(web_search, "perform_websearch", fake_perform_websearch)

    result = web_search.generate_and_search(
        "test query",
        {
            "engine": "google",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 1,
            "include_domains": ["allowed.example"],
        },
    )

    assert "web_search_results_dict" in result
    assert captured.get("site_whitelist") == ["allowed.example"]
    assert result["web_search_results_dict"].get("site_whitelist") == ["allowed.example"]


def test_generate_and_search_propagates_provider_warnings_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_perform_websearch(**_: Any) -> Dict[str, Any]:
        return {
            "results": [],
            "total_results_found": 0,
            "search_time": 0.0,
            "warnings": [{"board": "g", "phase": "catalog", "message": "timeout"}],
            "error": "4chan search failed for all requested boards.",
            "processing_error": None,
        }

    monkeypatch.setattr(web_search, "perform_websearch", fake_perform_websearch)

    result = web_search.generate_and_search(
        "rust memory safety",
        {
            "engine": "4chan",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 5,
        },
    )

    payload = result["web_search_results_dict"]
    assert payload["results"] == []
    assert payload.get("error") == "4chan search failed for all requested boards."
    assert isinstance(payload.get("warnings"), list)
    assert any(
        warning.get("board") == "g" and warning.get("phase") == "catalog"
        for warning in payload["warnings"]
        if isinstance(warning, dict)
    )


def test_generate_and_search_surfaces_provider_error_as_warning_when_results_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze_question(_question: str, _api_endpoint: str) -> Dict[str, Any]:
        return {
            "main_goal": "rust memory safety",
            "sub_questions": ["rust ownership basics"],
            "search_queries": [],
            "analysis_prompt": None,
        }

    def fake_sleep(_seconds: float) -> None:
        return None

    def fake_perform_websearch(**kwargs: Any) -> Dict[str, Any]:
        search_query = kwargs.get("search_query")
        if search_query == "rust memory safety":
            return {
                "results": [],
                "total_results_found": 0,
                "search_time": 0.01,
                "error": "4chan search failed for all requested boards.",
                "processing_error": None,
            }
        if search_query == "rust ownership basics":
            return {
                "results": [
                    {
                        "title": "Rust ownership overview",
                        "url": "https://boards.4chan.org/g/thread/123",
                        "content": "Ownership explains borrowing and moves.",
                    }
                ],
                "total_results_found": 1,
                "search_time": 0.02,
                "processing_error": None,
            }
        raise AssertionError(f"Unexpected query: {search_query}")

    monkeypatch.setattr(web_search, "analyze_question", fake_analyze_question)
    monkeypatch.setattr(web_search, "perform_websearch", fake_perform_websearch)
    monkeypatch.setattr(web_search.time, "sleep", fake_sleep)

    result = web_search.generate_and_search(
        "rust memory safety",
        {
            "engine": "4chan",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 5,
            "subquery_generation": True,
            "subquery_generation_llm": "openai",
        },
    )

    payload = result["web_search_results_dict"]
    assert len(payload["results"]) == 1
    assert payload.get("error") is None
    assert isinstance(payload.get("warnings"), list)
    assert any(
        warning.get("phase") == "provider"
        and warning.get("query") == "rust memory safety"
        and warning.get("message") == "4chan search failed for all requested boards."
        for warning in payload["warnings"]
        if isinstance(warning, dict)
    )


@pytest.mark.asyncio
async def test_search_result_relevance_uses_fallback_content_when_scrape_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    async def failing_scrape_article(_url: str) -> Dict[str, Any]:
        raise RuntimeError("blocked")

    def fake_summarize(input_data: str, **_: Any) -> str:
        captured["input_data"] = input_data
        return f"summary::{input_data}"

    def fake_chat_api_call(**_: Any) -> str:
        return "Selected Answer: True\nReasoning: Snippet is relevant."

    monkeypatch.setattr(web_search, "scrape_article", failing_scrape_article)
    monkeypatch.setattr(web_search, "summarize", fake_summarize)
    monkeypatch.setattr(web_search, "chat_api_call", fake_chat_api_call)
    monkeypatch.setattr(web_search, "get_loaded_config", lambda: {})

    relevant = await web_search.search_result_relevance(
        search_results=[
            {
                "id": "keep",
                "title": "Paris title",
                "url": "https://example.com/keep",
                "content": "",
                "metadata": {"snippet": "Paris is the capital of France."},
            }
        ],
        original_question="What is the capital of France?",
        sub_questions=["capital of France"],
        api_endpoint="fake-llm",
    )

    assert "keep" in relevant
    assert "Paris title" in captured["input_data"]
    assert "Paris is the capital of France." in captured["input_data"]
    assert relevant["keep"]["content"].startswith("summary::")
    assert "Paris is the capital of France." in relevant["keep"]["original_content"]


def test_search_web_duckduckgo_raises_on_egress_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=False, reason="blocked"),
    )

    with pytest.raises(ValueError, match="Blocked by outbound policy: blocked"):
        web_search.search_web_duckduckgo("capital of france")


def test_relevant_results_log_summary_redacts_original_content() -> None:
    summary = web_search._summarize_relevant_results_for_log(
        {
            "keep": {
                "content": "short summary",
                "original_content": "very secret original body",
                "reasoning": "Contains the answer",
            }
        }
    )

    serialized = str(summary)
    assert "very secret original body" not in serialized
    assert "short summary" not in serialized
    assert summary[0]["id"] == "keep"
    assert summary[0]["content_chars"] == len("short summary")
    assert summary[0]["original_content_chars"] == len("very secret original body")


@pytest.mark.asyncio
async def test_search_discussions_uses_preprocessed_results_without_reprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {"to_thread_calls": 0}

    def fake_perform_websearch(**kwargs: Any) -> Dict[str, Any]:
        captured["search_query"] = kwargs.get("search_query")
        return {
            "results": [
                {
                    "title": "Discussion thread",
                    "url": "https://reddit.com/r/test/comments/1",
                    "content": "Community feedback",
                }
            ]
        }

    def fail_process_results(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("process_web_search_results should not be called for normalized payloads")

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        captured["to_thread_calls"] += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(web_search, "perform_websearch", fake_perform_websearch)
    monkeypatch.setattr(web_search, "process_web_search_results", fail_process_results)
    monkeypatch.setattr(web_search.asyncio, "to_thread", fake_to_thread)

    docs = await web_search.search_discussions(
        query="rag feedback",
        platforms=["reddit"],
        max_results=1,
    )

    assert captured["to_thread_calls"] == 1
    assert "site:reddit.com" in str(captured.get("search_query", ""))
    assert len(docs) == 1
    assert docs[0]["source"] == "discussion"
    assert docs[0]["platform"] == "reddit"
