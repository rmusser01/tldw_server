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

    monkeypatch.setattr(web_search._DELAY_RANDOM, "uniform", lambda *_args: 0)
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


def test_search_web_google_redacts_api_key_from_legacy_logs(monkeypatch):
    secret_key = "secret-google-key"
    messages: list[str] = []

    class DummyResponse:
        def json(self):
            return {"items": []}

    class DummyLogger:
        def info(self, message, *args, **kwargs):
            messages.append(str(message))

        def error(self, message, *args, **kwargs):
            messages.append(str(message))

        def debug(self, message, *args, **kwargs):
            messages.append(str(message))

        def warning(self, message, *args, **kwargs):
            messages.append(str(message))

    monkeypatch.setattr(web_search, "logger", DummyLogger())
    monkeypatch.setattr(web_search, "fetch", lambda **_kwargs: DummyResponse())
    monkeypatch.setattr(
        web_search,
        "loaded_config_data",
        {
            "search_engines": {
                "google_search_api_url": "https://customsearch.googleapis.com/customsearch/v1",
                "google_simp_trad_chinese": "1",
                "limit_google_search_to_country": False,
                "google_search_country": "countryUS",
                "google_search_engine_id": "engine-id",
                "google_search_api_key": secret_key,
                "google_safe_search": "off",
            }
        },
    )

    web_search.search_web_google("query")

    logged_text = "\n".join(messages)
    assert secret_key not in logged_text
    assert "'key': '[REDACTED]'" in logged_text


def test_generate_and_search_marks_all_legacy_provider_failures(monkeypatch):
    def fail_provider(**_kwargs):
        return {"processing_error": "Error performing web search", "results": []}

    monkeypatch.setattr(web_search, "perform_websearch", fail_provider)

    result = web_search.generate_and_search(
        "query",
        {
            "engine": "google",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 1,
        },
    )

    web_results = result["web_search_results_dict"]
    assert web_results["results"] == []
    assert web_results["processing_error"] == "Error performing web search"
    assert web_results["warnings"] == [
        {
            "phase": "provider",
            "query": "query",
            "message": "Error performing web search",
        }
    ]


async def test_legacy_user_review_fails_fast_without_input(monkeypatch):
    async def fake_relevance(*_args, **_kwargs):
        return {"0": {"content": "summary", "reasoning": "matches"}}

    monkeypatch.setattr(web_search, "search_result_relevance", fake_relevance)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("input() should not be called")),
    )

    with pytest.raises(ValueError, match="Interactive user_review is not supported"):
        await web_search.analyze_and_aggregate(
            {"results": [{"id": "0", "url": "https://example.com", "content": "snippet"}]},
            {"main_goal": "query", "sub_questions": []},
            {"user_review": True, "relevance_analysis_llm": "fake", "final_answer_llm": None},
        )


async def test_analyze_and_aggregate_omits_raw_original_content_from_response(monkeypatch):
    raw_content = "raw scraped article text" * 100

    async def fake_relevance(*_args, **_kwargs):
        return {
            "0": {
                "content": "summary",
                "original_content": raw_content,
                "reasoning": "matches",
            }
        }

    monkeypatch.setattr(web_search, "search_result_relevance", fake_relevance)

    result = await web_search.analyze_and_aggregate(
        {"results": [{"id": "0", "url": "https://example.com", "content": "snippet"}]},
        {"main_goal": "query", "sub_questions": []},
        {"user_review": False, "relevance_analysis_llm": "fake", "final_answer_llm": None},
    )

    relevant_result = result["relevant_results"]["0"]
    assert "original_content" not in relevant_result
    assert relevant_result["original_content_chars"] == len(raw_content)
    assert result["final_answer"]["evidence"][0]["content"] == "summary"
    assert "original_content" not in result["final_answer"]["evidence"][0]


async def test_analyze_and_aggregate_can_opt_into_raw_original_content(monkeypatch):
    raw_content = "raw scraped article text"

    async def fake_relevance(*_args, **_kwargs):
        return {
            "0": {
                "content": "summary",
                "original_content": raw_content,
                "reasoning": "matches",
            }
        }

    monkeypatch.setattr(web_search, "search_result_relevance", fake_relevance)

    result = await web_search.analyze_and_aggregate(
        {"results": [{"id": "0", "url": "https://example.com", "content": "snippet"}]},
        {"main_goal": "query", "sub_questions": []},
        {
            "user_review": False,
            "relevance_analysis_llm": "fake",
            "final_answer_llm": None,
            "include_original_content": True,
        },
    )

    assert result["relevant_results"]["0"]["original_content"] == raw_content


def test_legacy_aggregate_evidence_omits_raw_original_content():
    result = web_search.aggregate_results(
        relevant_results={
            "0": {
                "content": "summary",
                "original_content": "raw scraped article text" * 100,
                "reasoning": "matches",
            }
        },
        question="query",
        sub_questions=[],
        api_endpoint=None,
    )

    evidence = result["evidence"][0]
    assert "original_content" not in evidence
    assert evidence["content"] == "summary"


@pytest.mark.parametrize("engine", ["baidu", "serper", "tavily", "searx", "yandex"])
def test_legacy_perform_websearch_reports_unimplemented_provider(engine):
    result = web_search.perform_websearch(
        search_engine=engine,
        search_query="query",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=1,
    )

    assert result["processing_error"] == f"Legacy WebSearch provider '{engine}' is not implemented"
    assert result["results"] == []


def test_search_web_duckduckgo_rejects_empty_keywords_with_value_error():
    with pytest.raises(ValueError, match="keywords is mandatory"):
        web_search.search_web_duckduckgo("")
