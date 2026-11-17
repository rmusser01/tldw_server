"""Caller-visible result budgets apply after provider normalization."""

from __future__ import annotations

import copy

import pytest

from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as search


@pytest.mark.parametrize("engine", ["searx", "GOOGLE"])
@pytest.mark.parametrize("requested", [1, 3, 10])
@pytest.mark.parametrize("returned", [0, 1, 3, 24])
def test_normalized_results_respect_each_requested_budget(
    monkeypatch: pytest.MonkeyPatch, engine: str, requested: int, returned: int
) -> None:
    rows = [
        {"title": f"Result {index}", "link": f"https://result.example/{index}", "snippet": f"Text {index}"}
        for index in range(returned)
    ]
    raw = (
        {"results": rows}
        if engine == "searx"
        else {
            "items": rows,
            "searchInformation": {"totalResults": str(returned), "searchTime": 0.1},
        }
    )
    original = copy.deepcopy(raw)
    monkeypatch.setattr(search, "search_web_searx", lambda *args, **kwargs: raw)
    monkeypatch.setattr(search, "search_web_google", lambda **kwargs: raw)
    monkeypatch.setattr(
        search,
        "get_loaded_config",
        lambda: {
            "search_engines": {"google_search_api_key": None, "google_search_engine_id": None},
        },
    )
    result = search.perform_websearch(
        search_engine=engine,
        search_query="controlled",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=requested,
    )
    count = min(requested, returned)
    # Contract assertion checks returned limits and preserved provider data.
    assert (  # nosec B101
        [row["url"] for row in result["results"]],
        result["result_count"],
        result["total_results_found"],
        result["search_engine"],
        raw,
    ) == (
        [f"https://result.example/{index}" for index in range(count)],
        count,
        returned,
        engine.lower(),
        original,
    )


def test_subqueries_keep_individual_budgets_and_report_the_combined_returned_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search, "analyze_question", lambda *args: {"sub_questions": ["follow"]})
    monkeypatch.setattr(search.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        search,
        "search_web_searx",
        lambda query, **kwargs: {
            "results": [
                {"title": f"{query} {index}", "link": f"https://result.example/{query}/{index}", "snippet": "Text"}
                for index in range(3)
            ]
        },
    )
    result = search.generate_and_search(
        "main",
        {
            "engine": "searx",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 2,
            "subquery_generation": True,
        },
    )["web_search_results_dict"]
    # Contract assertion checks returned limits and preserved provider data.
    assert (  # nosec B101
        [row["url"] for row in result["results"]],
        result["result_count"],
        result["total_results_found"],
    ) == (
        [f"https://result.example/{query}/{index}" for query in ["main", "follow"] for index in range(2)],
        4,
        6,
    )


@pytest.mark.parametrize("engine", ["searx", "google"])
def test_result_budget_preserves_provider_errors(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    raw = {"error": "provider unavailable"}
    monkeypatch.setattr(search, "search_web_searx", lambda *args, **kwargs: raw)
    monkeypatch.setattr(search, "search_web_google", lambda **kwargs: raw)
    monkeypatch.setattr(
        search,
        "get_loaded_config",
        lambda: {
            "search_engines": {"google_search_api_key": None, "google_search_engine_id": None},
        },
    )
    result = search.perform_websearch(
        search_engine=engine,
        search_query="controlled",
        content_country="US",
        search_lang="en",
        output_lang="en",
        result_count=3,
    )
    # Provider failure must survive normalization unchanged.
    assert (result["results"], result["result_count"], result["error"]) == ([], 0, "provider unavailable")  # nosec B101
