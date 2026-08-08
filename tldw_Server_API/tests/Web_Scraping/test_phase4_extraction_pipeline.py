"""Parity tests for the canonical article extraction pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import trafilatura as direct_trafilatura

URL = "https://example.com/article"
HTML = "<html><head><title>Article</title></head><body>demo@example.com</body></html>"


def _result(
    *,
    success: bool,
    content: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "url": URL,
        "title": "Article",
        "author": "Author",
        "date": "2026-08-08",
        "content": content,
        "extraction_successful": success,
        **extra,
    }


def _install_default_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "extract_jsonld_entities", lambda *_args: _result(success=False))
    monkeypatch.setattr(pipeline, "extract_llm_entities", lambda *_args, **_kwargs: _result(success=False))
    monkeypatch.setattr(pipeline, "extract_cluster_entities", lambda *_args, **_kwargs: _result(success=False))


def test_default_regex_enriches_but_does_not_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_strategies(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "extract_regex_entities",
        lambda *_args, **_kwargs: _result(success=True, content="email", regex_matches=[{"label": "email"}]),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_with_trafilatura",
        lambda *_args: _result(success=True, content="article body"),
    )

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=None)

    assert result["extraction_successful"] is True
    assert result["extraction_strategy"] == "trafilatura"
    assert result["regex_matches"] == [{"label": "email"}]
    assert [entry["strategy"] for entry in result["extraction_trace"]] == [
        "jsonld",
        "schema",
        "regex",
        "llm",
        "cluster",
        "trafilatura",
    ]


def test_explicit_empty_order_preserves_terminal_regex_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_strategies(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "extract_regex_entities",
        lambda *_args, **_kwargs: _result(success=True, content="email", regex_matches=[{"label": "email"}]),
    )
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=True, content="article"))

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=[])

    assert result["extraction_strategy"] == "regex"
    assert [entry["strategy"] for entry in result["extraction_trace"]] == ["jsonld", "schema", "regex"]


@pytest.mark.parametrize(
    ("strategy_order", "expected_order", "expected_trace"),
    [
        (
            [" json-ld ", "microdata", "regex", "regex", "unknown", "trafilatura"],
            ["jsonld", "regex", "trafilatura"],
            ["unknown", "jsonld", "regex", "trafilatura"],
        ),
        (
            ["unknown-a", "unknown-b"],
            ["jsonld", "schema", "regex", "llm", "cluster", "trafilatura"],
            ["unknown-a", "unknown-b", "jsonld", "schema", "regex", "llm", "cluster", "trafilatura"],
        ),
    ],
)
def test_strategy_order_normalization_preserves_aliases_duplicates_and_unknown_traces(
    monkeypatch: pytest.MonkeyPatch,
    strategy_order: list[str],
    expected_order: list[str],
    expected_trace: list[str],
) -> None:
    _install_default_strategies(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: _result(success=False))
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=True, content="article"))

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=strategy_order)

    assert result["extraction_strategy_order"] == expected_order
    assert [entry["strategy"] for entry in result["extraction_trace"]] == expected_trace
    assert [entry for entry in result["extraction_trace"] if entry["reason"] == "unknown_strategy"] == [
        {"strategy": item, "status": "skipped", "reason": "unknown_strategy"}
        for item in [value.strip().lower() for value in strategy_order if value.startswith("unknown")]
    ]


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.sampled_from(["json-ld", "jsonld", "regex", "trafilatura", "unknown"]), max_size=12))
def test_strategy_order_normalization_is_deterministic_for_duplicates_and_aliases(
    monkeypatch: pytest.MonkeyPatch,
    strategy_order: list[str],
) -> None:
    _install_default_strategies(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: _result(success=False))
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=False))

    first = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=strategy_order)
    second = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=strategy_order)

    assert first["extraction_strategy_order"] == second["extraction_strategy_order"]
    assert first["extraction_trace"] == second["extraction_trace"]


def test_disabling_llm_filters_only_llm_from_explicit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_strategies(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=True, content="article"))

    result = pipeline.extract_article_with_pipeline(
        HTML,
        URL,
        strategy_order=["unknown", "llm", "trafilatura"],
        allow_llm_extraction=False,
    )

    assert result["extraction_strategy_order"] == ["trafilatura"]
    assert result["extraction_trace"] == [
        {"strategy": "unknown", "status": "skipped", "reason": "unknown_strategy"},
        {"strategy": "trafilatura", "status": "success", "reason": "extracted"},
    ]


def test_default_regex_matches_survive_later_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_strategies(monkeypatch)
    regex_result = _result(
        success=True, content="email", regex_matches=[{"label": "email", "value": "demo@example.com"}]
    )
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: regex_result)
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=False))

    result = pipeline.extract_article_with_pipeline(HTML, URL)

    assert result["extraction_successful"] is False
    assert result["regex_matches"] == regex_result["regex_matches"]
    assert result["extraction_strategy"] is None


def test_jsonld_summary_carries_forward_without_mutating_strategy_result(monkeypatch: pytest.MonkeyPatch) -> None:
    jsonld_result = _result(success=False, summary="JSON-LD summary")
    llm_result = _result(success=True, content="LLM body")
    monkeypatch.setattr(pipeline, "extract_jsonld_entities", lambda *_args: jsonld_result)
    monkeypatch.setattr(pipeline, "extract_llm_entities", lambda *_args, **_kwargs: llm_result)

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=["jsonld", "llm"])

    assert result["summary"] == "JSON-LD summary"
    assert jsonld_result == _result(success=False, summary="JSON-LD summary")
    assert llm_result == _result(success=True, content="LLM body")


def test_pipeline_copies_handler_and_cached_results_before_adding_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler_result = _result(success=True, content="handled", extraction_trace=[{"legacy": True}])
    cached_result = _result(success=True, content="cached")
    monkeypatch.setattr(pipeline, "_schema_cache_get", lambda _key: cached_result)

    cached = pipeline.extract_article_with_pipeline(
        HTML,
        URL,
        strategy_order=["schema"],
        schema_rules={"fields": []},
    )
    handled = pipeline.extract_article_with_pipeline(
        HTML,
        URL,
        strategy_order=["schema"],
        handler=lambda *_args: handler_result,
    )

    assert cached["schema_cache_hit"] is True
    assert "schema_cache_hit" not in cached_result
    assert handled["handler_trace"] == [{"legacy": True}]
    assert "handler_trace" not in handler_result
    assert handler_result["extraction_trace"] == [{"legacy": True}]


def test_canonical_pipeline_builds_dependencies_at_public_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    dependencies = dataclasses.replace(
        build_default_dependencies(), cancellation_checkpoint=lambda: calls.append("check")
    )
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: _result(success=False))

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=["regex"])

    assert result["extraction_successful"] is False
    assert calls
    assert legacy.extract_article_with_pipeline is pipeline.extract_article_with_pipeline


def test_pipeline_cancellation_stops_before_strategy_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def cancel() -> None:
        raise asyncio.CancelledError

    dependencies = dataclasses.replace(build_default_dependencies(), cancellation_checkpoint=cancel)
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: pytest.fail("dispatched"))

    with pytest.raises(asyncio.CancelledError):
        pipeline._extract_article_with_pipeline_with_dependencies(
            HTML,
            URL,
            dependencies=dependencies,
            strategy_order=["regex"],
        )


def test_direct_trafilatura_is_separate_from_enhanced_json_path() -> None:
    direct_source = inspect.getsource(direct_trafilatura.extract_with_trafilatura)
    enhanced_source = inspect.getsource(enhanced_web_scraping.EnhancedWebScraper._extract_trafilatura_json)

    assert "include_tables=False" in direct_source
    assert "output_format='json'" in enhanced_source
    assert "include_tables=True" in enhanced_source
