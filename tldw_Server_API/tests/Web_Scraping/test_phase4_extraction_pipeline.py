"""Parity tests for the canonical article extraction pipeline."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import hashlib
import inspect
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping
from tldw_Server_API.app.core.Web_Scraping.browser_transport import decide_browser_transport
from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import cluster as cluster_strategy
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm as llm_strategy
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import trafilatura as direct_trafilatura

URL = "https://example.com/article"
HTML = "<html><head><title>Article</title></head><body>demo@example.com</body></html>"


@pytest.mark.unit
async def test_legacy_recursive_scrape_denies_transport_before_playwright_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Legacy recursive crawling must fail closed before starting Playwright."""
    denied = decide_browser_transport(
        configured_mode="disabled",
        auth_mode="single_user",
        outbound_policy_mode="compat",
    )
    launcher = Mock(side_effect=AssertionError("Playwright must not start"))
    monkeypatch.setattr(legacy, "default_browser_transport_decision", lambda: denied, raising=False)
    monkeypatch.setattr(legacy, "async_playwright", launcher)

    results = await legacy.recursive_scrape(
        URL,
        max_pages=1,
        max_depth=0,
        delay=0,
        resume_file=str(tmp_path / "progress.json"),
    )

    assert results == [
        {
            "url": URL,
            "error": "browser_transport_unavailable",
            "extraction_successful": False,
            "capability": denied.to_capability_metadata(),
        }
    ]
    launcher.assert_not_called()


@pytest.mark.unit
async def test_legacy_article_scrape_denies_transport_before_page_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied browser context must not bypass transport policy."""
    denied = decide_browser_transport(
        configured_mode="auto",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
    )
    context = SimpleNamespace(
        new_page=AsyncMock(side_effect=AssertionError("page must not be created"))
    )
    monkeypatch.setattr(legacy, "default_browser_transport_decision", lambda: denied, raising=False)

    result = await legacy.scrape_article_async(context, URL)

    assert result == {
        "url": URL,
        "error": "browser_transport_unavailable",
        "extraction_successful": False,
        "capability": denied.to_capability_metadata(),
    }
    context.new_page.assert_not_awaited()


@pytest.mark.unit
async def test_legacy_recursive_scrape_stops_when_transport_is_denied_mid_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Do not discover links after an inner article dispatch loses admission."""
    allowed = decide_browser_transport(
        configured_mode="auto",
        auth_mode="single_user",
        outbound_policy_mode="compat",
    )
    denied = decide_browser_transport(
        configured_mode="auto",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
    )
    decision_provider = Mock(side_effect=[allowed, denied])
    context = SimpleNamespace(
        new_page=AsyncMock(side_effect=AssertionError("link page must not be created")),
        add_cookies=AsyncMock(return_value=None),
    )
    browser = SimpleNamespace(
        new_context=AsyncMock(return_value=context),
        close=AsyncMock(return_value=None),
    )
    playwright = SimpleNamespace(
        chromium=SimpleNamespace(launch=AsyncMock(return_value=browser))
    )
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=playwright)
    manager.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(legacy, "default_browser_transport_decision", decision_provider)
    monkeypatch.setattr(legacy, "async_playwright", Mock(return_value=manager))

    results = await legacy.recursive_scrape(
        URL,
        max_pages=1,
        max_depth=1,
        delay=0,
        resume_file=str(tmp_path / "progress.json"),
    )

    assert results == [
        {
            "url": URL,
            "error": "browser_transport_unavailable",
            "extraction_successful": False,
            "capability": denied.to_capability_metadata(),
        }
    ]
    context.new_page.assert_not_awaited()
    assert decision_provider.call_count == 2


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
    monkeypatch.setattr(
        pipeline,
        "_extract_llm_entities_with_dependencies",
        lambda *_args, **_kwargs: _result(success=False),
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_cluster_entities_with_dependencies",
        lambda *_args, **_kwargs: _result(success=False),
    )


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
    assert result["extraction_trace"] == [
        {"strategy": "jsonld", "status": "failed", "reason": "jsonld_no_content"},
        {
            "strategy": "schema",
            "status": "skipped",
            "reason": "no_schema_rules_or_handler",
        },
        {"strategy": "regex", "status": "enriched", "reason": "regex_enriched"},
        {"strategy": "llm", "status": "failed", "reason": "llm_no_content"},
        {"strategy": "cluster", "status": "failed", "reason": "cluster_no_content"},
        {"strategy": "trafilatura", "status": "success", "reason": "extracted"},
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
    assert result["extraction_trace"] == [
        {"strategy": "jsonld", "status": "failed", "reason": "jsonld_no_content"},
        {
            "strategy": "schema",
            "status": "skipped",
            "reason": "no_schema_rules_or_handler",
        },
        {"strategy": "regex", "status": "success", "reason": "regex_extracted"},
    ]


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
@given(
    st.lists(
        st.sampled_from(
            [
                "json-ld",
                " JSON_LD ",
                "microdata",
                "schema_css",
                "schema_xpath",
                "clustering",
                "regex",
                "REGEX",
                "llm",
                "cluster",
                "trafilatura",
                "unknown-a",
                "unknown-b",
                " ",
            ]
        ),
        max_size=20,
    )
)
def test_strategy_order_normalization_preserves_public_invariants(
    monkeypatch: pytest.MonkeyPatch,
    strategy_order: list[str],
) -> None:
    _install_default_strategies(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: _result(success=False))
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=False))

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=strategy_order)

    aliases = {
        "json-ld": "jsonld",
        "json_ld": "jsonld",
        "microdata": "jsonld",
        "schema_css": "schema",
        "schema_xpath": "schema",
        "clustering": "cluster",
    }
    known = set(pipeline.DEFAULT_EXTRACTION_STRATEGY_ORDER)
    expected_order: list[str] = []
    expected_unknown: list[str] = []
    for value in strategy_order:
        normalized = aliases.get(value.strip().lower(), value.strip().lower())
        if not normalized:
            continue
        if normalized in known:
            if normalized not in expected_order:
                expected_order.append(normalized)
        else:
            expected_unknown.append(normalized)
    if not expected_order:
        expected_order = list(pipeline.DEFAULT_EXTRACTION_STRATEGY_ORDER)

    reason_by_strategy = {
        "jsonld": ("failed", "jsonld_no_content"),
        "schema": ("skipped", "no_schema_rules_or_handler"),
        "regex": ("failed", "regex_no_matches"),
        "llm": ("failed", "llm_no_content"),
        "cluster": ("failed", "cluster_no_content"),
        "trafilatura": ("failed", "no_content"),
    }
    expected_trace = [
        {"strategy": strategy, "status": "skipped", "reason": "unknown_strategy"} for strategy in expected_unknown
    ]
    expected_trace.extend(
        {
            "strategy": strategy,
            "status": reason_by_strategy[strategy][0],
            "reason": reason_by_strategy[strategy][1],
        }
        for strategy in expected_order
    )

    assert result["extraction_strategy_order"] == expected_order
    assert result["extraction_trace"] == expected_trace


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


def test_default_regex_result_survives_later_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_strategies(monkeypatch)
    regex_result = _result(
        success=True, content="email", regex_matches=[{"label": "email", "value": "demo@example.com"}]
    )
    monkeypatch.setattr(pipeline, "extract_regex_entities", lambda *_args, **_kwargs: regex_result)
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", lambda *_args: _result(success=False))

    result = pipeline.extract_article_with_pipeline(HTML, URL)

    assert result["extraction_successful"] is True
    assert result["content"] == "email"
    assert result["regex_matches"] == regex_result["regex_matches"]
    assert result["extraction_strategy"] == "regex"


def test_jsonld_summary_carries_forward_without_mutating_strategy_result(monkeypatch: pytest.MonkeyPatch) -> None:
    jsonld_result = _result(success=False, summary="JSON-LD summary")
    llm_result = _result(success=True, content="LLM body")
    monkeypatch.setattr(pipeline, "extract_jsonld_entities", lambda *_args: jsonld_result)
    monkeypatch.setattr(
        pipeline,
        "_extract_llm_entities_with_dependencies",
        lambda *_args, **_kwargs: llm_result,
    )

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


@pytest.mark.parametrize(
    ("strategy", "attribute"),
    [
        ("jsonld", "extract_jsonld_entities"),
        ("regex", "extract_regex_entities"),
        ("llm", "_extract_llm_entities_with_dependencies"),
        ("cluster", "_extract_cluster_entities_with_dependencies"),
    ],
)
def test_pipeline_deep_copies_direct_strategy_results(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    attribute: str,
) -> None:
    strategy_result = _result(
        success=True,
        content="strategy content",
        nested={"values": ["provider-owned"]},
    )
    monkeypatch.setattr(pipeline, attribute, lambda *_args, **_kwargs: strategy_result)

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=[strategy])
    result["nested"]["values"].append("caller-mutation")
    result["extraction_trace"][0]["reason"] = "caller-mutation"

    assert strategy_result["nested"] == {"values": ["provider-owned"]}
    assert "extraction_trace" not in strategy_result


def test_pipeline_deep_copies_fallback_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_result = _result(
        success=True,
        content="fallback content",
        nested={"values": ["fallback-owned"]},
    )

    result = pipeline.extract_article_with_pipeline(
        HTML,
        URL,
        strategy_order=["trafilatura"],
        fallback_extractor=lambda *_args: fallback_result,
    )
    result["nested"]["values"].append("caller-mutation")
    result["extraction_trace"][0]["reason"] = "caller-mutation"

    assert fallback_result["nested"] == {"values": ["fallback-owned"]}
    assert "extraction_trace" not in fallback_result


def test_pipeline_deep_copies_final_failure_result(monkeypatch: pytest.MonkeyPatch) -> None:
    failed_result = _result(
        success=False,
        nested={"values": ["provider-owned"]},
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_cluster_entities_with_dependencies",
        lambda *_args, **_kwargs: failed_result,
    )

    result = pipeline.extract_article_with_pipeline(HTML, URL, strategy_order=["cluster"])
    result["nested"]["values"].append("caller-mutation")
    result["extraction_trace"][0]["reason"] = "caller-mutation"

    assert failed_result["nested"] == {"values": ["provider-owned"]}
    assert "extraction_trace" not in failed_result


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


def test_pipeline_llm_uses_injected_provider_and_observes_cancellation_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False
    events: list[str] = []
    defaults = build_default_dependencies()

    def checkpoint() -> None:
        if cancelled:
            raise asyncio.CancelledError

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal cancelled
        events.append("provider")
        cancelled = True
        return {
            "choices": [{"message": {"content": '{"content": "injected"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "injected-model",
        }

    dependencies = dataclasses.replace(
        defaults,
        perform_chat_api_call=provider,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        log_counter=lambda *_args, **_kwargs: None,
        cancellation_checkpoint=checkpoint,
    )
    nested_defaults = dataclasses.replace(
        defaults,
        perform_chat_api_call=lambda **_kwargs: events.append("nested-provider")
        or {
            "choices": [{"message": {"content": '{"content": "nested"}'}}],
            "usage": {},
        },
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        log_counter=lambda *_args, **_kwargs: None,
    )

    def build_nested_defaults():
        events.append("nested-defaults")
        return nested_defaults

    monkeypatch.setattr(llm_strategy, "build_default_dependencies", build_nested_defaults)
    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: None)

    with pytest.raises(asyncio.CancelledError):
        pipeline._extract_article_with_pipeline_with_dependencies(
            "<p>" + " ".join(f"word-{index}" for index in range(80)) + "</p>",
            URL,
            dependencies=dependencies,
            strategy_order=["llm", "trafilatura"],
            llm_settings={
                "provider": "openai",
                "chunk_token_threshold": 50,
                "word_token_rate": 1.0,
                "overlap_rate": 0.0,
            },
            fallback_extractor=lambda *_args: events.append("fallback") or _result(success=True),
        )

    assert events == ["provider"]


def test_pipeline_llm_reuses_injected_clocks_sleeps_and_metric_sinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls = 0
    perf_calls = 0
    wall_time_calls = 0
    sleeps: list[float] = []
    counter_names: list[str] = []
    histogram_names: list[str] = []
    log_counter_names: list[str] = []
    defaults = build_default_dependencies()

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        return {
            "choices": [{"message": {"content": '{"content": "injected"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "injected-model",
        }

    def perf_counter() -> float:
        nonlocal perf_calls
        perf_calls += 1
        return float(perf_calls)

    def wall_time() -> float:
        nonlocal wall_time_calls
        wall_time_calls += 1
        return 1000.0

    dependencies = dataclasses.replace(
        defaults,
        perform_chat_api_call=provider,
        increment_counter=lambda name, *_args, **_kwargs: counter_names.append(name),
        observe_histogram=lambda name, *_args, **_kwargs: histogram_names.append(name),
        log_counter=lambda name, *_args, **_kwargs: log_counter_names.append(name),
        perf_counter=perf_counter,
        wall_time=wall_time,
        sleep=sleeps.append,
    )
    monkeypatch.setattr(
        llm_strategy,
        "build_default_dependencies",
        lambda: pytest.fail("pipeline rebuilt nested LLM dependencies"),
    )
    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: None)
    html = "<p>" + " ".join(f"word-{index}" for index in range(80)) + "</p>"
    llm_strategy.throttles.clear_throttle_state()
    try:
        result = pipeline._extract_article_with_pipeline_with_dependencies(
            html,
            URL,
            dependencies=dependencies,
            strategy_order=["llm"],
            llm_settings={
                "provider": "openai",
                "chunk_token_threshold": 50,
                "word_token_rate": 1.0,
                "overlap_rate": 0.0,
                "delay_ms": 10,
            },
        )
    finally:
        llm_strategy.throttles.clear_throttle_state()

    assert result["extraction_strategy"] == "llm"
    assert result["content"] == "injected"
    assert provider_calls == 2
    assert perf_calls == 2
    assert wall_time_calls == 2
    assert sleeps == pytest.approx([0.01])
    assert set(counter_names) == {
        "llm_tokens_used_total",
        "llm_tokens_used_total_by_operation",
    }
    assert histogram_names == [
        "extraction_strategy_duration_seconds",
        "extraction_content_length_bytes",
    ]
    assert log_counter_names == ["extraction_strategy_total"]


def test_pipeline_cluster_observes_injected_cancellation_before_successful_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False
    events: list[str] = []
    defaults = build_default_dependencies()

    def checkpoint() -> None:
        if cancelled:
            raise asyncio.CancelledError

    dependencies = dataclasses.replace(
        defaults,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        log_counter=lambda *_args, **_kwargs: None,
        cancellation_checkpoint=checkpoint,
    )
    nested_defaults = dataclasses.replace(
        defaults,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        log_counter=lambda *_args, **_kwargs: None,
    )

    def build_nested_defaults():
        events.append("nested-defaults")
        return nested_defaults

    def cancel_during_clustering(
        items: list[tuple[int, str, list[float], float]],
        *,
        cluster_threshold: float,
    ) -> list[dict[str, Any]]:
        nonlocal cancelled
        del cluster_threshold
        events.append("cluster-execution")
        cancelled = True
        index, block, vector, similarity = items[0]
        return [
            {
                "members": [(index, block, similarity)],
                "sum_vec": list(vector),
                "centroid": list(vector),
                "total_chars": len(block),
            }
        ]

    monkeypatch.setattr(cluster_strategy, "build_default_dependencies", build_nested_defaults)
    monkeypatch.setattr(cluster_strategy, "_cluster_blocks_greedy", cancel_during_clustering)
    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: None)
    html = """
    <html><head><title>Cluster</title></head><body>
    <p>Primary article block with enough words for deterministic cluster extraction.</p>
    <p>Secondary article block with enough words for deterministic cluster extraction.</p>
    </body></html>
    """

    with pytest.raises(asyncio.CancelledError):
        pipeline._extract_article_with_pipeline_with_dependencies(
            html,
            URL,
            dependencies=dependencies,
            strategy_order=["cluster", "trafilatura"],
            cluster_settings={
                "min_block_chars": 1,
                "min_word_count": 1,
                "prefilter_threshold": 0.0,
            },
            fallback_extractor=lambda *_args: events.append("fallback") or _result(success=True),
        )

    assert events == ["cluster-execution"]


def test_pipeline_cancellation_after_semaphore_wait_releases_permit_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = threading.Event()
    cancelled = threading.Event()
    semaphore = threading.BoundedSemaphore(1)
    assert semaphore.acquire(blocking=False)
    release_count = 0

    class SignalingSemaphore:
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            waiting.set()
            return semaphore.acquire(blocking=blocking, timeout=timeout)

        def release(self) -> None:
            nonlocal release_count
            release_count += 1
            semaphore.release()

    def checkpoint() -> None:
        if cancelled.is_set():
            raise asyncio.CancelledError

    dispatches: list[str] = []
    dependencies = dataclasses.replace(build_default_dependencies(), cancellation_checkpoint=checkpoint)
    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: SignalingSemaphore())
    monkeypatch.setattr(
        pipeline,
        "extract_regex_entities",
        lambda *_args, **_kwargs: dispatches.append("regex") or _result(success=False),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_with_trafilatura",
        lambda *_args: dispatches.append("trafilatura") or _result(success=False),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            pipeline._extract_article_with_pipeline_with_dependencies,
            HTML,
            URL,
            dependencies=dependencies,
            strategy_order=["regex", "trafilatura"],
        )
        assert waiting.wait(timeout=1.0)
        cancelled.set()
        semaphore.release()
        with pytest.raises(asyncio.CancelledError):
            future.result(timeout=1.0)

    assert release_count == 1
    assert semaphore.acquire(blocking=False)
    semaphore.release()
    assert dispatches == []


def test_pipeline_cancellation_after_retry_sleep_prevents_retry_and_next_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False
    dispatches: list[object] = []

    def checkpoint() -> None:
        if cancelled:
            raise asyncio.CancelledError

    def cancel_during_sleep(delay: float) -> None:
        nonlocal cancelled
        dispatches.append(("sleep", delay))
        cancelled = True

    def handler(*_args: object) -> dict[str, Any]:
        dispatches.append("handler")
        raise RuntimeError("retryable")

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        cancellation_checkpoint=checkpoint,
        sleep=cancel_during_sleep,
    )
    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "2")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "1")
    monkeypatch.setenv("EXTRACTOR_RETRY_JITTER_MS", "0")
    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: None)

    with pytest.raises(asyncio.CancelledError):
        pipeline._extract_article_with_pipeline_with_dependencies(
            HTML,
            URL,
            dependencies=dependencies,
            strategy_order=["schema", "trafilatura"],
            handler=handler,
            fallback_extractor=lambda *_args: dispatches.append("trafilatura") or _result(success=False),
        )

    assert dispatches == ["handler", ("sleep", 0.001)]


def test_direct_trafilatura_executes_metadata_and_sanitized_observability_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_url = "https://user:secret@example.com/article?token=private"
    sensitive_html = "<html><body>provider-payload-secret</body></html>"
    extract_calls: list[tuple[str, dict[str, object]]] = []
    metric_events: list[tuple[str, dict[str, str]]] = []
    log_messages: list[str] = []

    def extract(html: str, **kwargs: object) -> str:
        extract_calls.append((html, kwargs))
        return "Article body"

    monkeypatch.setattr(direct_trafilatura.trafilatura, "extract", extract)
    monkeypatch.setattr(
        direct_trafilatura.trafilatura,
        "extract_metadata",
        lambda html: (
            SimpleNamespace(title="Article", author="Ada", date="2026-08-08")
            if html == sensitive_html
            else pytest.fail("unexpected HTML")
        ),
    )
    monkeypatch.setattr(
        direct_trafilatura,
        "log_counter",
        lambda name, labels: metric_events.append((name, dict(labels))),
    )
    handler_id = logger.add(
        lambda message: log_messages.append(message.record["message"]),
        filter=lambda record: record["name"] == direct_trafilatura.__name__,
    )
    try:
        result = direct_trafilatura.extract_with_trafilatura(sensitive_html, sensitive_url)
    finally:
        logger.remove(handler_id)
    metadata, body = ContentMetadataHandler.extract_metadata(result["content"])

    assert extract_calls == [
        (
            sensitive_html,
            {
                "include_comments": False,
                "include_tables": False,
                "include_images": False,
            },
        )
    ]
    assert result == {
        "title": "Article",
        "author": "Ada",
        "content": result["content"],
        "date": "2026-08-08",
        "url": sensitive_url,
        "extraction_successful": True,
    }
    assert body == "Article body"
    assert metadata["url"] == sensitive_url
    assert set(metadata) == {
        "url",
        "ingestion_date",
        "content_hash",
        "scraping_pipeline",
        "extracted_date",
        "author",
    }
    assert metadata["content_hash"] == hashlib.sha256(b"Article body").hexdigest()
    assert metadata["scraping_pipeline"] == "Trafilatura"
    assert metadata["extracted_date"] == "2026-08-08"
    assert metadata["author"] == "Ada"
    assert metric_events == [("article_extracted", {"success": "true"})]
    assert log_messages == ["Extracting article data from HTML", "Content extracted successfully"]
    assert all(sensitive_url not in message and "provider-payload-secret" not in message for message in log_messages)


def test_direct_trafilatura_is_separate_from_enhanced_json_path() -> None:
    enhanced_source = inspect.getsource(enhanced_web_scraping.EnhancedWebScraper._extract_trafilatura_json)
    tree = ast.parse(textwrap.dedent(enhanced_source))
    extract_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "trafilatura"
        and node.func.attr == "extract"
    ]
    assert len(extract_calls) == 1
    keywords = {
        keyword.arg: keyword.value.value
        for keyword in extract_calls[0].keywords
        if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
    }

    assert keywords["output_format"] == "json"
    assert keywords["include_tables"] is True
