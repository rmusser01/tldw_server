"""Parity tests for the canonical article extraction pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping
from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
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


@pytest.mark.parametrize(
    ("strategy", "attribute"),
    [
        ("jsonld", "extract_jsonld_entities"),
        ("regex", "extract_regex_entities"),
        ("llm", "extract_llm_entities"),
        ("cluster", "extract_cluster_entities"),
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
    monkeypatch.setattr(pipeline, "extract_cluster_entities", lambda *_args, **_kwargs: failed_result)

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


def test_pipeline_cancellation_after_semaphore_wait_releases_permit_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = threading.Event()
    cancelled = threading.Event()
    semaphore = threading.BoundedSemaphore(1)
    assert semaphore.acquire(blocking=False)
    release_count = 0

    class SignalingSemaphore:
        def acquire(self) -> bool:
            waiting.set()
            return semaphore.acquire()

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
    monkeypatch.setattr(direct_trafilatura.logging, "info", lambda message: log_messages.append(str(message)))
    monkeypatch.setattr(direct_trafilatura.logging, "warning", lambda message: log_messages.append(str(message)))

    result = direct_trafilatura.extract_with_trafilatura(sensitive_html, sensitive_url)
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

    assert "output_format='json'" in enhanced_source
    assert "include_tables=True" in enhanced_source
