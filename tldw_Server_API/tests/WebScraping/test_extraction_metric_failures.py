import asyncio
import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.core.Metrics import metrics_logger
from tldw_Server_API.app.core.Web_Scraping.extraction import metrics
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm, trafilatura


class _MetricSinks:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        self.calls = 0

    def _raise_failure(self) -> None:
        self.calls += 1
        raise self.failure

    def increment_counter(self, *_args: Any, **_kwargs: Any) -> None:
        self._raise_failure()

    def observe_histogram(self, *_args: Any, **_kwargs: Any) -> None:
        self._raise_failure()

    def log_counter(self, *_args: Any, **_kwargs: Any) -> None:
        self._raise_failure()


class _MetricControlFlow(BaseException):
    pass


def _raising_callback(failure: BaseException) -> Callable[..., None]:
    def raise_failure(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    return raise_failure


def _emit(
    emitter: str,
    sinks: _MetricSinks,
    callback: Callable[..., None],
) -> None:
    if emitter == "counter":
        metrics.emit_counter(sinks, "article_extracted", labels={"success": "true"})
    elif emitter == "histogram":
        metrics.emit_histogram(
            sinks,
            "extraction_strategy_duration_seconds",
            0.1,
            labels={"strategy": "llm", "status": "success"},
        )
    elif emitter == "log_counter":
        metrics.emit_log_counter(sinks, "article_extracted", labels={"success": "true"})
    elif emitter == "callback_counter":
        metrics.emit_callback_counter(callback, "article_extracted", labels={"success": "true"})
    elif emitter == "global_counter":
        metrics.emit_global_counter("article_extracted", labels={"success": "true"})
    else:  # pragma: no cover - parametrization owns this inventory
        raise AssertionError(f"unknown emitter: {emitter}")


@pytest.mark.parametrize(
    "emitter",
    ["counter", "histogram", "log_counter", "callback_counter", "global_counter"],
)
def test_metric_emit_helpers_ignore_validation_errors(monkeypatch, emitter: str) -> None:
    calls = 0

    def record(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(metrics_logger, "log_counter", record)
    sinks = _MetricSinks(AssertionError("sink must not run"))

    if emitter == "counter":
        metrics.emit_counter(sinks, "uncontracted", labels={})
    elif emitter == "histogram":
        metrics.emit_histogram(sinks, "uncontracted", 0.1, labels={})
    elif emitter == "log_counter":
        metrics.emit_log_counter(sinks, "uncontracted", labels={})
    elif emitter == "callback_counter":
        metrics.emit_callback_counter(record, "uncontracted", labels={})
    else:
        metrics.emit_global_counter("uncontracted", labels={})

    assert sinks.calls == 0
    assert calls == 0


@pytest.mark.parametrize(
    "emitter",
    ["counter", "histogram", "log_counter", "callback_counter", "global_counter"],
)
def test_metric_emit_helpers_ignore_ordinary_sink_errors(monkeypatch, emitter: str) -> None:
    failure = RuntimeError("metric sink unavailable")
    monkeypatch.setattr(metrics_logger, "log_counter", _raising_callback(failure))

    _emit(emitter, _MetricSinks(failure), _raising_callback(failure))


@pytest.mark.parametrize(
    "emitter",
    ["counter", "histogram", "log_counter", "callback_counter", "global_counter"],
)
@pytest.mark.parametrize("failure_type", [asyncio.CancelledError, _MetricControlFlow])
def test_metric_emit_helpers_propagate_control_flow(
    monkeypatch,
    emitter: str,
    failure_type: type[BaseException],
) -> None:
    failure = failure_type()
    monkeypatch.setattr(metrics_logger, "log_counter", _raising_callback(failure))

    with pytest.raises(failure_type):
        _emit(emitter, _MetricSinks(failure), _raising_callback(failure))


def test_llm_retry_progresses_when_retry_metric_sink_fails(monkeypatch) -> None:
    attempts = 0
    response = {"choices": [{"message": {"content": '{"content": "Body"}'}}]}

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary provider failure")
        return response

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=provider,
        increment_counter=_raising_callback(RuntimeError("metric sink unavailable")),
    )
    monkeypatch.setattr(llm, "_retry_settings", lambda: (1, 0.0, 0.0))

    actual, failed = llm.call_llm_provider(
        provider="openai",
        settings={},
        messages=[],
        app_config=None,
        dependencies=dependencies,
        stage="extraction",
        url="https://example.com",
    )

    assert actual == response
    assert failed is False
    assert attempts == 2


def test_llm_success_and_usage_survive_metric_sink_failure(monkeypatch) -> None:
    def provider(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"title": "Hello", "content": "Body"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            "model": "gpt-test",
        }

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=provider,
        increment_counter=_raising_callback(RuntimeError("metric sink unavailable")),
    )
    monkeypatch.setattr(llm, "build_default_dependencies", lambda: dependencies)

    result = llm.extract_llm_entities(
        "<html><body><p>Body</p></body></html>",
        "https://example.com",
        llm_settings={"provider": "openai", "mode": "blocks"},
    )

    assert result["extraction_successful"] is True
    assert result["title"] == "Hello"
    assert result["content"] == "Body"
    assert result["llm_usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


@pytest.mark.parametrize(
    ("extracted", "expected_success"),
    [("article body", True), (None, False)],
)
def test_trafilatura_result_survives_global_metric_sink_failure(
    monkeypatch,
    extracted: str | None,
    expected_success: bool,
) -> None:
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda *_args, **_kwargs: extracted)
    monkeypatch.setattr(trafilatura.trafilatura, "extract_metadata", lambda _html: None)
    monkeypatch.setattr(
        metrics_logger,
        "log_counter",
        _raising_callback(RuntimeError("metric sink unavailable")),
    )

    result = trafilatura.extract_with_trafilatura("<html></html>", "https://example.com")

    assert result["extraction_successful"] is expected_success
    assert bool(result["content"]) is expected_success
