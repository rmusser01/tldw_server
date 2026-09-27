"""Fail-closed regressions for rolling LLM summarization."""

from io import StringIO

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chunking import exceptions
from tldw_Server_API.app.core.Chunking.strategies import rolling_summarize

_TEXT = "A sentence to summarize."
_TRACKER_KEY = "_provider_usage_tracker"
_SUCCEEDED_KEY = "provider_succeeded"
_SECRET = "rolling-provider-secret-sentinel"


def _run(strategy: rolling_summarize.RollingSummarizeStrategy, method_name: str):
    return getattr(strategy, method_name)(_TEXT, max_size=1)


def _assert_detached(error: exceptions.ProcessingError) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


def test_usage_tracker_contract_is_stable() -> None:
    assert rolling_summarize.LLM_USAGE_TRACKER_KEY == _TRACKER_KEY
    assert rolling_summarize.LLM_USAGE_SUCCEEDED_KEY == _SUCCEEDED_KEY


@pytest.mark.parametrize("method_name", ["chunk", "chunk_with_metadata"])
def test_missing_llm_fails_closed_without_marking_usage(method_name: str) -> None:
    tracker: dict[str, bool] = {}
    strategy = rolling_summarize.RollingSummarizeStrategy(
        llm_config={_TRACKER_KEY: tracker},
    )

    with pytest.raises(exceptions.ProcessingError) as exc_info:
        _run(strategy, method_name)

    assert "unavailable" in str(exc_info.value).lower()
    assert tracker == {}
    _assert_detached(exc_info.value)


@pytest.mark.parametrize("method_name", ["chunk", "chunk_with_metadata"])
@pytest.mark.parametrize(
    "provider_result",
    [None, "", " \t ", f"Error: {_SECRET}", (f"Error: {_SECRET}", {})],
)
def test_invalid_provider_output_fails_closed_without_marking_usage(
    method_name: str,
    provider_result,
) -> None:
    tracker: dict[str, bool] = {}
    strategy = rolling_summarize.RollingSummarizeStrategy(
        llm_call_func=lambda *_args, **_kwargs: provider_result,
        llm_config={_TRACKER_KEY: tracker},
    )

    with pytest.raises(exceptions.ProcessingError) as exc_info:
        _run(strategy, method_name)

    assert _SECRET not in str(exc_info.value)
    assert tracker == {}
    _assert_detached(exc_info.value)


@pytest.mark.parametrize("method_name", ["chunk", "chunk_with_metadata"])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
def test_provider_failure_is_sanitized_and_does_not_mark_usage(
    method_name: str,
    error_type: type[Exception],
) -> None:
    tracker: dict[str, bool] = {}

    def failing_provider(*_args, **_kwargs):
        raise error_type(_SECRET)

    strategy = rolling_summarize.RollingSummarizeStrategy(
        llm_call_func=failing_provider,
        llm_config={_TRACKER_KEY: tracker},
    )
    captured_logs = StringIO()
    sink_id = logger.add(captured_logs, format="{message}")
    try:
        with pytest.raises(exceptions.ProcessingError) as exc_info:
            _run(strategy, method_name)
    finally:
        logger.remove(sink_id)

    assert _SECRET not in str(exc_info.value)
    assert _SECRET not in captured_logs.getvalue()
    assert tracker == {}
    _assert_detached(exc_info.value)


@pytest.mark.parametrize("method_name", ["chunk", "chunk_with_metadata"])
def test_verified_provider_output_marks_usage(method_name: str) -> None:
    tracker: dict[str, bool] = {}
    strategy = rolling_summarize.RollingSummarizeStrategy(
        llm_call_func=lambda *_args, **_kwargs: ("Verified summary", {"ignored": True}),
        llm_config={_TRACKER_KEY: tracker},
    )

    result = _run(strategy, method_name)

    output = result[0] if method_name == "chunk" else result[0].text
    assert output == "Verified summary"
    assert tracker == {_SUCCEEDED_KEY: True}


def test_runtime_handle_reaches_rolling_analyzer_without_entering_app_config() -> None:
    handle = object()
    app_config = {"local_llm": {"model": "snapshot-model"}}
    calls = []

    def analyzer(*args, **kwargs):
        calls.append((args, kwargs))
        return "Verified summary"

    strategy = rolling_summarize.RollingSummarizeStrategy(
        llm_call_func=analyzer,
        llm_config={
            "api_name": "local-llm",
            "api_key": "snapshot-key",
            "app_config": app_config,
            "credentials_resolved": True,
            "provider_credentials": handle,
        },
    )

    assert _run(strategy, "chunk")[0] == "Verified summary"
    assert calls[0][1]["provider_credentials"] is handle
    assert calls[0][1]["app_config"] is app_config
    assert "provider_credentials" not in app_config
