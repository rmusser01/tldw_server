"""Direct execution components exercise real output with a live fake context."""

from __future__ import annotations

import importlib
from builtins import ExceptionGroup
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.tests.Chunking._execution_test_support import FakeExecutionContext, RecordingCache

pytestmark = pytest.mark.unit
PACKAGE = "tldw_Server_API.app.core.Chunking.execution"


def make_service(context: Any, counters: list[Any] | None = None, logs: list[Any] | None = None) -> Any:
    models = importlib.import_module(f"{PACKAGE}.models")
    service = importlib.import_module(f"{PACKAGE}.service")
    counters = [] if counters is None else counters
    logs = [] if logs is None else logs
    logger = SimpleNamespace(
        **{
            name: lambda *args, _level=name, **kwargs: logs.append((_level, args, kwargs))
            for name in ("debug", "info", "warning", "error")
        }
    )
    hooks = models.ExecutionHooks(
        get_logger=lambda: logger, increment_counter=lambda *args, **kwargs: counters.append((args, kwargs))
    )
    return service.ChunkExecutionService(context, hooks)


def test_text_service_constructor_is_inert() -> None:
    class TrapContext(FakeExecutionContext):
        armed = False

        def __getattribute__(self, name: str) -> Any:
            if name != "armed" and object.__getattribute__(self, "armed"):
                raise RuntimeError(f"early read: {name}")
            return super().__getattribute__(name)

    context = TrapContext()
    context.armed = True
    service = make_service(context)
    context.armed = False
    assert context.events == []
    assert service.chunk_text("source") == ["strategy text"]


@pytest.mark.parametrize("case", ["hit", "miss", "get", "signature", "key", "put", "disabled", "empty"])
def test_text_service_cache_boundaries(case: str) -> None:
    context = FakeExecutionContext()
    cache = RecordingCache(["cached"] if case == "hit" else None)
    context._cache = None if case == "disabled" else cache
    counters: list[Any] = []
    service = make_service(context, counters)
    failure = ValueError("boundary sentinel")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    if case == "get":
        cache.get_error = failure
    elif case == "signature":
        context._get_llm_signature = fail
    elif case == "key":
        context._get_cache_key = fail
    elif case == "put":
        cache.put_error = failure
    elif case == "empty":
        context.strategy.text_result = []
    if case in {"get", "signature", "key"}:
        with pytest.raises(ValueError) as caught:
            service.chunk_text("source")
        assert caught.value is failure and context.strategy.calls == []
    else:
        result = service.chunk_text("source")
        assert result is (cache.value if case == "hit" else context.strategy.text_result)
        assert result == (["cached"] if case == "hit" else [] if case == "empty" else ["strategy text"])
        if case == "hit":
            assert context.strategy.calls == []
        elif case == "put":
            assert counters[-1] == (("chunker_cache_put_total",), {"labels": {"result": "error", "reason": "allowed"}})
        elif case in {"empty", "disabled"}:
            assert not any(event[0] == "put" for event in cache.events)
    assert context._strategy_lock.depth == 0


def test_counter_failures_do_not_replace_real_chunks() -> None:
    context = FakeExecutionContext()
    context._cache = RecordingCache()
    service = make_service(context)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("counter sentinel")

    models = importlib.import_module(f"{PACKAGE}.models")
    service.hooks = models.ExecutionHooks(get_logger=service.hooks.get_logger, increment_counter=fail)
    assert service.chunk_text("source") is context.strategy.text_result
    assert context._cache.value == ["strategy text"]


def test_grouped_counter_failure_retains_baseline_error() -> None:
    context = FakeExecutionContext()
    service = make_service(context)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ExceptionGroup("grouped metric failure", [ValueError("counter sentinel")])

    models = importlib.import_module(f"{PACKAGE}.models")
    service.hooks = models.ExecutionHooks(get_logger=service.hooks.get_logger, increment_counter=fail)
    with pytest.raises(TypeError):
        service.chunk_text("source")
    assert context.strategy.calls == []
