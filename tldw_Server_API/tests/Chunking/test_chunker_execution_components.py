"""Direct execution components exercise real output with a live fake context."""

from __future__ import annotations

import importlib
from builtins import ExceptionGroup
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chunking.base import ChunkMetadata, ChunkResult
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


@pytest.mark.parametrize("alignment", [None, False, True])
@pytest.mark.parametrize("result_kind", ["list", "tuple", "missing", "raising"])
def test_metadata_service_keeps_alignment_regions(alignment: Any, result_kind: str) -> None:
    context = FakeExecutionContext()
    cache = RecordingCache()
    context._cache = cache
    metadata = ChunkMetadata(0, -2, 99, 1)
    item: Any = ChunkResult("normalized", metadata)
    if result_kind == "missing":
        item = SimpleNamespace(text="normalized")
    elif result_kind == "raising":

        class RaisingItem:
            text = "normalized"

            @property
            def metadata(self) -> Any:
                raise ValueError("metadata sentinel")

        item = RaisingItem()
    original = (item,) if result_kind == "tuple" else [item]
    context.strategy.metadata_result = original
    service = make_service(context)
    result = service.chunk_text_with_metadata("source", align_text_to_source=alignment)
    assert result is original and result[0] is item
    assert item.text == ("source" if alignment and result_kind == "list" else "normalized")
    if result_kind in {"list", "tuple"}:
        assert item.metadata is metadata and (metadata.start_char, metadata.end_char) == (-2, 99)
    assert cache.events == [] and context._strategy_lock.depth == 0
    assert service.chunk_text_with_metadata("  ") == []


def test_metadata_alignment_mutates_after_lock_release() -> None:
    context = FakeExecutionContext()
    mutations = []

    class Item:
        metadata = ChunkMetadata(0, True, 4, 1)
        _text = "normalized"

        @property
        def text(self) -> str:
            return self._text

        @text.setter
        def text(self, value: str) -> None:
            mutations.append((value, context._strategy_lock.depth))
            self._text = value

    item = Item()
    context.strategy.metadata_result = [item]
    result = make_service(context).chunk_text_with_metadata("source", align_text_to_source=True)
    assert result[0] is item and item.text == "our"
    assert mutations == [("our", 0)]


def test_metadata_success_logging_stays_outside_translation() -> None:
    context = FakeExecutionContext()
    service = make_service(context)
    failure = ValueError("success logging sentinel")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    models = importlib.import_module(f"{PACKAGE}.models")
    service.hooks = models.ExecutionHooks(get_logger=lambda: SimpleNamespace(debug=fail), increment_counter=fail)
    with pytest.raises(ValueError) as caught:
        service.chunk_text_with_metadata("source")
    assert caught.value is failure
    assert context.strategy.calls[0][0] == "metadata"


def test_generator_delegates_send_throw_close() -> None:
    context = FakeExecutionContext()
    operations = []

    class Iterator:
        def __iter__(self) -> Any:
            return self

        def __next__(self) -> str:
            operations.append(("next", context._strategy_lock.depth))
            return "first"

        def send(self, value: Any) -> str:
            operations.append(("send", value, context._strategy_lock.depth))
            return "sent"

        def throw(self, error: Any) -> str:
            operations.append(("throw", str(error), context._strategy_lock.depth))
            return "recovered"

        def close(self) -> None:
            operations.append(("close", context._strategy_lock.depth))

    context.strategy.iterator_factory = Iterator
    generator = make_service(context).iter_chunks("source")
    assert operations == [] and context.events == []
    assert next(generator) == "first"
    assert generator.send("value") == "sent"
    assert generator.throw(ValueError("controlled")) == "recovered"
    generator.close()
    assert operations == [("next", 0), ("send", "value", 0), ("throw", "controlled", 0), ("close", 0)]


@pytest.mark.parametrize("mode", ["shared", "thread", "call"])
def test_generator_shared_selection_and_lock_regions(mode: str) -> None:
    context = FakeExecutionContext()
    context._strategy_cache_mode = mode
    created = []

    def iterator_factory() -> Any:
        created.append(context._strategy_lock.depth)
        return iter(["yielded"])

    context.strategy.iterator_factory = iterator_factory
    generator = make_service(context).iter_chunks("source")
    assert context.events == []
    assert list(generator) == ["yielded"]
    assert created == [1 if mode == "shared" else 0]
    assert ("create" if mode == "shared" else "lookup", 1 if mode == "shared" else 0) in context.events
    assert context._cache is None and context._strategy_lock.depth == 0


@pytest.mark.parametrize("attribute", ["missing", None, "not callable"])
def test_generator_fallback_is_eager_only_during_locked_setup(attribute: Any) -> None:
    context = FakeExecutionContext()
    calls = []

    def chunk(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(context._strategy_lock.depth)
        return ["fallback"]

    strategy = SimpleNamespace(language="en", chunk=chunk)
    if attribute != "missing":
        strategy.chunk_generator = attribute
    context.strategy = strategy
    generator = make_service(context).iter_chunks("source")
    assert calls == []
    assert next(generator) == "fallback" and calls == [1]
    with pytest.raises(StopIteration) as caught:
        next(generator)
    assert caught.value.value is None and context._strategy_lock.depth == 0


@pytest.mark.parametrize("when", ["setup", "iteration"])
def test_generator_failures_are_raw_at_original_time(when: str) -> None:
    context = FakeExecutionContext()
    failure = ValueError("iterator sentinel")
    seen = []

    def during_iteration() -> Any:
        seen.append(context._strategy_lock.depth)
        yield "first"
        raise failure

    def factory() -> Any:
        if when == "setup":
            seen.append(context._strategy_lock.depth)
            raise failure
        return during_iteration()

    context.strategy.iterator_factory = factory
    generator = make_service(context).iter_chunks("source")
    assert seen == []
    if when == "iteration":
        assert next(generator) == "first"
    with pytest.raises(ValueError) as caught:
        next(generator)
    assert caught.value is failure and seen == [1 if when == "setup" else 0]
    assert context._strategy_lock.depth == 0
