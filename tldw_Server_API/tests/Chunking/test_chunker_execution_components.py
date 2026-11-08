"""Direct execution components exercise real output with a live fake context."""

from __future__ import annotations

import importlib
import json
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
    context: Any = FakeExecutionContext()
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
    operations: list[Any] = []

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


def prepare(context: Any, kind: str, options: dict[str, Any], hooks: Any = None) -> Any:
    module = importlib.import_module(f"{PACKAGE}.preparation")
    if hooks is None:
        hooks = make_service(context).hooks
    return module.prepare_execution(
        context, method="tokens", max_size=3, overlap=9, language="", user_options=options, kind=kind, hooks=hooks
    )


@pytest.mark.parametrize("kind", ["text", "metadata", "generator"])
@pytest.mark.parametrize("alignment", ["absent", None, False, True, "false"])
def test_preparation_preserves_path_specific_options(kind: str, alignment: Any) -> None:
    context = FakeExecutionContext()
    payload: list[str] = []
    options = {"payload": payload, "code_mode": "ast", "tokenizer_name": " new ", "tokenizer_name_or_path": "ignored"}
    if alignment != "absent":
        options["align_text_to_source"] = alignment
    prepared = prepare(context, kind, options)
    assert (prepared.method, prepared.max_size, prepared.overlap, prepared.language, prepared.tokenizer_override) == (
        "tokens",
        3,
        2,
        "en",
        "new",
    )
    assert prepared.use_per_call_strategy is True
    assert prepared.align_text_to_source is (kind == "metadata" and alignment is True)
    expected_raw = dict(options)
    expected_strategy: dict[str, Any] = {"payload": payload}
    if kind in {"text", "metadata"}:
        expected_raw.pop("align_text_to_source", None)
    if kind == "metadata" and alignment not in {"absent", None}:
        expected_strategy["align_text_to_source"] = alignment is True
    elif kind == "generator" and alignment != "absent":
        expected_strategy["align_text_to_source"] = alignment
    assert prepared.options_raw == expected_raw and prepared.strategy_options == expected_strategy
    assert prepared.options_raw is not options and prepared.strategy_options is not prepared.options_raw
    assert prepared.options_raw["payload"] is payload and prepared.strategy_options["payload"] is payload
    assert [event[0] for event in context.events] == ["normalize", "resolve", "aliases", "overlap"]


@pytest.mark.parametrize("kind", ["text", "metadata", "generator"])
def test_preparation_does_not_reserve_user_keys(kind: str) -> None:
    context = FakeExecutionContext()
    names = (
        "mode",
        "context",
        "hooks",
        "source",
        "path",
        "request",
        "strategy",
        "options",
        "kind",
        "user_options",
        "prepared",
        "effective_llm",
        "ephemeral",
    )
    values = {name: [f"user {name}"] for name in names}
    result = prepare(context, kind, values)
    assert result.strategy_options == values
    assert all(result.strategy_options[name] is values[name] for name in names)


@pytest.mark.parametrize("ephemeral", [True, False])
@pytest.mark.parametrize("per_call", [True, False])
def test_strategy_preparation_uses_supplied_llm_and_selection(ephemeral: bool, per_call: bool) -> None:
    class ModeTrap(FakeExecutionContext):
        def __getattribute__(self, name: str) -> Any:
            if name == "_strategy_cache_mode":
                raise AssertionError("strategy helper reread cache mode")
            return super().__getattribute__(name)

    context = ModeTrap()
    models = importlib.import_module(f"{PACKAGE}.models")
    strategy_module = importlib.import_module(f"{PACKAGE}.strategy")
    request = models.PreparedExecution(
        method="tokens",
        max_size=3,
        overlap=0,
        language="fr",
        tokenizer_override="new",
        options_raw={},
        strategy_options={},
        use_per_call_strategy=per_call,
        align_text_to_source=False,
    )
    llm = object()
    config = {"model": "supplied"}
    result = strategy_module.prepare_strategy(
        context, request, effective_llm=(llm, config), ephemeral=ephemeral, hooks=make_service(context).hooks
    )
    assert result is context.strategy
    assert result.llm_call_func is llm and result.llm_config is config
    assert (result.tokenizer_name, result._tokenizer, result._tokenizer_init_attempted, result.language) == (
        "new",
        None,
        False,
        "fr",
    )
    assert context.events == [
        ("create" if per_call or ephemeral else "lookup", 0),
        ("sync", llm, config),
        ("language", "fr"),
    ]


@pytest.mark.parametrize("kind", ["text", "metadata", "generator", "cache_hit"])
def test_service_uses_live_preparation_and_strategy_helpers(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    context = FakeExecutionContext()
    service = make_service(context)
    module = importlib.import_module(f"{PACKAGE}.service")
    prep = importlib.import_module(f"{PACKAGE}.preparation").prepare_execution
    strategy = importlib.import_module(f"{PACKAGE}.strategy").prepare_strategy
    events: list[Any] = []

    def preparation_spy(*args: Any, **kwargs: Any) -> Any:
        events.append(("prepare", kwargs["kind"]))
        return prep(*args, **kwargs)

    def strategy_spy(*args: Any, **kwargs: Any) -> Any:
        events.append(("strategy", kwargs["ephemeral"]))
        return strategy(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_execution", preparation_spy, raising=False)
    monkeypatch.setattr(module, "prepare_strategy", strategy_spy, raising=False)
    if kind == "cache_hit":
        context._cache = RecordingCache(["cached"])
        assert service.chunk_text("source") is context._cache.value
        assert events == [("prepare", "text")]
    else:
        result = (
            service.chunk_text("source")
            if kind == "text"
            else (
                service.chunk_text_with_metadata("source")
                if kind == "metadata"
                else list(service.iter_chunks("source"))
            )
        )
        assert (result[0].text if kind == "metadata" else result[0]) == (
            "strategy metadata" if kind == "metadata" else "strategy text"
        )
        assert events == [("prepare", kind), ("strategy", kind == "generator")]
    assert sum(event[0] == "llm" for event in context.events) == 1


def test_cache_key_mutation_does_not_rebuild_strategy_options() -> None:
    context: Any = FakeExecutionContext()
    context._cache = RecordingCache()
    payload: list[str] = []

    def key(
        text: str, method: str, size: int, overlap: int, language: str, options: dict[str, Any], **_kwargs: Any
    ) -> str:
        options["added"] = "raw only"
        options["payload"].append("shared")
        return "mutated"

    def emit(_path: str, options: dict[str, Any]) -> None:
        context.strategy.text_result = [json.dumps(options)]

    context._get_cache_key = key
    context.strategy.on_call = emit
    result = make_service(context).chunk_text("source", payload=payload)
    assert json.loads(result[0]) == {"payload": ["shared"]}
    assert context.strategy.calls[0][4]["payload"] is payload


def test_generator_ephemeral_decision_precedes_lock_entry() -> None:
    context = FakeExecutionContext()
    context._strategy_lock.on_enter = lambda: setattr(context, "_strategy_cache_mode", "call")
    assert list(make_service(context).iter_chunks("source")) == ["strategy text"]
    assert ("create", 1) in context.events and not any(event[0] == "lookup" for event in context.events)


@pytest.mark.parametrize("kind", ["text", "metadata", "generator"])
def test_overlap_check_failures_keep_path_logging_and_output(kind: str) -> None:
    class BrokenOverlap(int):
        def __lt__(self, other: Any) -> bool:
            raise ValueError("overlap sentinel")

    context = FakeExecutionContext()
    logs: list[Any] = []
    service = make_service(context, logs=logs)
    overlap = BrokenOverlap(1)
    if kind == "text":
        result = service.chunk_text("source", overlap=overlap)
    elif kind == "metadata":
        result = service.chunk_text_with_metadata("source", overlap=overlap)
    else:
        result = list(service.iter_chunks("source", overlap=overlap))
    assert (result[0].text if kind == "metadata" else result[0]) == (
        "strategy metadata" if kind == "metadata" else "strategy text"
    )
    assert context.strategy.calls[0][3] is overlap
    overlap_logs = [entry for entry in logs if entry[1][0].startswith("Overlap adjustment check failed:")]
    assert overlap_logs == (
        [("debug", ("Overlap adjustment check failed: overlap sentinel",), {})] if kind == "text" else []
    )


def test_tokenizer_update_failure_keeps_language_sync_and_result() -> None:
    context = FakeExecutionContext()
    context.strategy.tokenizer_error = ValueError("tokenizer sentinel")
    logs: list[Any] = []
    result = make_service(context, logs=logs).chunk_text("source", method="tokens", tokenizer_name="new", language="fr")
    assert result == ["strategy text"]
    assert context.strategy.tokenizer_name == "old" and context.strategy.language == "fr"
    assert ("debug", ("Failed to update tokenizer override",), {"exc_info": True}) in logs


def test_defaults_remain_live_until_normalization() -> None:
    context: Any = FakeExecutionContext()
    service = make_service(context)

    def normalize(method: Any) -> Any:
        context.config.default_max_size = 3
        context.config.default_overlap = 5
        context.config.language = "fr"
        return method

    context._normalize_method_argument = normalize
    result = service.chunk_text("source")
    assert result == ["strategy text"]
    assert context.strategy.calls[0][2:4] == (3, 2)
    assert context.strategy.language == "fr"
