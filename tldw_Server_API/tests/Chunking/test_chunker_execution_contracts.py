"""Public baseline contracts frozen before moving execution out of Chunker."""

from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking import chunker as facade
from tldw_Server_API.app.core.Chunking.base import ChunkerConfig, ChunkMetadata, ChunkResult
from tldw_Server_API.app.core.Chunking.exceptions import ChunkingError, InvalidInputError
from tldw_Server_API.app.core.Chunking.llm_context import llm_override_scope
from tldw_Server_API.tests.Chunking._execution_test_support import RecordingCache, RecordingLock, RecordingStrategy

pytestmark = pytest.mark.unit
PATHS = ("text", "metadata", "generator")
OPTION_NAMES = (
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
    "effective_llm",
    "ephemeral",
)

DOCSTRINGS = (
    """
        Chunk text using the specified method.

        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options

        Returns:
            List of text chunks

        Raises:
            InvalidInputError: If input validation fails
            InvalidChunkingMethodError: If method is not supported
            ChunkingError: For other chunking errors
        """,
    """
        Chunk text and return results with metadata.

        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options

        Returns:
            List of ChunkResult objects with text and metadata.
            Note: Extractive strategies (words/sentences/paragraphs) return source-aligned
            text by default. Set `align_text_to_source=False` to keep normalized text
            when supported, or `align_text_to_source=True` to force source slicing for
            transforming strategies.
        """,
    """
        Memory-efficient generator for chunking large texts.

        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options

        Yields:
            Individual text chunks
        """,
)


def make_case(**config_options: Any) -> SimpleNamespace:
    """Keep real facade facilities except the potentially external strategy."""
    config = {"enable_cache": False, "default_max_size": 8, "default_overlap": 0, "language": "en"}
    config.update(config_options)
    chunker = Chunker(ChunkerConfig(**config))
    events: list[Any] = []
    lock = RecordingLock(events)
    chunker._strategy_lock = lock
    strategy = RecordingStrategy(events, lock)

    def factory() -> RecordingStrategy:
        events.append(("factory", lock.depth))
        return strategy

    chunker._strategies["words"] = strategy
    chunker._strategy_factories["words"] = factory
    return SimpleNamespace(chunker=chunker, strategy=strategy, events=events, lock=lock)


@pytest.fixture
def facade_case() -> Callable[..., SimpleNamespace]:
    return make_case


def run_path(case: SimpleNamespace, path: str, /, text: Any = "alpha beta", **kwargs: Any) -> Any:
    methods = {
        "text": case.chunker.chunk_text,
        "metadata": case.chunker.chunk_text_with_metadata,
        "generator": case.chunker.chunk_text_generator,
    }
    result = methods[path](text, **kwargs)
    return list(result) if path == "generator" else result


def output_text(result: Any, path: str) -> str:
    return result[0].text if path == "metadata" else result[0]


def test_public_execution_contracts() -> None:
    p = inspect.Parameter
    parameters = [p("self", p.POSITIONAL_OR_KEYWORD), p("text", p.POSITIONAL_OR_KEYWORD, annotation=str)]
    parameters.extend(
        p(name, p.POSITIONAL_OR_KEYWORD, annotation=annotation, default=None)
        for name, annotation in (
            ("method", str | None),
            ("max_size", int | None),
            ("overlap", int | None),
            ("language", str | None),
        )
    )
    parameters.append(p("options", p.VAR_KEYWORD))
    methods = (Chunker.chunk_text, Chunker.chunk_text_with_metadata, Chunker.chunk_text_generator)
    for method, annotation, docstring in zip(
        methods, (list[str], list[ChunkResult], Generator[str, None, None]), DOCSTRINGS
    ):
        hints = get_type_hints(method)
        actual = inspect.signature(method)
        actual = actual.replace(
            parameters=[
                value.replace(annotation=hints.get(name, p.empty)) for name, value in actual.parameters.items()
            ],
            return_annotation=hints["return"],
        )
        assert actual == inspect.Signature(parameters, return_annotation=annotation)
        assert inspect.getdoc(method) == inspect.cleandoc(docstring)
    assert inspect.isgeneratorfunction(Chunker.chunk_text_generator)


@pytest.mark.parametrize("path", PATHS)
def test_path_input_and_option_contracts(facade_case: Any, path: str) -> None:
    case = facade_case()
    sizes = []
    case.chunker._enforce_text_size = lambda text, *, source: sizes.append((text, source))
    result = run_path(case, path, " \t\n")
    if path == "generator":
        assert result == ["strategy text"]
        assert sizes == [(" \t\n", "chunk_text_generator")]
    else:
        assert result == []
        assert sizes == [] and case.strategy.calls == []
    with pytest.raises(InvalidInputError, match="Expected string input"):
        run_path(case, path, 42)


@pytest.mark.parametrize("path", PATHS)
def test_size_failure_keeps_path_label_and_identity(facade_case: Any, path: str) -> None:
    case = facade_case()
    failure = ValueError("size sentinel")
    sources = []

    def reject(text: str, *, source: str) -> None:
        sources.append(source)
        raise failure

    case.chunker._enforce_text_size = reject
    with pytest.raises(ValueError) as caught:
        run_path(case, path)
    assert caught.value is failure
    assert sources == [
        {"text": "chunk_text", "metadata": "chunk_text_with_metadata", "generator": "chunk_text_generator"}[path]
    ]
    assert case.strategy.calls == []


@pytest.mark.parametrize("path", PATHS)
def test_option_namespace_survives_all_paths(facade_case: Any, path: str) -> None:
    case = facade_case()
    values = {name: f"user-{name}" for name in OPTION_NAMES}
    payload: list[str] = []
    values["payload"] = payload

    def emit(_path: str, options: dict[str, Any]) -> None:
        encoded = json.dumps({key: options[key] for key in OPTION_NAMES}, sort_keys=True)
        case.strategy.text_result = [encoded]
        case.strategy.metadata_result = [ChunkResult(encoded, ChunkMetadata(0, 0, 1, 1))]

    case.strategy.on_call = emit
    result = run_path(case, path, **values)
    assert json.loads(output_text(result, path)) == {key: values[key] for key in OPTION_NAMES}
    assert case.strategy.calls[0][4]["payload"] is payload


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("alignment", [None, False, True, "false"])
def test_path_alias_overlap_tokenizer_order(facade_case: Any, path: str, alignment: Any) -> None:
    case = facade_case()
    case.chunker._strategies["tokens"] = case.strategy
    case.chunker._strategy_factories["tokens"] = lambda: case.strategy
    events = []
    original_sync = case.chunker._sync_strategy_llm
    original_language = case.chunker._apply_strategy_language

    def sync(strategy: Any, **kwargs: Any) -> None:
        events.append(("llm", strategy._tokenizer_init_attempted))
        original_sync(strategy, **kwargs)

    def language(strategy: Any, lang: str) -> None:
        events.append(("language", strategy.tokenizer_name, strategy._tokenizer, strategy._tokenizer_init_attempted))
        original_language(strategy, lang)

    case.chunker._sync_strategy_llm = sync
    case.chunker._apply_strategy_language = language
    result = run_path(
        case,
        path,
        method="tokens",
        max_size=3,
        overlap=9,
        language="fr",
        tokenizer_name=" new ",
        tokenizer_name_or_path="ignored",
        code_mode="ast",
        align_text_to_source=alignment,
    )
    assert result
    assert case.strategy.calls[0][2:4] == (3, 2)
    expected = (
        {"align_text_to_source": alignment}
        if path == "generator"
        else ({"align_text_to_source": alignment is True} if path == "metadata" and alignment is not None else {})
    )
    assert case.strategy.calls[0][4] == expected
    assert events == [("llm", True), ("language", "new", None, False)]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("size", "overlap", "expected"),
    [(None, None, (8, 0)), (0, -2, (0, 0)), (True, True, (True, 0)), ("3", "2", ("3", "2"))],
)
def test_defaults_only_replace_none(facade_case: Any, path: str, size: Any, overlap: Any, expected: Any) -> None:
    case = facade_case()
    assert run_path(case, path, max_size=size, overlap=overlap)
    assert case.strategy.calls[0][2:4] == expected


@pytest.mark.parametrize("path", PATHS)
def test_resolution_alias_and_overlap_order(facade_case: Any, path: str) -> None:
    case = facade_case()
    case.chunker._strategies["semantic"] = case.strategy
    case.chunker._strategy_factories["semantic"] = lambda: case.strategy
    result = run_path(case, path, method="semantic", semantic_similarity_threshold=0.7, semantic_overlap_sentences="2")
    assert result
    assert case.strategy.calls[0][3] == 2
    assert case.strategy.calls[0][4] == {
        "semantic_similarity_threshold": 0.7,
        "similarity_threshold": 0.7,
        "semantic_overlap_sentences": "2",
    }
    case.chunker._strategies["code_ast"] = case.strategy
    case.chunker._strategy_factories["code_ast"] = lambda: case.strategy
    assert run_path(case, path, method="code", language="python", code_mode="auto")
    assert case.strategy.calls[-1][4] == {}


def test_text_cache_identity_and_error_regions(facade_case: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    case = facade_case()
    cached = ["cached text"]
    cache = RecordingCache(cached)
    case.chunker._cache = cache
    metrics = []
    monkeypatch.setattr(facade, "increment_counter", lambda name, **kwargs: metrics.append((name, kwargs)))
    assert case.chunker.chunk_text("alpha") is cached
    assert case.strategy.calls == [] and case.events == []
    assert metrics == [("chunker_cache_get_total", {"labels": {"result": "hit", "reason": "allowed"}})]
    cache.value = None
    cache.put_error = RuntimeError("put sentinel")
    assert case.chunker.chunk_text("alpha") is case.strategy.text_result
    assert metrics[-1] == ("chunker_cache_put_total", {"labels": {"result": "error", "reason": "allowed"}})


@pytest.mark.parametrize("seam", ["_get_effective_llm_hooks", "_get_llm_signature", "_get_cache_key", "get"])
def test_cache_pre_execution_failures_are_raw(facade_case: Any, seam: str) -> None:
    case = facade_case()
    cache = RecordingCache()
    case.chunker._cache = cache
    failure = ValueError("cache sentinel")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    if seam == "get":
        cache.get_error = failure
    else:
        setattr(case.chunker, seam, fail)
    with pytest.raises(ValueError) as caught:
        case.chunker.chunk_text("alpha")
    assert caught.value is failure and case.strategy.calls == []


@pytest.mark.parametrize(
    ("minimum", "maximum", "reason"),
    [(20, 100, "skipped_min_len"), (0, 2, "skipped_max_len"), ("bad", 100, "policy_error")],
)
def test_cache_policy_skips_keep_real_output(
    facade_case: Any, monkeypatch: pytest.MonkeyPatch, minimum: Any, maximum: Any, reason: str
) -> None:
    case = facade_case()
    case.chunker._cache = RecordingCache()
    case.chunker.config.min_text_length_to_cache = minimum
    case.chunker.config.max_text_length_to_cache = maximum
    metrics = []
    monkeypatch.setattr(facade, "increment_counter", lambda name, **kwargs: metrics.append((name, kwargs["labels"])))
    assert case.chunker.chunk_text("alpha") == ["strategy text"]
    assert case.chunker._cache.events == []
    assert metrics == [
        ("chunker_cache_get_total", {"result": "skip", "reason": reason}),
        ("chunker_cache_put_total", {"result": "skipped", "reason": reason}),
    ]


@pytest.mark.parametrize("copy_on_access", [True, False])
def test_real_lru_copy_modes(facade_case: Any, copy_on_access: bool) -> None:
    case = facade_case(enable_cache=True, cache_copy_on_access=copy_on_access)
    case.chunker._cache.copy_on_access = copy_on_access
    first = case.chunker.chunk_text("alpha")
    first.append("caller mutation")
    second = case.chunker.chunk_text("alpha")
    assert second == (["strategy text"] if copy_on_access else ["strategy text", "caller mutation"])
    assert (second is first) is (not copy_on_access)


@pytest.mark.parametrize("path", PATHS)
def test_strategy_failure_regions(facade_case: Any, path: str) -> None:
    case = facade_case()
    failure = ValueError("strategy sentinel")

    def fail(_path: str, _options: Any) -> None:
        raise failure

    case.strategy.on_call = fail
    if path == "generator":
        with pytest.raises(ValueError) as caught:
            run_path(case, path)
        assert caught.value is failure
    else:
        with pytest.raises(ChunkingError, match="Chunking failed: strategy sentinel") as caught:
            run_path(case, path)
        assert caught.value.__cause__ is failure
    assert case.lock.depth == 0


@pytest.mark.parametrize("path", ("text", "metadata"))
def test_chunking_error_passthrough(facade_case: Any, path: str) -> None:
    case = facade_case()
    failure = ChunkingError("already translated")

    def fail(_path: str, _options: Any) -> None:
        raise failure

    case.strategy.on_call = fail
    with pytest.raises(ChunkingError) as caught:
        run_path(case, path)
    assert caught.value is failure


@pytest.mark.parametrize(
    ("start", "end", "expected"), [(-5, 99, "a bcd"), (3, 1, ""), (True, True, ""), (False, True, "a")]
)
def test_metadata_alignment_identity_and_error_regions(facade_case: Any, start: Any, end: Any, expected: str) -> None:
    case = facade_case()
    metadata = ChunkMetadata(0, start, end, 1)
    item = ChunkResult("normalized", metadata)
    results = [item]
    case.strategy.metadata_result = results
    cache = RecordingCache()
    case.chunker._cache = cache
    result = case.chunker.chunk_text_with_metadata("a\u202ebcd", align_text_to_source=True)
    assert result is results and result[0] is item and item.metadata is metadata
    assert item.text == expected
    assert (metadata.start_char, metadata.end_char) == (start, end)
    assert cache.events == []


def test_malformed_metadata_items_do_not_hide_other_results(facade_case: Any) -> None:
    class RaisingItem:
        @property
        def metadata(self) -> Any:
            raise ValueError("bad metadata")

    case = facade_case()
    valid = ChunkResult("wrong", ChunkMetadata(0, 1, 4, 1))
    missing = SimpleNamespace(text="missing")
    noninteger = SimpleNamespace(text="unchanged", metadata=SimpleNamespace(start_char="1", end_char=4))
    results = [RaisingItem(), missing, noninteger, valid]
    case.strategy.metadata_result = results
    assert case.chunker.chunk_text_with_metadata("abcde", align_text_to_source=True) is results
    assert valid.text == "bcd" and missing.text == "missing" and noninteger.text == "unchanged"
    case.strategy.metadata_result = (valid,)
    valid.text = "tuple unchanged"
    assert case.chunker.chunk_text_with_metadata("abcde", align_text_to_source=True) is case.strategy.metadata_result
    assert valid.text == "tuple unchanged"


@pytest.mark.parametrize("path", ["text", "metadata"])
def test_success_logging_failure_keeps_original_region(
    facade_case: Any, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    case = facade_case()
    failure = ValueError("success log sentinel")

    class Log:
        def debug(self, message: str, **_kwargs: Any) -> None:
            if message.startswith("Created"):
                raise failure

        def error(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(facade, "logger", Log())
    with pytest.raises(ChunkingError if path == "text" else ValueError) as caught:
        run_path(case, path)
    assert (caught.value.__cause__ if path == "text" else caught.value) is failure


@pytest.mark.parametrize("mode", ["shared", "thread", "call"])
@pytest.mark.parametrize("path", PATHS)
def test_strategy_modes_and_injected_generator_rule(facade_case: Any, mode: str, path: str) -> None:
    case = facade_case(strategy_cache_mode=mode)
    assert run_path(case, path)
    assert (
        ("factory", 1) in case.events
        if path == "generator" and mode == "shared"
        else not any(event[0] == "factory" for event in case.events)
    )
    assert case.strategy.calls[0][0] == path
    assert (path, 1 if mode == "shared" else 0) in case.events


def test_generator_baseline_lifecycle_and_live_state(facade_case: Any) -> None:
    case = facade_case()
    generator = case.chunker.chunk_text_generator("alpha")
    assert case.events == []
    case.chunker._sanitize_input = lambda *_args, **_kwargs: "later source"
    case.chunker.config.default_max_size = 3
    assert next(generator) == "strategy text"
    assert case.strategy.calls[0][1:4] == ("later source", 3, 0)
    with pytest.raises(StopIteration) as caught:
        next(generator)
    assert caught.value.value is None
    case.events.clear()
    case.chunker.chunk_text_generator(42).close()
    assert case.events == []


def test_generator_uses_advancing_threads_llm_override(facade_case: Any) -> None:
    case = facade_case()
    generator = case.chunker.chunk_text_generator("alpha")
    results = []
    errors = []
    worker_llm = object()

    def advance() -> None:
        try:
            with llm_override_scope(case.chunker, worker_llm, {"model": "worker"}):
                results.extend(generator)
        except (RuntimeError, ValueError, TypeError) as error:
            errors.append(error)

    thread = threading.Thread(target=advance)
    thread.start()
    try:
        thread.join(5)
        assert not thread.is_alive() and errors == []
        assert results == ["strategy text"]
        assert case.strategy.llm_call_func is worker_llm and case.strategy.llm_config == {"model": "worker"}
    finally:
        thread.join(5)
        if not thread.is_alive():
            generator.close()


JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=12),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=5), children, max_size=3),
    max_leaves=8,
)


@settings(max_examples=50, deadline=None)
@given(st.dictionaries(st.sampled_from(OPTION_NAMES), JSON_VALUES, min_size=1, max_size=len(OPTION_NAMES)))
def test_generated_user_options_reach_each_path(options: dict[str, Any]) -> None:
    for path in PATHS:
        case = make_case()

        def emit(_path: str, received: dict[str, Any], case: Any = case) -> None:
            value = json.dumps(received, sort_keys=True)
            case.strategy.text_result = [value]
            case.strategy.metadata_result = [ChunkResult(value, ChunkMetadata(0, 0, 1, 1))]

        case.strategy.on_call = emit
        assert json.loads(output_text(run_path(case, path, **options), path)) == options
        assert all(
            case.strategy.calls[0][4][key] is value for key, value in options.items() if isinstance(value, (dict, list))
        )


@settings(max_examples=50, deadline=None)
@given(st.text(max_size=30), st.integers(), st.integers())
def test_generated_alignment_preserves_objects_and_slices(suffix: str, start: int, end: int) -> None:
    case = make_case()
    text = "a" + suffix
    sanitized = case.chunker._sanitize_input(text)
    metadata = ChunkMetadata(0, start, end, 1)
    item = ChunkResult("wrong", metadata)
    case.strategy.metadata_result = [item]
    result = case.chunker.chunk_text_with_metadata(text, align_text_to_source=True)
    left = max(0, min(start, len(sanitized)))
    right = max(left, min(end, len(sanitized)))
    assert result is case.strategy.metadata_result and result[0] is item and item.metadata is metadata
    assert item.text == sanitized[left:right]
    assert (metadata.start_char, metadata.end_char) == (start, end)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8), min_size=1, max_size=20),
    st.integers(min_value=1, max_value=10),
    st.data(),
)
def test_words_eager_generator_equivalence(words: list[str], size: int, data: Any) -> None:
    overlap = data.draw(st.integers(min_value=0, max_value=size - 1))
    chunker = Chunker(ChunkerConfig(enable_cache=False))
    text = " ".join(words)
    eager = chunker.chunk_text(text, method="words", max_size=size, overlap=overlap)
    assert eager == list(chunker.chunk_text_generator(text, method="words", max_size=size, overlap=overlap))
    assert eager and eager[0].split()[0] == words[0]


def test_metric_replacement_during_execution_is_live(facade_case: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    case = facade_case()
    case.chunker._cache = RecordingCache()
    old_events = []
    new_events = []
    monkeypatch.setattr(facade, "increment_counter", lambda name, **kwargs: old_events.append((name, kwargs)))

    def replace(_path: str, _options: Any) -> None:
        monkeypatch.setattr(facade, "increment_counter", lambda name, **kwargs: new_events.append((name, kwargs)))

    case.strategy.on_call = replace
    assert case.chunker.chunk_text("alpha") == ["strategy text"]
    assert old_events == [("chunker_cache_get_total", {"labels": {"result": "miss", "reason": "allowed"}})]
    assert new_events == [("chunker_cache_put_total", {"labels": {"result": "stored", "reason": "allowed"}})]


@pytest.mark.parametrize("path", PATHS)
def test_effective_llm_is_obtained_once(facade_case: Any, path: str) -> None:
    case = facade_case()
    calls = []
    llm = object()
    config = {"model": "first"}

    def effective() -> Any:
        calls.append("effective")
        return llm, config

    case.chunker._get_effective_llm_hooks = effective
    assert run_path(case, path)
    assert calls == ["effective"]
    assert case.strategy.llm_call_func is llm
    assert case.strategy.llm_config == config and case.strategy.llm_config is not config


def test_generator_send_throw_close_and_completion(facade_case: Any) -> None:
    case = facade_case()
    events = []

    def underlying() -> Any:
        try:
            events.append(("next", case.lock.depth))
            sent = yield "first"
            events.append(("send", sent, case.lock.depth))
            try:
                yield "second"
            except ValueError:
                events.append(("throw", case.lock.depth))
                yield "recovered"
            return "not propagated"
        finally:
            events.append(("close", case.lock.depth))

    case.strategy.iterator_factory = underlying
    generator = case.chunker.chunk_text_generator("alpha")
    assert case.events == [] and events == []
    assert next(generator) == "first"
    assert generator.send("sentinel") == "second"
    assert generator.throw(ValueError("controlled")) == "recovered"
    with pytest.raises(StopIteration) as caught:
        next(generator)
    assert caught.value.value is None
    assert events == [("next", 0), ("send", "sentinel", 0), ("throw", 0), ("close", 0)]
    events.clear()
    generator = case.chunker.chunk_text_generator("alpha")
    assert next(generator) == "first"
    generator.close()
    assert events == [("next", 0), ("close", 0)]


@pytest.mark.parametrize("generator_attribute", [None, "not callable"])
def test_generator_eager_fallback_is_created_under_lock(facade_case: Any, generator_attribute: Any) -> None:
    case = facade_case()
    case.strategy.chunk_generator = generator_attribute
    generator = case.chunker.chunk_text_generator("alpha")
    assert case.events == []
    assert list(generator) == ["strategy text"]
    assert case.strategy.calls[0][0] == "text"
    assert ("text", 1) in case.events and case.lock.depth == 0


def test_empty_strategy_result_is_not_cached(facade_case: Any) -> None:
    case = facade_case()
    case.strategy.text_result = []
    case.chunker._cache = RecordingCache()
    assert case.chunker.chunk_text("alpha") is case.strategy.text_result
    assert [entry[0] for entry in case.chunker._cache.events] == ["get"]


@pytest.mark.parametrize("mode", ["shared", "thread", "call"])
@pytest.mark.parametrize("path", PATHS)
def test_factory_reuse_without_injected_strategy(facade_case: Any, mode: str, path: str) -> None:
    case = facade_case(strategy_cache_mode=mode)
    case.chunker._strategies.clear()
    instances = []

    def factory() -> RecordingStrategy:
        strategy = RecordingStrategy(case.events, case.lock)
        strategy.text_result = [f"instance {len(instances) + 1}"]
        strategy.metadata_result[0].text = strategy.text_result[0]
        instances.append(strategy)
        return strategy

    case.chunker._strategy_factories["words"] = factory
    first = output_text(run_path(case, path), path)
    second = output_text(run_path(case, path), path)
    separate = mode == "call" or (path == "generator" and mode == "shared")
    assert first == "instance 1" and second == ("instance 2" if separate else "instance 1")
    assert len(instances) == (2 if separate else 1)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("primary", "alias", "expected"),
    [(None, " alias ", "alias"), (" ", "ignored", "old"), (False, "ignored", "old"), (17, "ignored", 17)],
)
def test_tokenizer_selector_precedence_and_dynamic_values(
    facade_case: Any, path: str, primary: Any, alias: Any, expected: Any
) -> None:
    case = facade_case()
    case.chunker._strategies["tokens"] = case.strategy
    case.chunker._strategy_factories["tokens"] = lambda: case.strategy
    assert run_path(case, path, method="tokens", tokenizer_name=primary, tokenizer_name_or_path=alias)
    assert case.strategy.tokenizer_name == expected
    assert case.strategy.calls[0][4] == {}


def test_alignment_offset_and_assignment_failures_are_per_item(facade_case: Any) -> None:
    class BrokenOffset:
        @property
        def start_char(self) -> int:
            raise ValueError("offset sentinel")

    class ReadOnlyItem:
        metadata = ChunkMetadata(0, 0, 1, 1)

        @property
        def text(self) -> str:
            return "read only"

    case = facade_case()
    broken = SimpleNamespace(text="kept", metadata=BrokenOffset())
    read_only = ReadOnlyItem()
    valid = ChunkResult("wrong", ChunkMetadata(0, 1, 3, 1))
    case.strategy.metadata_result = [broken, read_only, valid]
    result = case.chunker.chunk_text_with_metadata("abcd", align_text_to_source=True)
    assert result is case.strategy.metadata_result
    assert (broken.text, read_only.text, valid.text) == ("kept", "read only", "bc")
