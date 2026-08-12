"""Failure, cancellation, and sanitization tests for canonical extraction."""

import ast
import asyncio
import dataclasses
import inspect
from pathlib import Path
from typing import Any, Optional

import pytest
from loguru import logger

from tldw_Server_API.app.core.exceptions import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import extraction
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline, throttles
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    ExtractionDependencies,
    build_default_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm as llm_strategy
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import schema as schema_strategy

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_SCRAPING_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping"
SECRET = "https://user:api-key@example.com/private/path?token=secret#payload"


class _UnavailableSemaphore:
    """Test double that records timed acquisition attempts without granting a slot."""

    def __init__(self) -> None:
        self.acquire_attempts = 0
        self.release_calls = 0

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        del blocking, timeout
        self.acquire_attempts += 1
        return False

    def release(self) -> None:
        self.release_calls += 1


def _dependencies(**changes: Any) -> ExtractionDependencies:
    defaults = build_default_dependencies()
    replacements = {
        "increment_counter": lambda *_args, **_kwargs: None,
        "observe_histogram": lambda *_args, **_kwargs: None,
        "log_counter": lambda *_args, **_kwargs: None,
        "cancellation_checkpoint": lambda: None,
    }
    replacements.update(changes)
    return dataclasses.replace(
        defaults,
        **replacements,
    )


def _install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    **changes: Any,
) -> ExtractionDependencies:
    dependencies = _dependencies(**changes)
    monkeypatch.setattr(module, "build_default_dependencies", lambda: dependencies)
    return dependencies


def _response(content: str, *, model: str = "gpt-test") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "model": model,
    }


def test_llm_and_generator_exports_are_object_identical_with_exact_signatures() -> None:
    llm_signature = inspect.Signature(
        parameters=(
            inspect.Parameter("html_text", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "llm_settings",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[dict[str, Any]],
            ),
            inspect.Parameter(
                "schema_rules",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[dict[str, Any]],
            ),
        ),
        return_annotation=dict[str, Any],
    )
    schema_signature = inspect.Signature(
        parameters=(
            inspect.Parameter("html_text", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "llm_settings",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[dict[str, Any]],
            ),
            inspect.Parameter("query", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str]),
            inspect.Parameter(
                "example_json",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[str],
            ),
        ),
        return_annotation=dict[str, Any],
    )
    regex_signature = inspect.Signature(
        parameters=(
            inspect.Parameter("html_text", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "llm_settings",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[dict[str, Any]],
            ),
            inspect.Parameter("label", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str]),
            inspect.Parameter("query", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str]),
            inspect.Parameter(
                "examples",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[list[str]],
            ),
        ),
        return_annotation=dict[str, Any],
    )

    assert extraction.extract_llm_entities is llm_strategy.extract_llm_entities
    assert legacy.extract_llm_entities is llm_strategy.extract_llm_entities
    assert extraction.generate_schema_rules_from_llm is schema_strategy.generate_schema_rules_from_llm
    assert legacy.generate_schema_rules_from_llm is schema_strategy.generate_schema_rules_from_llm
    assert extraction.generate_regex_pattern_from_llm is schema_strategy.generate_regex_pattern_from_llm
    assert legacy.generate_regex_pattern_from_llm is schema_strategy.generate_regex_pattern_from_llm
    assert inspect.signature(llm_strategy.extract_llm_entities) == llm_signature
    assert inspect.signature(schema_strategy.generate_schema_rules_from_llm) == schema_signature
    assert inspect.signature(schema_strategy.generate_regex_pattern_from_llm) == regex_signature


def test_llm_resolves_default_provider_and_uses_call_time_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=lambda **kwargs: calls.append(kwargs) or _response('{"content": "Body"}'),
    )
    monkeypatch.setattr(llm_strategy, "_load_app_config", lambda: {"RAG_DEFAULT_LLM_PROVIDER": "OpenAI"})

    result = extraction.extract_llm_entities("<p>Body</p>", "https://example.com")

    assert result["extraction_successful"] is True
    assert result["llm_provider"] == "openai"
    assert calls[0]["api_provider"] == "openai"


def test_llm_uses_the_first_json_object_per_chunk_and_aggregates_chunk_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _response('{"title": "Title"}{"blocks": [{"text": "First"}]}'),
            _response('{"content": "Second"}'),
            _response('{"summary": "Third"}'),
        ]
    )
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=lambda **_kwargs: next(responses),
    )

    html = "<p>" + " ".join(f"word-{index}" for index in range(120)) + "</p>"
    result = extraction.extract_llm_entities(
        html,
        "https://example.com/chunks",
        llm_settings={
            "provider": "openai",
            "chunk_token_threshold": 50,
            "word_token_rate": 1.0,
            "overlap_rate": 0.0,
        },
    )

    assert result["title"] == "Title"
    assert result["content"] == "Second"
    assert result["llm_extraction"] == {"title": "Title", "content": "Second", "summary": "Third"}
    assert result["llm_usage"] == {"prompt_tokens": 6, "completion_tokens": 9, "total_tokens": 15}


def test_fenced_json_is_parsed_as_a_direct_candidate() -> None:
    candidates = llm_strategy._json_candidates('prefix\n```json\n{"content": "Body"}\n```\ntrailer')

    assert candidates[0] == '{"content": "Body"}'


def test_strict_json_failure_uses_only_stable_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=lambda **_kwargs: _response(f"not-json {SECRET}"),
    )

    result = extraction.extract_llm_entities(
        "<p>Body</p>",
        "https://example.com/strict",
        llm_settings={"provider": "openai", "strict_json": True},
    )

    assert result["llm_error"] == "strict_json_failed"
    assert SECRET not in str(result)


def test_cancellation_runs_before_llm_work(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def cancel() -> None:
        events.append("checkpoint")
        raise asyncio.CancelledError

    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=lambda **_kwargs: events.append("provider"),
        cancellation_checkpoint=cancel,
    )

    with pytest.raises(asyncio.CancelledError):
        extraction.extract_llm_entities("", "https://example.com/cancel")

    assert events == ["checkpoint"]


def test_cancellation_runs_before_next_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints = 0
    provider_calls = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 4:
            raise asyncio.CancelledError

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        return _response('{"content": "Body"}')

    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=provider,
        cancellation_checkpoint=checkpoint,
    )
    html = "<p>" + " ".join(f"word-{index}" for index in range(80)) + "</p>"

    with pytest.raises(asyncio.CancelledError):
        extraction.extract_llm_entities(
            html,
            "https://example.com/chunks",
            llm_settings={
                "provider": "openai",
                "chunk_token_threshold": 50,
                "word_token_rate": 1.0,
                "overlap_rate": 0.0,
            },
        )

    assert provider_calls == 1


def test_cancellation_runs_between_provider_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = 0
    provider_calls = 0
    sleeps: list[float] = []

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 4:
            raise asyncio.CancelledError

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        raise RuntimeError(SECRET)

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "2")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "10")
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=provider,
        cancellation_checkpoint=checkpoint,
        sleep=sleeps.append,
    )

    with pytest.raises(asyncio.CancelledError):
        extraction.extract_llm_entities(
            "<p>Body</p>",
            "https://example.com/retry",
            llm_settings={"provider": "openai"},
        )

    assert provider_calls == 1
    assert sleeps == []


def test_cancellation_during_throttle_wait_prevents_provider_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = False
    provider_calls = 0

    def checkpoint() -> None:
        if cancelled:
            raise asyncio.CancelledError

    def sleep(_delay: float) -> None:
        nonlocal cancelled
        cancelled = True

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        return _response('{"content": "Unexpected"}')

    monkeypatch.setenv("LLM_DELAY_MS", "10")
    throttles.clear_throttle_state()
    throttles.apply_llm_delay(
        "openai",
        10.0,
        0.0,
        wall_time=lambda: 1000.0,
        sleep=lambda _delay: None,
    )
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=provider,
        cancellation_checkpoint=checkpoint,
        sleep=sleep,
        wall_time=lambda: 1000.0,
    )

    with pytest.raises(asyncio.CancelledError):
        extraction.extract_llm_entities(
            "<p>Body</p>",
            "https://example.com/throttle-cancel",
            llm_settings={"provider": "openai"},
        )

    assert provider_calls == 0


def test_strategy_throttle_polls_cancellation_while_semaphore_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semaphore = _UnavailableSemaphore()
    entered = False

    def checkpoint() -> None:
        if semaphore.acquire_attempts >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: semaphore)
    dependencies = _dependencies(cancellation_checkpoint=checkpoint)

    with pytest.raises(asyncio.CancelledError):
        with pipeline._strategy_throttle("regex", dependencies):
            entered = True

    assert entered is False
    assert semaphore.acquire_attempts == 2
    assert semaphore.release_calls == 0


def test_llm_throttle_polls_cancellation_while_semaphore_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semaphore = _UnavailableSemaphore()
    entered = False

    def checkpoint() -> None:
        if semaphore.acquire_attempts >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(throttles, "get_llm_semaphore", lambda *_args: semaphore)
    dependencies = _dependencies(cancellation_checkpoint=checkpoint)

    with pytest.raises(asyncio.CancelledError):
        with llm_strategy._llm_throttle("openai", {"max_concurrency": 1}, dependencies):
            entered = True

    assert entered is False
    assert semaphore.acquire_attempts == 2
    assert semaphore.release_calls == 0


def test_llm_concurrency_normalizes_provider_identity() -> None:
    throttles.clear_throttle_state()
    try:
        mixed_case = throttles.get_llm_semaphore(" OpenAI ", 1)
        canonical = throttles.get_llm_semaphore("openai", 1)

        assert mixed_case is canonical
    finally:
        throttles.clear_throttle_state()


def test_llm_delay_normalizes_provider_identity() -> None:
    sleeps: list[float] = []
    throttles.clear_throttle_state()
    try:
        throttles.apply_llm_delay(
            "",
            100.0,
            0.0,
            wall_time=lambda: 1000.0,
            sleep=sleeps.append,
        )
        throttles.apply_llm_delay(
            " DEFAULT ",
            100.0,
            0.0,
            wall_time=lambda: 1000.0,
            sleep=sleeps.append,
        )

        assert sleeps == [0.1]
    finally:
        throttles.clear_throttle_state()


def test_cancellation_during_retry_backoff_prevents_later_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False
    provider_calls = 0

    def checkpoint() -> None:
        if cancelled:
            raise asyncio.CancelledError

    def sleep(_delay: float) -> None:
        nonlocal cancelled
        cancelled = True

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise RuntimeError(SECRET)
        return _response('{"content": "Unexpected"}')

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "1")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "10")
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=provider,
        cancellation_checkpoint=checkpoint,
        sleep=sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        extraction.extract_llm_entities(
            "<p>Body</p>",
            "https://example.com/retry-cancel",
            llm_settings={"provider": "openai"},
        )

    assert provider_calls == 1


def test_provider_retry_uses_backoff_and_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_calls = 0
    sleeps: list[float] = []

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise TimeoutError(SECRET)
        return _response('{"content": "Recovered"}')

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "1")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "25")
    monkeypatch.setenv("EXTRACTOR_RETRY_JITTER_MS", "0")
    _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=provider,
        sleep=sleeps.append,
    )

    result = extraction.extract_llm_entities(
        "<p>Body</p>",
        "https://example.com/retry",
        llm_settings={"provider": "openai"},
    )

    assert result["content"] == "Recovered"
    assert provider_calls == 2
    assert sleeps == [0.025]


class _ArbitraryProviderSDKError(Exception):
    pass


@pytest.mark.parametrize(
    "error_type",
    [
        ChatAuthenticationError,
        ChatBadRequestError,
        ChatRateLimitError,
        ChatProviderError,
        _ArbitraryProviderSDKError,
    ],
)
def test_provider_exceptions_are_sanitized_and_retried(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    provider_calls = 0

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        raise error_type(SECRET)

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "1")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "0")
    monkeypatch.setenv("EXTRACTOR_RETRY_JITTER_MS", "0")
    _install_dependencies(monkeypatch, llm_strategy, perform_chat_api_call=provider)
    records: list[Any] = []
    handler_id = logger.add(records.append, level="WARNING")
    try:
        result = extraction.extract_llm_entities(
            "<p>Body</p>",
            "https://example.com/provider-error",
            llm_settings={"provider": "openai"},
        )
    finally:
        logger.remove(handler_id)

    assert provider_calls == 2
    assert result["llm_error"] == "provider_error"
    assert SECRET not in str(result)
    assert len(records) == 2
    assert all(SECRET not in str(record.record) for record in records)


@pytest.mark.parametrize("signal_type", [asyncio.CancelledError, KeyboardInterrupt])
def test_provider_base_exceptions_are_not_translated_or_retried(
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    provider_calls = 0

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        raise signal_type

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "2")
    _install_dependencies(monkeypatch, llm_strategy, perform_chat_api_call=provider)

    with pytest.raises(signal_type):
        extraction.extract_llm_entities(
            "<p>Body</p>",
            "https://example.com/provider-cancel",
            llm_settings={"provider": "openai"},
        )

    assert provider_calls == 1


def test_pipeline_cancellation_after_terminal_provider_error_is_not_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False
    provider_calls = 0

    def checkpoint() -> None:
        if cancelled:
            raise asyncio.CancelledError

    def provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal cancelled, provider_calls
        provider_calls += 1
        cancelled = True
        raise ChatProviderError(SECRET)

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "0")
    monkeypatch.setattr(pipeline, "get_strategy_semaphore", lambda *_args: None)
    dependencies = _dependencies(
        perform_chat_api_call=provider,
        cancellation_checkpoint=checkpoint,
    )

    with pytest.raises(asyncio.CancelledError):
        pipeline._extract_article_with_pipeline_with_dependencies(
            "<p>Body</p>",
            "https://example.com/provider-cancel",
            dependencies=dependencies,
            strategy_order=["llm"],
            llm_settings={"provider": "openai"},
        )

    assert provider_calls == 1


@pytest.mark.parametrize("llm_response", ['{"score": 0}', '{"flag": false}'])
def test_scalar_llm_data_allows_trafilatura_fallback(
    monkeypatch: pytest.MonkeyPatch,
    llm_response: str,
) -> None:
    calls: list[str] = []
    dependencies = _install_dependencies(
        monkeypatch,
        llm_strategy,
        perform_chat_api_call=lambda **_kwargs: calls.append("llm") or _response(llm_response),
    )
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)

    def fallback(_html: str, url: str) -> dict[str, Any]:
        calls.append("fallback")
        return {
            "url": url,
            "title": "Fallback",
            "author": "N/A",
            "content": "Fallback body",
            "date": "N/A",
            "extraction_successful": True,
        }

    result = legacy.extract_article_with_pipeline(
        "<p>Body</p>",
        "https://example.com/scalar",
        strategy_order=["llm", "trafilatura"],
        llm_settings={"provider": "openai"},
        fallback_extractor=fallback,
    )

    assert calls == ["llm", "fallback"]
    assert result["extraction_strategy"] == "trafilatura"
    assert result["content"] == "Fallback body"


@pytest.mark.parametrize(
    ("schema_rules", "expected"),
    [
        ({"fields": [], "title_selector": "h1"}, []),
        ({"baseFields": [], "title_selector": "h1"}, []),
        ({"fields": {"headline": {"type": "headline"}}, "title_selector": "h1"}, [{"name": "title", "type": "text"}]),
        (
            {
                "baseFields": {"container": {"type": "container"}},
                "fields": {"headline": {"type": "headline"}},
                "title_selector": "h1",
            },
            [{"name": "container", "type": "container"}, {"name": "headline", "type": "headline"}],
        ),
    ],
)
def test_schema_rules_to_field_specs_preserves_predecessor_branching(
    schema_rules: dict[str, Any],
    expected: list[dict[str, str]],
) -> None:
    assert llm_strategy.schema_rules_to_field_specs(schema_rules) == expected


@pytest.mark.parametrize(
    ("module", "function_name", "stage", "error_field"),
    [
        (llm_strategy, "extract_llm_entities", "llm_extraction", "llm_error"),
        (schema_strategy, "generate_schema_rules_from_llm", "schema_generation", "error"),
        (schema_strategy, "generate_regex_pattern_from_llm", "regex_generation", "error"),
    ],
)
def test_provider_failure_is_sanitized_in_results_and_structured_logs(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    function_name: str,
    stage: str,
    error_field: str,
) -> None:
    def provider(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(SECRET)

    _install_dependencies(monkeypatch, module, perform_chat_api_call=provider)
    records: list[Any] = []
    handler_id = logger.add(records.append, level="WARNING")
    try:
        result = getattr(extraction, function_name)(
            "<p>Body</p>",
            "https://user:password@example.com/private?api_key=secret",
            llm_settings={"provider": "openai", "api_key": "top-secret"},
        )
    finally:
        logger.remove(handler_id)

    assert result[error_field] == "provider_error"
    assert SECRET not in str(result)
    assert "top-secret" not in str(result)
    provider_record = next(
        record.record for record in records if record.record["message"] == "LLM provider call failed"
    )
    provider_fields = {key: provider_record["extra"][key] for key in ("code", "exception_class", "stage", "host")}
    assert provider_fields == {
        "code": "provider_error",
        "exception_class": "RuntimeError",
        "stage": stage,
        "host": "example.com",
    }
    assert provider_record["message"] == "LLM provider call failed"
    assert SECRET not in str(provider_record)
    assert "top-secret" not in str(provider_record)


def test_schema_generator_uses_dependency_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    validations: list[tuple[dict[str, Any], str]] = []

    def validate(rules: dict[str, Any], *, html_text: str) -> dict[str, Any]:
        validations.append((rules, html_text))
        return {"errors": [], "warnings": []}

    _install_dependencies(
        monkeypatch,
        schema_strategy,
        perform_chat_api_call=lambda **_kwargs: _response('{"schema": {"fields": []}}'),
        validate_selector_rules=validate,
    )

    result = extraction.generate_schema_rules_from_llm(
        "<p>Body</p>",
        "https://example.com/schema",
        llm_settings={"provider": "openai", "strict_json": True},
    )

    assert result["success"] is True
    assert validations == [({"fields": []}, "<p>Body</p>")]


def test_pipeline_trace_receives_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def provider(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(SECRET)

    dependencies = _install_dependencies(monkeypatch, llm_strategy, perform_chat_api_call=provider)
    monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)

    result = legacy.extract_article_with_pipeline(
        "<p>Body</p>",
        "https://example.com/pipeline",
        strategy_order=["llm"],
        llm_settings={"provider": "openai"},
    )

    assert result["llm_error"] == "provider_error"
    assert result["extraction_trace"] == [
        {
            "strategy": "llm",
            "status": "failed",
            "reason": "llm_no_content",
            "detail": "provider_error",
        }
    ]
    assert SECRET not in str(result)


@pytest.mark.parametrize(
    ("function_name", "args", "expected"),
    [
        ("generate_schema_rules_from_llm", ("", "https://example.com"), "schema_llm_empty_html"),
        ("generate_regex_pattern_from_llm", ("", "https://example.com"), "regex_llm_empty_html"),
    ],
)
def test_generator_empty_html_codes_are_preserved(
    function_name: str,
    args: tuple[str, str],
    expected: str,
) -> None:
    assert getattr(extraction, function_name)(*args)["error"] == expected


def test_neutral_observability_is_bounded_and_has_no_upward_imports() -> None:
    from tldw_Server_API.app.core.Web_Scraping.observability import (
        bounded_code,
        bounded_stage,
        sanitized_host,
    )

    assert sanitized_host(SECRET) == "example.com"
    assert sanitized_host("https://example.com\\@secret.test/private") == "unknown"
    assert bounded_stage("schema_generation") == "schema_generation"
    assert bounded_stage(SECRET) == "runtime"
    assert bounded_code("provider_error") == "provider_error"
    assert bounded_code(SECRET) == "other"

    path = WEB_SCRAPING_ROOT / "observability.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {
        "Article_Extractor_Lib",
        "Watchlists",
        "WebSearch",
        "enhanced_web_scraping",
        "extraction",
        "orchestration",
        "playwright",
        "policy",
        "preflight",
        "routing",
        "scraper_router",
    }
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert [name for name in imports if set(name.split(".")) & forbidden] == []


def test_strategy_modules_do_not_import_chat_provider_or_upward_layers() -> None:
    forbidden = {
        "Article_Extractor_Lib",
        "Chat",
        "Watchlists",
        "WebSearch",
        "enhanced_web_scraping",
        "orchestration",
        "playwright",
        "policy",
        "preflight",
        "routing",
        "scraper_router",
    }
    violations: list[tuple[str, str]] = []
    for filename in ("llm.py", "schema.py"):
        path = WEB_SCRAPING_ROOT / "extraction" / "strategies" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            violations.extend((filename, name) for name in imports if set(name.split(".")) & forbidden)
    assert violations == []
