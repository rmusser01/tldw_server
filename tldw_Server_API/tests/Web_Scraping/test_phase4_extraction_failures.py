import ast
import asyncio
import dataclasses
import inspect
from pathlib import Path
from typing import Any, Optional

import pytest
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import extraction
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    ExtractionDependencies,
    build_default_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm as llm_strategy
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import schema as schema_strategy

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_SCRAPING_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping"
SECRET = "https://user:api-key@example.com/private/path?token=secret#payload"


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


def test_llm_merges_concatenated_json_and_aggregates_chunk_usage(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert result["llm_extraction"]["blocks"] == [{"text": "First"}]
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
        if checkpoints == 3:
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
        if checkpoints == 3:
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

    _install_dependencies(monkeypatch, llm_strategy, perform_chat_api_call=provider)

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
