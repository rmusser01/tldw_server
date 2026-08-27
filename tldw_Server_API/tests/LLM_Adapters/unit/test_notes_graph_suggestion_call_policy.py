from __future__ import annotations

import io
from dataclasses import replace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Notes_Graph import suggestion_generation
from tldw_Server_API.app.core.Notes_Graph.suggestion_capabilities import (
    ProviderCapabilityContract,
    build_suggestion_capabilities,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import (
    content_fingerprint,
    split_evidence_windows,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_generation import (
    GenerationProvider,
    SuggestionGenerationError,
    build_generation_request,
    generate_suggestions_once,
    validate_redirect_origin,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import RetrievalResult

pytestmark = pytest.mark.unit


def _prepared():
    title = "Private title"
    content = "Private source evidence"
    retrieval = RetrievalResult(
        source_note_id="source-note",
        source_fingerprint=content_fingerprint(title, content),
        source_windows=split_evidence_windows(
            note_id="source-note",
            title=title,
            content=content,
            max_windows=4,
            max_code_points=480,
        ),
        terms=("private", "source"),
        candidates=(),
        tag_catalog=(),
        backend_overfetch_count=0,
        excluded_oversized_candidate_count=0,
        projection_fresh=True,
        estimated_input_tokens=10,
    )
    return build_generation_request(
        retrieval=retrieval,
        source_title=title,
        source_content=content,
    )


def _provider(**overrides: object) -> GenerationProvider:
    values: dict[str, object] = {
        "adapter": "openai",
        "model": "gpt-test",
        "api_key": "PRIVATE-CREDENTIAL",
        "app_config": {"openai_api": {"api_retries": 9}},
        "provider_capabilities": {"supports_json_schema": True},
        "supports_one_attempt": True,
        "enforces_same_origin_redirects": True,
    }
    values.update(overrides)
    return GenerationProvider(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_generation_uses_exactly_one_bounded_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"relationships":[],"tags":[]}'}}]}

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)
    result = await generate_suggestions_once(prepared=_prepared(), provider=_provider())

    assert result.relationships == ()
    assert len(calls) == 1
    call = calls[0]
    assert call["api_endpoint"] == "openai"
    assert call["model"] == "gpt-test"
    assert call["max_tokens"] == 2_000
    assert call["timeout"] == 120
    assert call["streaming"] is False
    assert call["tools"] is None
    assert call["stop"] is None
    assert call["n"] == 1
    assert call["response_format"]["type"] == "json_schema"
    policy = call["call_policy"]
    assert (
        policy.max_transport_attempts,
        policy.allow_streaming,
        policy.allow_tools,
        policy.allow_stop,
        policy.allow_response_format,
        policy.candidate_count,
        policy.privacy_safe_errors,
    ) == (1, False, False, False, True, 1, True)


@pytest.mark.asyncio
async def test_structured_mode_is_capability_dependent_but_local_json_is_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)
    provider = _provider(provider_capabilities={})

    with pytest.raises(SuggestionGenerationError) as exc_info:
        await generate_suggestions_once(prepared=_prepared(), provider=provider)

    assert exc_info.value.code == "notes_graph_suggestion_invalid_model_output"
    assert "response_format" not in calls[0]
    assert calls[0]["call_policy"].allow_response_format is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "reason"),
    [
        (
            _provider(supports_one_attempt=False),
            "notes_graph_provider_retry_policy_unsupported",
        ),
        (
            _provider(enforces_same_origin_redirects=False),
            "notes_graph_provider_redirect_policy_unsupported",
        ),
    ],
)
async def test_unsupported_transport_contract_fails_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    provider: GenerationProvider,
    reason: str,
) -> None:
    calls = 0

    async def fake_call(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)
    with pytest.raises(SuggestionGenerationError) as exc_info:
        await generate_suggestions_once(prepared=_prepared(), provider=provider)

    assert exc_info.value.code == reason
    assert calls == 0


def test_canonical_same_origin_redirects_only() -> None:
    validate_redirect_origin(
        "HTTPS://Example.test:443/v1/chat/completions",
        "https://example.test/redirected",
    )

    with pytest.raises(SuggestionGenerationError) as exc_info:
        validate_redirect_origin(
            "https://example.test/v1/chat/completions",
            "https://other.example.test/v1/chat/completions",
        )
    assert exc_info.value.code == "notes_graph_provider_cross_origin_redirect"


@pytest.mark.asyncio
async def test_provider_failures_do_not_expose_prompt_credentials_endpoint_or_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_values = (
        "PRIVATE-CREDENTIAL",
        "Private source evidence",
        "https://private-endpoint.example.test/v1",
        "PRIVATE-UPSTREAM-RESPONSE",
    )

    async def fake_call(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(" ".join(private_values))

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(SuggestionGenerationError) as exc_info:
            await generate_suggestions_once(
                prepared=_prepared(),
                provider=_provider(endpoint_url="https://private-endpoint.example.test/v1"),
            )
    finally:
        logger.remove(sink_id)

    rendered = f"{exc_info.value!s}\n{exc_info.value!r}\n{sink.getvalue()}"
    assert exc_info.value.code == "notes_graph_provider_call_failed"
    assert exc_info.value.__cause__ is None
    assert all(value not in rendered for value in private_values)


def test_capability_and_invocation_transport_preflights_agree() -> None:
    provider = _provider(supports_one_attempt=False)
    capability = build_suggestion_capabilities(
        ProviderCapabilityContract(
            adapter=provider.adapter,
            model=provider.model,
            endpoint_url="https://example.test/v1",
            call_policy=suggestion_generation.build_provider_call_policy(allow_response_format=True),
            data_boundary="unknown",
            supports_one_attempt=provider.supports_one_attempt,
            enforces_same_origin_redirects=provider.enforces_same_origin_redirects,
            credentials_available=True,
            provider_healthy=True,
        )
    )

    assert capability.generation_available is False
    assert capability.unavailable_reason == "notes_graph_provider_retry_policy_unsupported"
    assert replace(provider, supports_one_attempt=True).supports_one_attempt is True


def test_generation_transport_proofs_must_be_explicit() -> None:
    with pytest.raises(TypeError):
        GenerationProvider(adapter="openai", model="gpt-test")
