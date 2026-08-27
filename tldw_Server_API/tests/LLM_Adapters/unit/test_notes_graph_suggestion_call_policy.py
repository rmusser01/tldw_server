from __future__ import annotations

import asyncio
import io
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.LLM_Calls import adapter_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.Notes_Graph import suggestion_generation
from tldw_Server_API.app.core.Notes_Graph.suggestion_capabilities import (
    ProviderCapabilityContract,
    SuggestionCapabilityLimits,
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
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import RetrievalResult

pytestmark = pytest.mark.unit


def _prepared(*, limits: SuggestionCapabilityLimits | None = None):
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
        limits=limits or SuggestionCapabilityLimits(),
    )


def _provider(**overrides: object) -> GenerationProvider:
    values: dict[str, object] = {
        "adapter": "openai",
        "model": "gpt-test",
        "api_key": "PRIVATE-CREDENTIAL",
        "app_config": {
            "openai_api": {
                "api_retries": 9,
                "api_base_url": "https://api.openai.com/v1",
            }
        },
        "provider_capabilities": {"supports_json_schema": True},
        "endpoint_url": "https://api.openai.com/v1",
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
        policy.maximum_timeout_seconds,
        policy.required_endpoint_scope.matches("https://api.openai.com/v1/chat/completions"),
    ) == (1, False, False, False, True, 1, True, 120, True)


@pytest.mark.asyncio
async def test_generation_returns_bounded_provider_usage_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"relationships":[],"tags":[]}'}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45},
        }

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)

    result = await generate_suggestions_once(prepared=_prepared(), provider=_provider())

    assert (result.input_tokens, result.output_tokens) == (123, 45)


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
    "response",
    [
        {"choices": []},
        {
            "choices": [
                {"message": {"content": '{"relationships":[],"tags":[]}'}},
                {"message": {"content": '{"relationships":[],"tags":[]}'}},
            ]
        },
        {
            "choices": [],
            "content": '{"relationships":[],"tags":[]}',
        },
    ],
)
async def test_zero_multiple_and_ambiguous_choice_responses_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
) -> None:
    async def fake_call(**_kwargs: Any) -> dict[str, Any]:
        return response

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)

    with pytest.raises(SuggestionGenerationError) as exc_info:
        await generate_suggestions_once(prepared=_prepared(), provider=_provider())

    assert exc_info.value.code == "notes_graph_suggestion_invalid_model_output"


@pytest.mark.asyncio
async def test_unambiguous_non_choice_content_response_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(**_kwargs: Any) -> dict[str, Any]:
        return {"content": '{"relationships":[],"tags":[]}'}

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)

    result = await generate_suggestions_once(prepared=_prepared(), provider=_provider())

    assert result.relationships == ()
    assert result.tags == ()


@pytest.mark.asyncio
async def test_configured_lower_limits_reach_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    limits = SuggestionCapabilityLimits(
        max_candidates=2,
        max_relationships=1,
        max_tags=2,
        max_new_tags=1,
        max_tag_catalog=3,
        max_estimated_input_tokens=2_000,
        max_output_tokens=500,
        provider_timeout_seconds=30,
        response_candidates=1,
    )

    async def fake_call(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"relationships":[],"tags":[]}'}}]}

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)

    await generate_suggestions_once(
        prepared=_prepared(limits=limits),
        provider=_provider(),
    )

    assert calls[0]["max_tokens"] == 500
    assert calls[0]["timeout"] == 30
    assert calls[0]["n"] == 1
    assert calls[0]["call_policy"].maximum_timeout_seconds == 30


@pytest.mark.asyncio
async def test_unsupported_transport_contract_fails_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_call(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(suggestion_generation, "perform_chat_api_call_async", fake_call)
    with pytest.raises(SuggestionGenerationError) as exc_info:
        await generate_suggestions_once(
            prepared=_prepared(),
            provider=_provider(adapter="anthropic"),
        )

    assert exc_info.value.code == "notes_graph_provider_call_policy_unsupported"
    assert calls == 0


def _install_openai_adapter(
    monkeypatch: pytest.MonkeyPatch,
    adapter: OpenAIAdapter,
) -> None:
    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("openai", adapter)
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: registry)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [307, 308])
async def test_actual_adapter_follows_only_same_origin_redirects(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    requests: list[str] = []
    endpoint = "http://127.0.0.1:18071/v1"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                status_code,
                headers={"location": "/v1/redirected"},
                request=request,
            )
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"relationships":[],"tags":[]}'}}]},
        )

    adapter = OpenAIAdapter()
    _install_openai_adapter(monkeypatch, adapter)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter.http_fetcher = lambda **kwargs: http_client.fetch(client=client, **kwargs)
        result = await generate_suggestions_once(
            prepared=_prepared(),
            provider=_provider(
                endpoint_url=endpoint,
                app_config={"openai_api": {"api_base_url": endpoint}},
            ),
        )

    assert result.relationships == ()
    assert requests == [
        f"{endpoint}/chat/completions",
        "http://127.0.0.1:18071/v1/redirected",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [307, 308])
async def test_actual_adapter_rejects_cross_origin_redirect_before_second_request(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    requests: list[str] = []
    endpoint = "http://127.0.0.1:18072/v1"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            status_code,
            headers={"location": "http://127.0.0.1:18073/stolen"},
            request=request,
        )

    adapter = OpenAIAdapter()
    _install_openai_adapter(monkeypatch, adapter)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter.http_fetcher = lambda **kwargs: http_client.fetch(client=client, **kwargs)
        with pytest.raises(SuggestionGenerationError) as exc_info:
            await generate_suggestions_once(
                prepared=_prepared(),
                provider=_provider(
                    endpoint_url=endpoint,
                    app_config={"openai_api": {"api_base_url": endpoint}},
                ),
            )

    assert exc_info.value.code == "notes_graph_provider_call_failed"
    assert requests == [f"{endpoint}/chat/completions"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_timeout", "expected_timeout"),
    [(30, 30.0), (999, 120.0)],
)
async def test_actual_adapter_clamps_configured_timeout_to_effective_limit(
    monkeypatch: pytest.MonkeyPatch,
    configured_timeout: int,
    expected_timeout: float,
) -> None:
    captured_timeouts: list[float] = []
    endpoint = "http://127.0.0.1:18074/v1"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"relationships":[],"tags":[]}'}}]},
        )

    adapter = OpenAIAdapter()
    _install_openai_adapter(monkeypatch, adapter)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:

        def fetcher(**kwargs: Any) -> httpx.Response:
            captured_timeouts.append(float(kwargs["timeout"]))
            return http_client.fetch(client=client, **kwargs)

        adapter.http_fetcher = fetcher
        await generate_suggestions_once(
            prepared=_prepared(),
            provider=_provider(
                endpoint_url=endpoint,
                app_config={
                    "openai_api": {
                        "api_base_url": endpoint,
                        "api_timeout": configured_timeout,
                    }
                },
            ),
        )

    assert captured_timeouts == [expected_timeout]


@pytest.mark.asyncio
async def test_outer_wall_clock_deadline_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = OpenAIAdapter()
    adapter.async_chat_is_native = True

    async def slow_chat(_request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        await asyncio.sleep(2)
        return {}

    adapter.achat = slow_chat
    _install_openai_adapter(monkeypatch, adapter)

    with pytest.raises(SuggestionGenerationError) as exc_info:
        await generate_suggestions_once(
            prepared=_prepared(limits=SuggestionCapabilityLimits(provider_timeout_seconds=1)),
            provider=_provider(),
        )

    assert exc_info.value.code == "notes_graph_provider_call_failed"
    assert exc_info.value.__cause__ is None


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
    provider = _provider(adapter="anthropic")
    capability = build_suggestion_capabilities(
        ProviderCapabilityContract(
            adapter=provider.adapter,
            model=provider.model,
            endpoint_url="https://example.test/v1",
            call_policy=suggestion_generation.build_provider_call_policy(
                allow_response_format=True,
                endpoint_url="https://example.test/v1",
            ),
            data_boundary="unknown",
            credentials_available=True,
            provider_healthy=True,
        )
    )

    assert capability.generation_available is False
    assert capability.unavailable_reason == "notes_graph_provider_call_policy_unsupported"


def test_generation_endpoint_binding_must_be_explicit() -> None:
    with pytest.raises(TypeError):
        GenerationProvider(adapter="openai", model="gpt-test")


def test_generation_transport_safety_cannot_be_self_attested() -> None:
    with pytest.raises(TypeError):
        _provider(supports_one_attempt=True)

    with pytest.raises(TypeError):
        _provider(enforces_same_origin_redirects=True)
