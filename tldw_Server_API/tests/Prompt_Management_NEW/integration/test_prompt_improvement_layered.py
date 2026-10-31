"""Layered integration coverage for prompt-improvement infrastructure wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.Chat.chat_route_resolver import resolve_chat_route
from tldw_Server_API.app.core.LLM_Calls.routing import InMemoryRoutingDecisionStore
from tldw_Server_API.app.core.Prompt_Management.prompt_improvement_dispatch import (
    PromptImprovementDispatchError,
    dispatch_prompt_improvement,
)

pytestmark = pytest.mark.integration


def _provider_listing() -> dict[str, Any]:
    return {
        "default_provider": "openai",
        "providers": [
            {
                "name": "openai",
                "is_configured": True,
                "models_info": [
                    {"name": "gpt-best", "quality_rank": 1},
                    {"name": "gpt-other", "quality_rank": 20},
                ],
            },
            {
                "name": "anthropic",
                "is_configured": True,
                "models_info": [
                    {"name": "claude-best", "quality_rank": 1},
                    {"name": "claude-other", "quality_rank": 20},
                ],
            },
            {
                "name": "moonshot",
                "is_configured": True,
                "models_info": [{"name": "kimi-k2", "quality_rank": 1}],
            },
        ],
    }


def _identity_provider_listing(listing: dict[str, Any]) -> dict[str, Any]:
    return listing


def _no_model_priority(*_args: Any) -> None:
    return None


def _request_context() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(user_id=None, api_key_id=None))


def _route_user() -> SimpleNamespace:
    return SimpleNamespace(id="layered-user", id_int=None)


def _auto_request(*, provider: str | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto",
        api_provider=provider,
        routing={"strategy": "rules_router", "mode": "sticky_session"},
        messages=[
            ChatCompletionUserMessageParam(
                role="user",
                content="Route this prompt improvement.",
            )
        ],
        stream=False,
    )


@pytest.mark.asyncio
async def test_real_resolver_uses_inventory_for_unpinned_auto_route() -> None:
    listing = _provider_listing()

    route = await resolve_chat_route(
        _auto_request(),
        request=_request_context(),
        sticky_store=InMemoryRoutingDecisionStore(),
        current_user=_route_user(),
        request_id="layered-unpinned",
        configured_providers_getter=lambda: listing,
        scope="prompt-unpinned",
        default_provider="openai",
        metrics_default_provider="openai",
        apply_provider_overrides_fn=_identity_provider_listing,
        priority_resolver=_no_model_priority,
    )

    assert (route.provider, route.model) == ("openai", "gpt-best")
    assert route.routing_decision is not None
    assert route.routing_decision.decision_source == "rules_router"
    assert route.debug["candidate_count"] == 2
    assert route.debug["policy"]["boundary_mode"] == "server_default_provider"


@pytest.mark.asyncio
async def test_real_resolver_pins_provider_and_reuses_sticky_decision() -> None:
    listing = _provider_listing()
    store = InMemoryRoutingDecisionStore()

    first = await resolve_chat_route(
        _auto_request(provider="anthropic"),
        request=_request_context(),
        sticky_store=store,
        current_user=_route_user(),
        request_id="layered-sticky-first",
        configured_providers_getter=lambda: listing,
        scope="prompt-sticky",
        default_provider="openai",
        metrics_default_provider="openai",
        apply_provider_overrides_fn=_identity_provider_listing,
        priority_resolver=_no_model_priority,
    )

    anthropic_models = listing["providers"][1]["models_info"]
    anthropic_models[0]["quality_rank"] = 99
    anthropic_models[1]["quality_rank"] = 1
    second = await resolve_chat_route(
        _auto_request(provider="anthropic"),
        request=_request_context(),
        sticky_store=store,
        current_user=_route_user(),
        request_id="layered-sticky-second",
        configured_providers_getter=lambda: listing,
        scope="prompt-sticky",
        default_provider="openai",
        metrics_default_provider="openai",
        apply_provider_overrides_fn=_identity_provider_listing,
        priority_resolver=_no_model_priority,
    )

    assert (first.provider, first.model) == ("anthropic", "claude-best")
    assert (second.provider, second.model) == ("anthropic", "claude-best")
    assert first.routing_decision is not None
    assert first.routing_decision.decision_source == "rules_router"
    assert second.routing_decision is not None
    assert second.routing_decision.decision_source == "sticky_reuse"
    assert second.debug["policy"]["boundary_mode"] == "pinned_provider"


class _SuccessResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"status":"no_change","improved_text":null,'
                            '"findings":[]}'
                        ),
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_dispatch_crosses_real_resolver_inventory_and_adapter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"posts": 0}

    class FakeSession:
        def post(self, _url, **kwargs):
            captured["posts"] += 1
            captured["payload"] = kwargs["json"]
            return _SuccessResponse()

        def close(self) -> None:
            return None

    def fake_session_factory(**kwargs):
        captured["retry"] = kwargs
        return FakeSession()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        fake_session_factory,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
        lambda: {"moonshot": "test-secret-boundary"},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.API_KEYS",
        {"moonshot": "test-secret-boundary"},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS",
        {"moonshot": "test-secret-boundary"},
    )
    messages = [
        {"role": "system", "content": "private meta instruction"},
        {"role": "user", "content": "private draft"},
    ]

    result = await dispatch_prompt_improvement(
        request=_request_context(),
        current_user=_route_user(),
        routing_decision_store=InMemoryRoutingDecisionStore(),
        selected_model="moonshot/kimi-k2",
        provider_hint=None,
        messages=messages,
        request_id="layered-dispatch",
        configured_providers_getter=_provider_listing,
    )

    assert (result.provider, result.model) == ("moonshot", "kimi-k2")
    assert captured["posts"] == 1
    assert captured["retry"]["total"] == 0
    payload = captured["payload"]
    assert payload["messages"] == messages
    assert payload["stream"] is False
    assert payload["n"] == 1
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.95
    assert "tools" not in payload
    assert "functions" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected_model",
    [
        "openai/not-a-real-catalog-model",
        "not-a-provider/ghost-model",
    ],
)
async def test_dispatch_rejects_unknown_provider_or_model_from_real_inventory(
    selected_model: str,
) -> None:
    with pytest.raises(PromptImprovementDispatchError) as captured:
        await dispatch_prompt_improvement(
            request=_request_context(),
            current_user=_route_user(),
            routing_decision_store=InMemoryRoutingDecisionStore(),
            selected_model=selected_model,
            provider_hint=None,
            messages=[{"role": "user", "content": "private draft"}],
            request_id="layered-unknown",
            configured_providers_getter=_provider_listing,
        )

    assert captured.value.code == "unsupported_model"
