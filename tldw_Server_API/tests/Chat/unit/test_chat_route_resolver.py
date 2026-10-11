from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision


@pytest.fixture
def router_dependencies():
    captured: dict[str, object] = {}

    def configured_providers():
        return {
            "providers": [
                {
                    "name": "openai",
                    "models_info": [
                        {
                            "name": "gpt-test",
                            "tool_support": True,
                            "quality_rank": 1,
                        }
                    ],
                }
            ],
            "default_provider": "openai",
        }

    def route_model(**kwargs):
        captured["router_request"] = kwargs["request"]
        return RoutingDecision(
            provider="openai",
            model="gpt-test",
            canonical=True,
            decision_source="rules_router",
        )

    return {
        "request": SimpleNamespace(state=SimpleNamespace(user_id="user-1", api_key_id="key-1")),
        "sticky_store": SimpleNamespace(),
        "current_user": SimpleNamespace(id="user-1", id_int=1),
        "request_id": "request-1",
        "configured_providers_getter": configured_providers,
        "apply_provider_overrides_fn": lambda listing: listing,
        "priority_resolver": lambda *_args: None,
        "route_model_fn": route_model,
        "default_provider": "openai",
        "captured": captured,
    }


@pytest.mark.asyncio
async def test_auto_route_matches_chat_router_decision(router_dependencies):
    from tldw_Server_API.app.core.Chat.chat_route_resolver import resolve_chat_route

    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Improve this prompt")],
        stream=False,
    )

    route = await resolve_chat_route(
        request,
        **{key: value for key, value in router_dependencies.items() if key != "captured"},
    )

    assert route.provider == "openai"
    assert route.model == "gpt-test"
    assert route.was_auto is True
    assert route.debug["candidate_count"] == 1
    assert router_dependencies["captured"]["router_request"].latest_user_turn == "Improve this prompt"


@pytest.mark.asyncio
async def test_concrete_route_uses_the_requested_provider_and_model(router_dependencies):
    from tldw_Server_API.app.core.Chat.chat_route_resolver import resolve_chat_route

    request = ChatCompletionRequest(
        model="gpt-test",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Improve this prompt")],
        stream=False,
    )

    route = await resolve_chat_route(
        request,
        **{key: value for key, value in router_dependencies.items() if key != "captured"},
    )

    assert route.provider == "openai"
    assert route.model == "gpt-test"
    assert route.was_auto is False
    assert route.routing_decision is None
    assert route.debug["requested_capabilities"] == {
        "tools": False,
        "vision": False,
        "json_mode": False,
        "reasoning": False,
    }


@pytest.mark.asyncio
async def test_auto_route_raises_a_typed_error_when_no_candidate_is_available(router_dependencies):
    from tldw_Server_API.app.core.Chat.chat_route_resolver import (
        ChatRouteResolutionError,
        resolve_chat_route,
    )

    router_dependencies["configured_providers_getter"] = lambda: {
        "providers": [],
        "default_provider": "openai",
    }
    router_dependencies["route_model_fn"] = lambda **_kwargs: None
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Improve this prompt")],
        stream=False,
    )

    with pytest.raises(ChatRouteResolutionError) as exc_info:
        await resolve_chat_route(
            request,
            **{key: value for key, value in router_dependencies.items() if key != "captured"},
        )

    assert exc_info.value.code == "auto_routing_no_candidates"
    assert exc_info.value.debug["candidate_count"] == 0


@pytest.mark.asyncio
async def test_auto_route_raises_a_typed_error_when_router_returns_no_decision(router_dependencies):
    from tldw_Server_API.app.core.Chat.chat_route_resolver import (
        ChatRouteResolutionError,
        resolve_chat_route,
    )

    router_dependencies["route_model_fn"] = lambda **_kwargs: None
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Improve this prompt")],
        stream=False,
    )

    with pytest.raises(ChatRouteResolutionError) as exc_info:
        await resolve_chat_route(
            request,
            **{key: value for key, value in router_dependencies.items() if key != "captured"},
        )

    assert exc_info.value.code == "auto_routing_failed"
    assert exc_info.value.debug["candidate_count"] == 1
    assert exc_info.value.debug["policy"]["server_default_provider"] == "openai"


@pytest.mark.asyncio
async def test_auto_route_requires_an_injected_provider_listing(router_dependencies):
    from tldw_Server_API.app.core.Chat.chat_route_resolver import resolve_chat_route

    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Improve this prompt")],
        stream=False,
    )
    dependencies = {
        key: value
        for key, value in router_dependencies.items()
        if key not in {"captured", "configured_providers_getter"}
    }

    with pytest.raises(TypeError, match="configured_providers_getter"):
        await resolve_chat_route(request, **dependencies)


@pytest.mark.asyncio
async def test_router_call_is_marked_self_recorded():
    """The router's provider call runs with records_itself=True (usage accounting spec §3)."""
    from tldw_Server_API.app.core.Chat.chat_route_resolver import _select_auto_llm_router_choice
    from tldw_Server_API.app.core.LLM_Calls.routing.models import RouterRequest, RoutingPolicy
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

    seen: list[bool] = []

    async def _probe(**kwargs):
        seen.append(current_llm_call_context().records_itself)
        return {
            "choices": [{"message": {"content": '{"provider":"openai","model":"gpt-test"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def _byok(*args, **kwargs):
        async def _touch():
            return None

        return SimpleNamespace(api_key="k", app_config={}, touch_last_used=_touch)

    async def _log(**kwargs):
        return None

    await _select_auto_llm_router_choice(
        router_request=RouterRequest(model="auto", surface="chat", latest_user_turn="route me", scope="chat-1"),
        policy=RoutingPolicy(
            request_model="auto", server_default_provider="openai", boundary_mode="server_default_provider"
        ),
        candidates=[
            {"provider": "openai", "model": "gpt-test"},
            {"provider": "openai", "model": "gpt-alt"},
        ],
        provider_listing={
            "default_provider": "openai",
            "providers": [
                {
                    "name": "openai",
                    "default_model": "router-model",
                    "is_configured": True,
                    "models_info": [{"name": "router-model"}],
                }
            ],
        },
        request=SimpleNamespace(state=SimpleNamespace(user_id="user-1", api_key_id="key-1")),
        current_user=SimpleNamespace(id="user-1", id_int=1),
        request_id="request-1",
        surface="chat",
        endpoint="POST:/api/v1/chat/completions",
        resolve_provider_api_key_fn=lambda *a, **k: ("k", {}),
        resolve_byok_credentials_fn=_byok,
        perform_chat_api_call_async_fn=_probe,
        log_model_router_usage_fn=_log,
    )

    assert seen == [True]
