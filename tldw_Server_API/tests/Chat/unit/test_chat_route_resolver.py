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
