"""The chat-adjacent LLM surfaces name themselves to the gateway (TASK-13950.27 PR E), and a refusal surfaces
as that surface's refusal rather than a fallback or an empty result."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.tests.Moderation.conftest import reply

pytestmark = pytest.mark.integration


class _Spy:
    """Records the surface the gateway sees; ``replies`` may hold a refusal to raise."""

    def __init__(self) -> None:
        self.surfaces: list[str | None] = []


@pytest.fixture
def spy(monkeypatch, stub_llm_registry):
    """A test pre-hook recording ``call.surface`` for every gateway call."""
    from tldw_Server_API.app.core.LLM_Calls import gateway

    recorder = _Spy()
    remove = gateway.add_policy_hooks(pre_call=lambda call: recorder.surfaces.append(call.surface))
    yield recorder
    remove()


def _provider_kwargs() -> dict:
    return {"api_endpoint": "e2e-fake", "api_key": "k", "model": "m", "messages_payload": [{"role": "user", "content": "hi"}]}


def _dialogue():
    from tldw_Server_API.app.core.Chat_Workflows.dialogue_orchestrator import ChatWorkflowDialogueOrchestrator

    orch = ChatWorkflowDialogueOrchestrator()
    return asyncio.run(orch._call_text_model(
        selection={"provider": "e2e-fake", "model": "m"}, system_prompt="s", prompt="hi", user_identifier="u"))


def _branch():
    from tldw_Server_API.app.core.Chat_Macros.branch_runner import ChatMacroLLMBranchRunner

    snapshot = SimpleNamespace(messages=[])
    return asyncio.run(ChatMacroLLMBranchRunner().run_branch(
        prompt="p", snapshot=snapshot, model_selection={"provider": "e2e-fake", "model": "m"}))


def _resolver():
    from tldw_Server_API.app.core.Chat import chat_route_resolver as r
    import inspect

    default = inspect.signature(r.resolve_chat_route).parameters["perform_chat_api_call_async_fn"].default
    return asyncio.run(default(**_provider_kwargs()))


SITES = {
    "chat.prompt_improvement": _resolver,
    "chat.macros": _branch,
    "chat.dialogue_workflows": _dialogue,
}


@pytest.mark.parametrize("surface", sorted(SITES))
def test_surface_reaches_gateway(spy, surface):
    try:
        SITES[surface]()
    except Exception:  # noqa: BLE001 - only the surface the gateway saw matters here
        pass
    assert spy.surfaces == [surface]


def test_surface_reaches_gateway_document_generation(spy):
    from tldw_Server_API.app.core.Chat import document_generator as dg

    gen = dg.DocumentGeneratorService.__new__(dg.DocumentGeneratorService)
    try:
        gen._call_llm(provider="e2e-fake", model="m", api_key="k", app_config=None, system_prompt="s", user_prompt="u",
                      temperature=0.1, max_tokens=5, stream=False)
    except Exception:  # noqa: BLE001
        pass
    assert spy.surfaces == ["chat.document_generation"]


def test_surface_reaches_gateway_memory_extraction(spy):
    from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import extract_character_memories

    extract_character_memories([{"sender": "user", "content": "hi"}], "A", "B", [], api_endpoint="e2e-fake", api_key="k", model="m")
    assert spy.surfaces == ["character.memory_extraction"]


def test_prompt_improvement_router_block_propagates(moderation_hooks, stub_llm_registry):
    """The router prompt is refused: the resolver raises the refusal, it does not fall back to a rules choice."""
    from tldw_Server_API.app.core.Chat import chat_route_resolver as r
    from tldw_Server_API.app.core.LLM_Calls.routing import RouterRequest, RoutingPolicy

    policy = RoutingPolicy(request_model="auto", server_default_provider="e2e-fake", boundary_mode="server_default_provider")
    listing = {"default_provider": "e2e-fake", "providers": [{"name": "e2e-fake", "default_model": "r", "models": ["a", "b"]}]}
    request = RouterRequest(model="auto", surface="prompt_improvement", latest_user_turn="forbidden", scope=None, requested_capabilities={})

    async def byok(*_a, **_k):
        return SimpleNamespace(api_key="k", app_config=None, source="env", touch_last_used=_noop)

    async def _noop():
        return None

    async def go():
        import inspect
        default = inspect.signature(r.resolve_chat_route).parameters["perform_chat_api_call_async_fn"].default
        return await r._select_auto_llm_router_choice(
            router_request=request, policy=policy, candidates=[{"provider": "e2e-fake", "model": "a"}, {"provider": "e2e-fake", "model": "b"}],
            provider_listing=listing, request=None, current_user=SimpleNamespace(id=1), request_id=None, surface="prompt_improvement",
            endpoint="e", resolve_provider_api_key_fn=lambda *a, **k: ("k", {}), resolve_byok_credentials_fn=byok,
            perform_chat_api_call_async_fn=default, log_model_router_usage_fn=_noop)

    with pytest.raises(LLMPolicyRefusal):
        asyncio.run(go())
    assert stub_llm_registry.requests == []


def test_prompt_improvement_content_filter_not_refallen_back(moderation_hooks, stub_llm_registry):
    """A router answer the output hook blocked (content_filter) gives no choice and is not asked again.

    The resolver has no provider-fallback loop, so one request is the whole story; this pins that."""
    from tldw_Server_API.app.core.Chat import chat_route_resolver as r
    from tldw_Server_API.app.core.LLM_Calls.routing import RouterRequest, RoutingPolicy

    stub_llm_registry.reply = reply("forbidden")
    policy = RoutingPolicy(request_model="auto", server_default_provider="e2e-fake", boundary_mode="server_default_provider")
    listing = {"default_provider": "e2e-fake", "providers": [{"name": "e2e-fake", "default_model": "r", "models": ["a", "b"]}]}
    request = RouterRequest(model="auto", surface="prompt_improvement", latest_user_turn="hello", scope=None, requested_capabilities={})

    async def _noop(*_a, **_k):
        return None

    async def byok(*_a, **_k):
        return SimpleNamespace(api_key="k", app_config=None, source="env", touch_last_used=_noop)

    async def go():
        import inspect
        default = inspect.signature(r.resolve_chat_route).parameters["perform_chat_api_call_async_fn"].default
        return await r._select_auto_llm_router_choice(
            router_request=request, policy=policy, candidates=[{"provider": "e2e-fake", "model": "a"}, {"provider": "e2e-fake", "model": "b"}],
            provider_listing=listing, request=None, current_user=SimpleNamespace(id=1), request_id=None, surface="prompt_improvement",
            endpoint="e", resolve_provider_api_key_fn=lambda *a, **k: ("k", {}), resolve_byok_credentials_fn=byok,
            perform_chat_api_call_async_fn=default, log_model_router_usage_fn=_noop)

    choice, _debug = asyncio.run(go())
    assert choice is None
    assert len(stub_llm_registry.requests) == 1


@pytest.fixture
def dispatch(monkeypatch):
    """The prompt-improvement dispatch module with routing and credentials resolved locally to the e2e-fake provider."""
    from tldw_Server_API.app.core.Prompt_Management import prompt_improvement_dispatch as module

    async def _noop():
        return None

    async def route(*_a, **_k):
        return SimpleNamespace(provider="e2e-fake", model="m")

    async def byok(*_a, **_k):
        return SimpleNamespace(api_key="k", app_config=None, source="env", touch_last_used=_noop)

    monkeypatch.setattr(module, "resolve_chat_route", route)
    monkeypatch.setattr(module, "resolve_byok_credentials", byok)
    monkeypatch.setattr(module, "is_model_known_for_provider", lambda *_a: True)
    monkeypatch.setattr(module, "provider_requires_api_key", lambda *_a: False)
    return module


def test_surface_reaches_gateway_prompt_improvement_dispatch(spy, dispatch):
    """The dispatch's own provider call (not the router's) names the surface."""
    try:
        asyncio.run(dispatch.dispatch_prompt_improvement(
            request=None, current_user=SimpleNamespace(id=1), routing_decision_store=None, selected_model="openai/gpt-4o-mini",
            provider_hint=None, messages=[{"role": "user", "content": "hi"}], request_id="r",
            configured_providers_getter=lambda: {"default_provider": "e2e-fake"}))
    except Exception:  # noqa: BLE001 - only the surface matters
        pass
    assert spy.surfaces == ["chat.prompt_improvement"]


def test_prompt_improvement_dispatch_input_block_propagates(moderation_hooks, stub_llm_registry, dispatch):
    """The dispatch's broad except maps provider errors; a refusal stays a 400."""
    with pytest.raises(LLMPolicyRefusal):
        asyncio.run(dispatch.dispatch_prompt_improvement(
            request=None, current_user=SimpleNamespace(id=1), routing_decision_store=None, selected_model="openai/gpt-4o-mini",
            provider_hint=None, messages=[{"role": "user", "content": "forbidden"}], request_id="r",
            configured_providers_getter=lambda: {"default_provider": "e2e-fake"}))
    assert stub_llm_registry.requests == []


def test_shared_workspace_refusal_maps_to_400():
    """The service turns a refusal into its own error (not ``generation_failed``, 503) and the route answers 400."""
    from tldw_Server_API.app.api.v1.endpoints import sharing
    from tldw_Server_API.app.core.exceptions import SharedWorkspaceGenerationRefused

    error = sharing._shared_chat_service_http_error(SharedWorkspaceGenerationRefused())
    assert error.status_code == 400
    assert error.detail["code"] == "llm_policy_refused"
    assert error.detail["retryable"] is False


def test_surface_reaches_gateway_shared_workspace_chat():
    """The call names its surface; the file's one gateway call is not reachable without a full workspace."""
    import ast
    import inspect

    from tldw_Server_API.app.core.Sharing import shared_workspace_chat_service as mod

    calls = [n for n in ast.walk(ast.parse(inspect.getsource(mod))) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "perform_chat_api_call_async"]
    assert [k.value.value for c in calls for k in c.keywords if k.arg == "llm_surface"] == ["sharing.workspace_chat"]
