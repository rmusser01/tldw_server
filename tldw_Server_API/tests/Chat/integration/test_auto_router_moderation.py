"""The chat auto-router is moderated at the gateway; a block is the caller's 400 (TASK-13950.27 PR C, part b)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls.routing import runtime
from tldw_Server_API.app.core.LLM_Calls.routing.models import RouterRequest, RoutingPolicy
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_client as persona_chat_client,
)
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_db as persona_chat_db,
)
from tldw_Server_API.tests.Moderation.conftest import make_service as make_service  # noqa: F401
from tldw_Server_API.tests.Moderation.conftest import reply, stub_llm_registry as stub_llm_registry  # noqa: F401

pytest_plugins = ("tldw_Server_API.tests.Chat.credential_runtime_fixtures",)
pytestmark = pytest.mark.integration

_LISTING = {
    "providers": [
        {
            "name": "openrouter",
            "default_model": "anthropic/claude-4.5-sonnet",
            "models_info": [
                {"name": "anthropic/claude-4.5-sonnet", "tool_support": True, "vision_support": True, "quality_rank": 20},
                {"name": "openai/gpt-4.1-mini", "tool_support": True, "vision_support": True, "quality_rank": 10},
            ],
        }
    ],
    "default_provider": "openrouter",
}
_CHOICE = '{"provider":"openrouter","model":"anthropic/claude-4.5-sonnet"}'


class _Manager:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.circuit_breakers: dict = {}

    def record_failure(self, provider, error) -> None:
        self.failures.append(provider)

    def record_success(self, provider, latency) -> None:
        pass

    def get_available_provider(self, exclude=None):
        return None


def _post(client, headers, content: str, manager: _Manager | None = None):
    dispatched: list[dict] = []

    async def fake_execute(**kwargs):
        dispatched.append(kwargs)
        return {"id": "x", "object": "chat.completion", "model": kwargs.get("model"),
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers", return_value=_LISTING),
        patch("tldw_Server_API.app.core.Chat.provider_manager.get_provider_manager", return_value=manager or _Manager()),
        patch("tldw_Server_API.app.core.Chat.chat_service.execute_non_stream_call", side_effect=fake_execute),
    ):
        response = client.post("/api/v1/chat/completions", headers=headers, json={
            "model": "auto", "api_provider": "openrouter", "messages": [{"role": "user", "content": content}],
        })
    return response, dispatched


def test_router_input_block_returns_400_without_record_failure(
    persona_chat_client, execution_scoped_provider_credentials, make_service, stub_llm_registry
):
    client, headers, _ = persona_chat_client
    # Chat's own check is off for the surface, so only the router's gateway call can block.
    make_service(enabled=True, **{"surface.chat.completions": False})
    manager = _Manager()
    response, dispatched = _post(client, headers, "this is forbidden", manager)
    assert response.status_code == 400, response.text
    assert stub_llm_registry.requests == [] and manager.failures == [] and dispatched == []


def test_router_receives_redacted_latest_turn(
    persona_chat_client, execution_scoped_provider_credentials, make_service, stub_llm_registry
):
    client, headers, _ = persona_chat_client
    make_service(enabled=True, **{"surface.chat.completions": False})
    stub_llm_registry.reply = reply(_CHOICE)
    response, _ = _post(client, headers, "my secret plan")
    assert response.status_code == 200, response.text
    sent = json.dumps(stub_llm_registry.requests[0]["messages"])
    assert "[X]" in sent and "secret" not in sent


def test_router_global_off_zero_moderation_calls(
    persona_chat_client, execution_scoped_provider_credentials, make_service, stub_llm_registry, monkeypatch
):
    client, headers, _ = persona_chat_client
    make_service(enabled=False)
    calls: list[str] = []
    for name in ("evaluate_text", "redact_text", "redact_text_with_count"):
        real = getattr(PolicyEvaluator, name)
        monkeypatch.setattr(PolicyEvaluator, name, lambda self, *a, _r=real, _n=name, **k: (calls.append(_n), _r(self, *a, **k))[1])
    stub_llm_registry.reply = reply(_CHOICE)
    response, _ = _post(client, headers, "forbidden secret")
    assert response.status_code == 200, response.text
    assert "forbidden secret" in json.dumps(stub_llm_registry.requests[0]["messages"])
    assert calls == []


def _policy() -> RoutingPolicy:
    return RoutingPolicy(request_model="auto", server_default_provider="openrouter", boundary_mode="server_default")


async def _select(execute):
    return await runtime.select_llm_router_choice(
        router_request=RouterRequest(model="auto", surface="chat", latest_user_turn="hi"),
        policy=_policy(), candidates=[{"provider": "openrouter", "model": "a"}, {"provider": "openrouter", "model": "b"}],
        provider_listing=_LISTING, execute_router_call=execute,
    )


@pytest.mark.asyncio
async def test_runtime_still_swallows_other_router_errors():
    async def broken(model, messages):
        raise RuntimeError("router down")

    choice, info = await _select(broken)
    assert choice is None and info["error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_runtime_reraises_a_policy_refusal():
    async def refused(model, messages):
        raise LLMPolicyRefusal("Input violates moderation policy")

    with pytest.raises(LLMPolicyRefusal):
        await _select(refused)
