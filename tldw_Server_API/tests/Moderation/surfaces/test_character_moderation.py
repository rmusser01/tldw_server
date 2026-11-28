"""Character chat is moderated at the gateway (TASK-13950.27 PR E): complete-v2 (blocking and streaming), the
character auto-router and memory extraction. Roleplay resume lives next to its fixtures, in
``tests/Chat_NEW/integration/test_roleplay_moderation.py``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.tests.Moderation.conftest import (
    DONE,
    anthropic_frame,
    google_frame,
    finish_frame,
    openai_frame,
    reply,
    usage_frame,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def chat(client_user_only, monkeypatch, tmp_path):
    """A character chat in an isolated DB, and the provider env for an anthropic-named call."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    character_id = client_user_only.get("/api/v1/characters/").json()[0]["id"]
    created = client_user_only.post("/api/v1/chats/", json={"character_id": character_id})
    assert created.status_code == 201, created.text
    return client_user_only, f"/api/v1/chats/{created.json()['id']}/complete-v2"


def _body(text: str = "hi", **extra) -> dict:
    return {
        "provider": "anthropic", "model": "claude-3-sonnet-20240229", "append_user_message": text,
        "save_to_db": False, **extra,
    }


def test_complete_v2_input_block_400(chat, moderation_hooks, stub_llm_registry):
    client, url = chat
    response = client.post(url, json=_body("forbidden"))
    assert response.status_code == 400, response.text
    assert "Input violates moderation policy" in response.text
    assert stub_llm_registry.requests == []


def test_complete_v2_output_redact(chat, moderation_hooks, stub_llm_registry):
    client, url = chat
    stub_llm_registry.reply = reply("the secret word")
    response = client.post(url, json=_body())
    assert response.status_code == 200, response.text
    assert response.json()["assistant_content"] == "the [X] word"


@pytest.mark.parametrize("frame", [anthropic_frame, google_frame], ids=["anthropic", "google"])
def test_complete_v2_stream_redact_with_provider_frames_leaks_nothing(chat, moderation_hooks, stub_llm_registry, frame):
    """The raw-line forwarding must carry the moderated frame: no ``provider_response`` copy of the original text."""
    client, url = chat
    stub_llm_registry.items = [frame("the sec"), frame("ret word"), finish_frame(), usage_frame(), DONE]
    with client.stream("POST", url, json=_body(stream=True)) as response:
        assert response.status_code == 200, response.read()
        lines = [line for line in response.iter_lines() if line]
    text = "\n".join(lines)
    assert "[X]" in text
    assert "provider_response" not in text
    assert "sec" not in text.replace("[X]", "")
    assert "ret word" not in text


def test_character_surface_off_is_not_moderated(chat, make_service, stub_llm_registry):
    from tldw_Server_API.app.core.Moderation import gateway_hooks

    make_service(enabled=True, **{"surface.character.completion": False})
    gateway_hooks.register_moderation_gateway_hooks()
    client, url = chat
    response = client.post(url, json=_body("forbidden"))
    assert response.status_code == 200, response.text
    assert len(stub_llm_registry.requests) == 1


def test_character_router_block_400_no_fallback(make_service, stub_llm_registry, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    """A refused router prompt is the caller's 400 (an HTTPException), never a default choice or a failure mark."""
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as sessions
    from tldw_Server_API.app.core.LLM_Calls.routing import RouterRequest, RoutingPolicy
    from tldw_Server_API.app.core.Moderation import gateway_hooks
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import capture_provider_override_call_snapshot
    from tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime import ProviderCredentialRuntime

    make_service(enabled=True, **{"surface.character.completion": False})
    gateway_hooks.register_moderation_gateway_hooks()
    used: list = []

    runtime = ProviderCredentialRuntime(
        user_id=1, team_ids=None, org_ids=None, trusted_base_url_override=False,
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )

    async def mark_used(self, handle):
        used.append(handle)

    monkeypatch.setattr(ProviderCredentialRuntime, "mark_used", mark_used)

    policy = RoutingPolicy(request_model="auto", server_default_provider="anthropic", boundary_mode="server_default_provider")
    candidates = [{"provider": "anthropic", "model": "a"}, {"provider": "anthropic", "model": "b"}]
    listing = {"default_provider": "anthropic", "providers": [{"name": "anthropic", "default_model": "router-m", "models": ["a", "b"]}]}
    request = RouterRequest(model="auto", surface="character_chat", latest_user_turn="forbidden", scope="c", requested_capabilities={})
    with pytest.raises(LLMPolicyRefusal) as err:
        asyncio.run(sessions._select_auto_character_llm_router_choice(
            router_request=request, policy=policy, candidates=candidates, provider_listing=listing,
            current_user=SimpleNamespace(id=1), credential_runtime=runtime,
        ))
    assert err.value.status_code == 400
    assert stub_llm_registry.requests == []
    assert used == []


def test_memory_extraction_input_block_propagates(moderation_hooks, stub_llm_registry):
    from tldw_Server_API.app.core.Character_Chat.modules import character_memory_extraction as mod

    with pytest.raises(LLMPolicyRefusal):
        mod.extract_character_memories(
            [{"sender": "user", "content": "forbidden"}], "Ari", "Bo", [],
            api_endpoint="e2e-fake", api_key="k", model="m",
        )
    assert stub_llm_registry.requests == []
