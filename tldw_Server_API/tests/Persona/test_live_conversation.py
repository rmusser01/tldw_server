"""Live conversation must retain the ordinary HTTP admission boundary."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request

from tldw_Server_API.app.core.Persona import live_conversation as live


def test_target_resolution_does_not_require_static_credentials(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service, chat_target_resolution

    target = SimpleNamespace(provider="deepseek", model="deepseek-chat")
    monkeypatch.setattr(chat_target_resolution, "resolve_chat_target", lambda **kwargs: target)

    def unexpected_static_credentials(*args, **kwargs):
        pytest.fail("Text must leave effective credential resolution to authenticated Chat")

    monkeypatch.setattr(chat_service, "resolve_static_provider_fallback", unexpected_static_credentials)
    assert live.resolve_persona_conversation_target() is target


def test_voice_preflight_rejects_missing_server_credentials(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils, provider_metadata

    monkeypatch.setattr(
        live, "resolve_persona_conversation_target", lambda: SimpleNamespace(provider="deepseek", model="deepseek-chat")
    )
    monkeypatch.setattr(
        chat_service, "resolve_static_provider_fallback", lambda provider: SimpleNamespace(api_key=None, app_config={})
    )
    monkeypatch.setattr(provider_metadata, "provider_requires_api_key", lambda provider: True)
    monkeypatch.setattr(adapter_utils, "provider_auth_is_resolved", lambda *args, **kwargs: False)
    with pytest.raises(live.PersonaConversationError, match="server-configured credentials"):
        live.require_persona_voice_conversation_credentials()


@pytest.mark.parametrize("text", ["Hi Migu, reply hello", "What does URL mean?", "Tell me about search algorithms"])
def test_ordinary_conversation_does_not_require_tool_plan(text):
    assert not live.requires_tool_plan(text)


@pytest.mark.parametrize(
    "text",
    [
        "Search my notes for cats",
        "Please ingest https://example.com",
        "skill: calendar today",
        "https://example.com",
        "Find documents about cats",
    ],
)
def test_explicit_tools_retain_plan(text):
    assert live.requires_tool_plan(text)


@pytest.mark.asyncio
async def test_completion_runs_auth_dependency_and_keeps_persona_history(monkeypatch):
    app = FastAPI()
    captured = {}

    async def admission(request: Request):
        if request.headers.get("authorization") != "Bearer test-only":
            raise HTTPException(401)
        captured["client"] = request.client.host

    @app.post("/api/v1/chat/completions", dependencies=[Depends(admission)])
    async def chat(request: Request):
        captured["body"] = await request.json()
        return {"choices": [{"message": {"content": "Hello from Migu"}}]}

    monkeypatch.setattr(
        live, "resolve_persona_conversation_target", lambda: SimpleNamespace(provider="deepseek", model="deepseek-chat")
    )
    result = await live.complete_persona_conversation(
        app=app,
        headers={"authorization": "Bearer test-only"},
        client=("192.0.2.2", 1234),
        system_prompt="You are Migu.",
        turns=[{"role": "user", "content": "Hi"}],
    )
    assert result == "Hello from Migu"
    assert captured["client"] == "192.0.2.2"
    assert captured["body"]["save_to_db"] is False
    assert "tools" not in captured["body"]
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "You are Migu."},
        {"role": "user", "content": "Hi"},
    ]
    with pytest.raises(live.PersonaConversationError, match="not authorized"):
        await live.complete_persona_conversation(
            app=app, headers={}, client=None, system_prompt="Migu", turns=[{"role": "user", "content": "Hi"}]
        )


@pytest.mark.asyncio
async def test_completion_error_does_not_echo_provider_body(monkeypatch):
    app = FastAPI()

    @app.post("/api/v1/chat/completions")
    async def chat():
        raise HTTPException(503, "secret provider diagnostic")

    monkeypatch.setattr(
        live, "resolve_persona_conversation_target", lambda: SimpleNamespace(provider="deepseek", model="deepseek-chat")
    )
    with pytest.raises(live.PersonaConversationError) as failure:
        await live.complete_persona_conversation(
            app=app, headers={}, client=None, system_prompt="Migu", turns=[{"role": "user", "content": "Hi"}]
        )
    assert "secret" not in str(failure.value)


@pytest.mark.parametrize("command", ["/skill calendar today", "  /weather London"])
@pytest.mark.asyncio
async def test_completion_rejects_slash_commands_before_chat_admission(monkeypatch, command):
    app = FastAPI()
    admitted = []

    @app.post("/api/v1/chat/completions")
    async def chat():
        admitted.append(True)
        return {"choices": [{"message": {"content": "command executed"}}]}

    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "true")
    monkeypatch.setattr(
        live, "resolve_persona_conversation_target", lambda: SimpleNamespace(provider="deepseek", model="deepseek-chat")
    )
    with pytest.raises(live.PersonaConversationError, match="Slash commands"):
        await live.complete_persona_conversation(
            app=app,
            headers={},
            client=None,
            system_prompt="Migu",
            turns=[{"role": "user", "content": command}],
        )
    assert admitted == []


@pytest.mark.parametrize("blank", ["", "  ", "\n\t"])
@pytest.mark.asyncio
async def test_blank_send_cannot_replay_previous_slash_command(monkeypatch, blank):
    app = FastAPI()
    admitted = []

    @app.post("/api/v1/chat/completions")
    async def chat(request: Request):
        admitted.append(await request.json())
        return {"choices": [{"message": {"content": "command executed"}}]}

    monkeypatch.setattr(
        live, "resolve_persona_conversation_target", lambda: SimpleNamespace(provider="deepseek", model="deepseek-chat")
    )
    with pytest.raises(live.PersonaConversationError):
        await live.complete_persona_conversation(
            app=app,
            headers={},
            client=None,
            system_prompt="Migu",
            turns=[{"role": "user", "content": "/skill calendar today"}, {"role": "user", "content": blank}],
        )
    assert admitted == []


@pytest.mark.asyncio
async def test_cancel_invalidates_old_task_and_preserves_next_owner():
    registry = live.PersonaLiveTurnRegistry()
    old = asyncio.create_task(asyncio.Event().wait())
    new = asyncio.create_task(asyncio.Event().wait())
    try:
        registry.register(user_id="1", session_id="s", task=old)
        registry.cancel(user_id="1", session_id="s")
        registry.register(user_id="1", session_id="s", task=new)
        registry.release(user_id="1", session_id="s", task=old)
        assert not registry.is_current(user_id="1", session_id="s", task=old)
        assert registry.is_current(user_id="1", session_id="s", task=new)
        assert not registry.is_current(user_id="2", session_id="s", task=new)
    finally:
        old.cancel()
        new.cancel()
        await asyncio.gather(old, new, return_exceptions=True)


@pytest.mark.asyncio
async def test_registry_keeps_queued_owners_until_explicit_stop():
    registry = live.PersonaLiveTurnRegistry()
    first = asyncio.create_task(asyncio.Event().wait())
    queued = asyncio.create_task(asyncio.Event().wait())
    other = asyncio.create_task(asyncio.Event().wait())
    try:
        registry.register(user_id="1", session_id="s", task=first)
        registry.register(user_id="1", session_id="s", task=queued)
        registry.register(user_id="2", session_id="s", task=other)
        await asyncio.sleep(0)
        assert registry.is_current(user_id="1", session_id="s", task=first)
        assert registry.is_current(user_id="1", session_id="s", task=queued)
        registry.cancel(user_id="1", session_id="s")
        assert not registry.is_current(user_id="1", session_id="s", task=first)
        assert not registry.is_current(user_id="1", session_id="s", task=queued)
        await asyncio.gather(first, queued, return_exceptions=True)
        assert first.cancelled() and queued.cancelled()
        assert registry.is_current(user_id="2", session_id="s", task=other)
        assert not other.done()
    finally:
        for task in (first, queued, other):
            task.cancel()
        await asyncio.gather(first, queued, other, return_exceptions=True)


@pytest.mark.asyncio
async def test_releasing_completed_turn_preserves_other_registered_turn():
    registry = live.PersonaLiveTurnRegistry()
    first = asyncio.create_task(asyncio.Event().wait())
    queued = asyncio.create_task(asyncio.Event().wait())
    try:
        registry.register(user_id="1", session_id="s", task=first)
        registry.register(user_id="1", session_id="s", task=queued)
        registry.release(user_id="1", session_id="s", task=queued)
        assert registry.is_current(user_id="1", session_id="s", task=first)
        assert not registry.is_current(user_id="1", session_id="s", task=queued)
    finally:
        first.cancel()
        queued.cancel()
        await asyncio.gather(first, queued, return_exceptions=True)
