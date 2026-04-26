import os

import pytest
from fastapi import HTTPException


@pytest.mark.integration
def test_chat_command_replace_mode(monkeypatch, test_client, auth_headers):
     # Enable commands and set injection to replace
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "replace")

    # Capture messages payload passed to provider call
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    captured = {"messages": None}

    def fake_call(**kwargs):

        captured["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    # Submit a slash command
    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    # Assert the last user message content is replaced with the command result prefix
    msgs = captured["messages"] or []
    last_user = None
    for m in reversed(msgs):
        if m.get("role") == "user":
            last_user = m
            break
    assert last_user is not None
    parts = last_user.get("content")
    if isinstance(parts, list):
        text_part = next((p for p in parts if p.get("type") == "text"), None)
        assert text_part is not None
        assert text_part.get("text", "").startswith("[/time]")
    elif isinstance(parts, str):
        assert parts.startswith("[/time]")


@pytest.mark.integration
def test_chat_command_revalidates_after_injection(monkeypatch, test_client, auth_headers):
     # Enable commands and system injection so a new system message is appended
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    # Force a very small text limit so injected content exceeds the cap
    monkeypatch.setattr(chat_endpoint, "MAX_TEXT_LENGTH", 10)

    class Result:
        ok = True
        content = "X" * 50
        metadata = {}

    async def fake_dispatch(ctx, name, args):
        return Result()

    monkeypatch.setattr(command_router, "async_dispatch_command", fake_dispatch)

    def fail_call(**_kwargs):

        raise AssertionError("Provider call should not occur when post-injection validation fails")

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fail_call)

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    resp = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    assert resp.status_code == 413


@pytest.mark.integration
def test_chat_command_replace_mode_truncates_injected_text(monkeypatch, test_client, auth_headers):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "replace")
    monkeypatch.setenv("CHAT_COMMANDS_MAX_CHARS", "24")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    class Result:
        ok = True
        content = "very long command payload " * 10
        metadata = {}

    async def fake_dispatch(ctx, name, args):
        return Result()

    monkeypatch.setattr(command_router, "async_dispatch_command", fake_dispatch)

    captured = {"messages": None}

    def fake_call(**kwargs):
        captured["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    msgs = captured["messages"] or []
    last_user = None
    for m in reversed(msgs):
        if m.get("role") == "user":
            last_user = m
            break
    assert last_user is not None
    text = ""
    parts = last_user.get("content")
    if isinstance(parts, list):
        text_part = next((p for p in parts if p.get("type") == "text"), None)
        assert text_part is not None
        text = text_part.get("text", "")
    elif isinstance(parts, str):
        text = parts
    assert text.startswith("[/time]")
    assert len(text) <= 24


@pytest.mark.integration
def test_chat_command_non_audit_503_does_not_abort_request(monkeypatch, test_client, auth_headers):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    async def fake_dispatch(_ctx, _name, _args):
        raise HTTPException(
            status_code=503,
            detail={"error_code": "upstream_unavailable", "message": "temporary outage"},
        )

    monkeypatch.setattr(command_router, "async_dispatch_command", fake_dispatch)

    def fake_call(**_kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    resp = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
