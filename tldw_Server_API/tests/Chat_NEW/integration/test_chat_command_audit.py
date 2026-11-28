import pytest


@pytest.fixture
def command_audit_events(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user

    events = []

    def install(client, *, fail_flush=False):
        class AuditRecorder:
            async def log_event(self, *, action, metadata=None, **_):
                if action == "chat.command.executed":
                    events.append(metadata or {})

            async def flush(self, *, raise_on_failure=False):
                if fail_flush and raise_on_failure:
                    raise OSError("audit flush failed")

        async def override_audit():
            return AuditRecorder()

        monkeypatch.setitem(client.app.dependency_overrides, get_audit_service_for_user, override_audit)
        return events

    return install


def _install_request_user_override(app, *, permissions=None, roles=None, is_admin=False):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user

    async def _override_user():
        return User(
            id=42,
            username="command-audit-user",
            email=None,
            is_active=True,
            roles=list(roles or []),
            permissions=list(permissions or []),
            is_admin=is_admin,
        )

    app.dependency_overrides[get_request_user] = _override_user
    return get_request_user


@pytest.mark.integration
def test_chat_command_audit_logged_system_mode(monkeypatch, test_client, auth_headers, command_audit_events):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    events = command_audit_events(test_client)

    # Avoid provider network calls
    from tldw_Server_API.app.core.Chat import chat_service
    monkeypatch.setattr(chat_service, "perform_chat_api_call", lambda **kwargs: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    assert any(event.get("command") == "time" and event.get("mode") == "system" for event in events)


@pytest.fixture
def command_moderation_at_capacity(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Moderation import async_execution, moderation_service
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError

    rejected = AsyncMock(side_effect=ModerationExecutionError("capacity_exhausted"))
    monkeypatch.setattr(async_execution, "_run_moderation_offload", rejected)
    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call",
        lambda **_: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    def configure(*, enabled, input_enabled):
        policy = moderation_service.ModerationPolicy(enabled=enabled, input_enabled=input_enabled, output_enabled=False)
        service = SimpleNamespace(get_effective_policy=lambda *_args, **_kwargs: policy)
        monkeypatch.setattr(moderation_service, "get_moderation_service", lambda: service)
        return rejected

    return configure


@pytest.mark.integration
@pytest.mark.parametrize(
    "enabled,input_enabled,expected_status", [(False, True, 200), (True, False, 200), (True, True, 503)],
)
def test_command_respects_input_policy_when_moderation_is_busy(
    credentialed_test_client, auth_headers, command_audit_events, command_moderation_at_capacity,
    enabled, input_enabled, expected_status,
):
    events = command_audit_events(credentialed_test_client)
    rejected = command_moderation_at_capacity(enabled=enabled, input_enabled=input_enabled)
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False},
        headers=auth_headers,
    )

    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        assert any(event.get("command") == "time" for event in events)
        rejected.assert_not_awaited()
    else:
        assert response.json()["detail"]["error_code"] == "moderation_unavailable"
        assert not events
        rejected.assert_awaited_once()


@pytest.mark.integration
def test_command_audit_flush_failure_stops_provider_dispatch(
    monkeypatch, credentialed_test_client, auth_headers, command_audit_events, command_moderation_at_capacity,
):
    from tldw_Server_API.app.core.Chat import chat_service

    command_moderation_at_capacity(enabled=False, input_enabled=False)
    events = command_audit_events(credentialed_test_client, fail_flush=True)
    provider_calls = 0

    def provider(**_):
        nonlocal provider_calls
        provider_calls += 1
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call", provider)
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False},
        headers=auth_headers,
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == "mandatory_audit_unavailable"
    assert not events
    assert provider_calls == 0


@pytest.mark.integration
def test_chat_command_audit_logged_preface_mode(
    monkeypatch, credentialed_test_client, auth_headers, command_audit_events
):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "preface")

    events = command_audit_events(credentialed_test_client)

    captured_msg = {"messages": None}
    def fake_call(**kwargs):
        captured_msg["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", fake_call)

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    response = credentialed_test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
    assert response.status_code == 200, response.text

    assert any(event.get("command") == "time" and event.get("mode") == "preface" for event in events)

    # Additional assertion: user message is prefixed in messages payload
    msgs = captured_msg["messages"] or []
    # Find the last user message
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
def test_chat_command_rbac_enforcement(monkeypatch, test_client, auth_headers, command_audit_events):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_REQUIRE_PERMISSIONS", "1")
    # Force multi-user mode in router and steer permission checks
    from tldw_Server_API.app.core.Chat import command_router
    monkeypatch.setattr(command_router, "is_single_user_mode", lambda: False)
    user_override_key = _install_request_user_override(
        test_client.app,
        is_admin=False,
        permissions=[],
    )

    events = command_audit_events(test_client)

    from tldw_Server_API.app.core.Chat import chat_service
    monkeypatch.setattr(chat_service, "perform_chat_api_call", lambda **kwargs: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}

    try:
        # Deny: no permission
        monkeypatch.setattr(command_router, "_user_has_permission", lambda uid, perm: False)
        _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        assert any(event.get("result_ok") is False for event in events)

        # Allow: grant permission
        events.clear()
        monkeypatch.setattr(command_router, "_user_has_permission", lambda uid, perm: True)
        _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        assert any(event.get("result_ok") is True for event in events)
    finally:
        test_client.app.dependency_overrides.pop(user_override_key, None)


@pytest.mark.integration
def test_chat_command_weather_default_location_system_mode(
    monkeypatch, credentialed_test_client, auth_headers, command_audit_events
):
     # Enable commands and system injection; ensure env default_location not set
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")
    monkeypatch.delenv("DEFAULT_LOCATION", raising=False)

    # Provide default_location via config.txt loader
    import configparser
    from tldw_Server_API.app.core.Chat import command_router
    cp = configparser.ConfigParser()
    cp.add_section('Chat-Commands')
    cp.set('Chat-Commands', 'default_location', 'Atlantis')
    monkeypatch.setattr(command_router, "load_comprehensive_config", lambda: cp)

    # Mock weather provider
    class OkClient:
        def get_current(self, location=None, lat=None, lon=None):
            class R:
                ok = True
                summary = f"Sunny at {location or 'nowhere'}"
                metadata = {"provider": "test"}
            return R()
    monkeypatch.setattr(command_router, "get_weather_client", lambda: OkClient())

    # Capture system_message passed to provider call
    captured = {"system_message": None}
    def fake_call(**kwargs):
        captured["system_message"] = kwargs.get("system_message")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", fake_call)

    events = command_audit_events(credentialed_test_client)

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/weather"}], "stream": False}
    response = credentialed_test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
    assert response.status_code == 200, response.text

    # Assert system_message contains default location injected output
    sys_msg = captured["system_message"] or ""
    assert "/weather" in sys_msg
    assert "Atlantis" in sys_msg

    # Assert audit event logged for weather
    assert any(event.get("command") == "weather" for event in events)
