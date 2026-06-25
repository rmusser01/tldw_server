import pytest


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
def test_chat_command_audit_logged_system_mode(monkeypatch, test_client, auth_headers):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    # Capture audit events
    captured_events = {"events": []}

    class DummyAudit:
        async def log_event(self, *args, **kwargs):
            captured_events["events"].append({"args": args, "kwargs": kwargs})

    async def override_audit():
        return DummyAudit()

    # Override dependencies
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
    test_client.app.dependency_overrides[get_audit_service_for_user] = override_audit

    # Avoid provider network calls
    from tldw_Server_API.app.core.Chat import chat_service
    monkeypatch.setattr(chat_service, "perform_chat_api_call", lambda **kwargs: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    found = False
    for e in captured_events["events"]:
        kw = e.get("kwargs", {})
        if kw.get("action") == "chat.command.executed":
            md = kw.get("metadata") or {}
            if md.get("command") == "time":
                found = True
                assert md.get("mode") == "system"
                break
    assert found, f"Expected chat.command.executed audit event, got: {captured_events['events']}"


@pytest.mark.integration
def test_chat_command_audit_logged_preface_mode(monkeypatch, test_client, auth_headers):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "preface")

    captured_events = {"events": []}

    class DummyAudit:
        async def log_event(self, *args, **kwargs):
            captured_events["events"].append({"args": args, "kwargs": kwargs})

    async def override_audit():
        return DummyAudit()

    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
    test_client.app.dependency_overrides[get_audit_service_for_user] = override_audit

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    captured_msg = {"messages": None}
    def fake_call(**kwargs):
        captured_msg["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}
    _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    found = False
    for e in captured_events["events"]:
        kw = e.get("kwargs", {})
        if kw.get("action") == "chat.command.executed":
            md = kw.get("metadata") or {}
            if md.get("command") == "time":
                found = True
                assert md.get("mode") == "preface"
                break
    assert found

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
def test_chat_command_rbac_enforcement(monkeypatch, test_client, auth_headers):
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

    captured = {"events": []}
    class DummyAudit:
        async def log_event(self, *args, **kwargs):
            captured["events"].append({"args": args, "kwargs": kwargs})
    async def override_audit():
        return DummyAudit()
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
    test_client.app.dependency_overrides[get_audit_service_for_user] = override_audit

    from tldw_Server_API.app.core.Chat import chat_service
    monkeypatch.setattr(chat_service, "perform_chat_api_call", lambda **kwargs: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/time"}], "stream": False}

    try:
        # Deny: no permission
        monkeypatch.setattr(command_router, "_user_has_permission", lambda uid, perm: False)
        _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        denied_found = any(e.get("kwargs", {}).get("action") == "chat.command.executed" and (e.get("kwargs", {}).get("metadata", {}) or {}).get("result_ok") is False for e in captured["events"])
        assert denied_found

        # Allow: grant permission
        captured["events"].clear()
        monkeypatch.setattr(command_router, "_user_has_permission", lambda uid, perm: True)
        _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        allowed_found = any(e.get("kwargs", {}).get("action") == "chat.command.executed" and (e.get("kwargs", {}).get("metadata", {}) or {}).get("result_ok") is True for e in captured["events"])
        assert allowed_found
    finally:
        test_client.app.dependency_overrides.pop(user_override_key, None)


@pytest.mark.integration
def test_chat_command_weather_default_location_system_mode(monkeypatch, test_client, auth_headers):
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
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    captured = {"system_message": None}
    def fake_call(**kwargs):
        captured["system_message"] = kwargs.get("system_message")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    # Override audit dep
    captured_events = {"events": []}
    class DummyAudit:
        async def log_event(self, *args, **kwargs):
            captured_events["events"].append({"args": args, "kwargs": kwargs})
    async def override_audit():
        return DummyAudit()
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
    test_client.app.dependency_overrides[get_audit_service_for_user] = override_audit

    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "/weather"}], "stream": False}
    _ = test_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    # Assert system_message contains default location injected output
    sys_msg = captured["system_message"] or ""
    assert "/weather" in sys_msg
    assert "Atlantis" in sys_msg

    # Assert audit event logged for weather
    weather_found = False
    for e in captured_events["events"]:
        kw = e.get("kwargs", {})
        if kw.get("action") == "chat.command.executed" and (kw.get("metadata", {}) or {}).get("command") == "weather":
            weather_found = True
            break
    assert weather_found
