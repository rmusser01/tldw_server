import pytest


def _install_request_user_override(*, permissions=None, roles=None, is_admin=False):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
    from tldw_Server_API.app.main import app

    async def _override_user():
        return User(
            id=42,
            username="command-user",
            email=None,
            is_active=True,
            roles=list(roles or []),
            permissions=list(permissions or []),
            is_admin=is_admin,
        )

    app.dependency_overrides[get_request_user] = _override_user
    return app, get_request_user


@pytest.mark.integration
def test_list_chat_commands_basic(test_client, auth_headers, monkeypatch):
     # Ensure commands are enabled for this test
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "true")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_USER", "7/min")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "70/min")
    r = test_client.get("/api/v1/chat/commands", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert "commands" in data
    names = {c.get("name") for c in data["commands"]}
    # Built-ins should be present
    assert "time" in names
    assert "weather" in names
    assert "skills" in names
    assert "skill" in names
    # Ensure permission metadata present
    by_name = {c.get("name"): c for c in data["commands"]}
    assert "required_permission" in by_name["time"]
    assert by_name["time"]["required_permission"] in (None, "chat.commands.time")
    assert by_name["time"]["usage"] == "/time [timezone]"
    assert by_name["time"]["args"] == ["timezone"]
    assert by_name["time"]["requires_api_key"] is True
    assert by_name["time"]["rbac_required"] is True
    assert "per-user 7/min" in by_name["time"]["rate_limit"]
    assert "global 70/min" in by_name["time"]["rate_limit"]
    assert by_name["weather"]["usage"] == "/weather [location]"
    assert by_name["weather"]["args"] == ["location"]
    assert by_name["skills"]["required_permission"] == "chat.commands.skills"
    assert by_name["skills"]["usage"] == "/skills [filter]"
    assert by_name["skills"]["args"] == ["filter"]
    assert by_name["skills"]["requires_api_key"] is True
    assert by_name["skills"]["rbac_required"] is True
    assert by_name["skill"]["required_permission"] == "chat.commands.skill"
    assert by_name["skill"]["usage"] == "/skill <name> [args]"
    assert by_name["skill"]["args"] == ["name", "args"]
    assert by_name["skill"]["requires_api_key"] is True
    assert by_name["skill"]["rbac_required"] is True


@pytest.mark.integration
def test_list_chat_commands_rbac_filtering(test_client, auth_headers, monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "true")
    monkeypatch.delenv("CHAT_COMMANDS_REQUIRE_PERMISSIONS", raising=False)
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_mod

    monkeypatch.setattr(chat_mod, "user_has_permission", lambda user_id, perm: False, raising=True)

    app, override_key = _install_request_user_override(is_admin=False, permissions=[])
    try:
        r = test_client.get("/api/v1/chat/commands", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "commands" in data
        names = {c.get("name") for c in data["commands"]}
        assert "time" not in names
        assert "weather" not in names
        assert "skills" not in names
        assert "skill" not in names
    finally:
        app.dependency_overrides.pop(override_key, None)


@pytest.mark.integration
def test_list_chat_commands_rbac_allow_path(test_client, auth_headers, monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "true")
    monkeypatch.delenv("CHAT_COMMANDS_REQUIRE_PERMISSIONS", raising=False)
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_mod

    monkeypatch.setattr(chat_mod, "user_has_permission", lambda user_id, perm: True, raising=True)

    app, override_key = _install_request_user_override(is_admin=False, permissions=[])
    try:
        r = test_client.get("/api/v1/chat/commands", headers=auth_headers)
        assert r.status_code == 200
        names = {c.get("name") for c in r.json()["commands"]}
        assert "time" in names
        assert "weather" in names
        assert "skills" in names
        assert "skill" in names
    finally:
        app.dependency_overrides.pop(override_key, None)
