from __future__ import annotations

import base64

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.utils.chatops.telegram_support import (
    TelegramCommand,
    evaluate_telegram_message_policy,
    parse_telegram_command,
    _reset_telegram_webhook_state_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _make_principal(
    *,
    active_org_id: int | None = None,
    active_team_id: int | None = None,
    org_ids: list[int] | None = None,
    team_ids: list[int] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=404,
        api_key_id=None,
        subject="telegram-command-test",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=["system.configure"],
        is_admin=True,
        org_ids=org_ids or [],
        team_ids=team_ids or [],
        active_org_id=active_org_id,
        active_team_id=active_team_id,
    )


def _override_principal(client, principal: AuthPrincipal) -> None:
    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        request.state.active_org_id = principal.active_org_id
        request.state.active_team_id = principal.active_team_id
        return principal

    client.app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal


@pytest.fixture()
def client(client_user_only, monkeypatch):
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"c"))
    monkeypatch.delenv("BYOK_SECONDARY_ENCRYPTION_KEY", raising=False)
    reset_settings()
    _reset_telegram_webhook_state_for_tests()
    return client_user_only


@pytest.fixture()
def principal_override(client):
    def _install(principal: AuthPrincipal) -> None:
        _override_principal(client, principal)

    yield _install
    client.app.dependency_overrides.pop(get_auth_principal, None)
    _reset_telegram_webhook_state_for_tests()


def _seed_telegram_bot(
    client,
    principal_override,
    *,
    scope_type: str,
    scope_id: int,
    bot_token: str,
    webhook_secret: str,
) -> None:
    if scope_type == "team":
        principal = _make_principal(active_team_id=scope_id, team_ids=[scope_id], org_ids=[1])
    else:
        principal = _make_principal(active_org_id=scope_id, org_ids=[scope_id], team_ids=[1])
    principal_override(principal)

    response = client.put(
        "/api/v1/telegram/admin/bot",
        json={
            "bot_token": bot_token,
            "webhook_secret": webhook_secret,
            "enabled": True,
        },
    )
    assert response.status_code == 200, response.text


def test_parse_telegram_command_mode():
    command = parse_telegram_command("/mode persona")

    assert command == TelegramCommand(action="mode", input="persona")


def test_parse_telegram_command_ask():
    command = parse_telegram_command("/ask summarize the last report")

    assert command == TelegramCommand(action="ask", input="summarize the last report")


def test_parse_telegram_command_with_target_bot():
    command = parse_telegram_command("/ask@ourbot summarize the last report")

    assert command == TelegramCommand(
        action="ask",
        input="summarize the last report",
        target_bot_username="ourbot",
    )


def test_group_freeform_message_without_reply_is_ignored():
    policy = evaluate_telegram_message_policy(chat_type="group", text="hello everyone", reply_to_bot=False)

    assert policy.should_process is False
    assert policy.reason == "group_freeform_requires_command_or_reply"
    assert policy.command is None


def test_group_command_is_processed():
    policy = evaluate_telegram_message_policy(chat_type="group", text="/ask summarize this", reply_to_bot=False)

    assert policy.should_process is True
    assert policy.reason is None
    assert policy.command == TelegramCommand(action="ask", input="summarize this")


def test_group_reply_to_bot_is_processed():
    policy = evaluate_telegram_message_policy(chat_type="group", text="please summarize", reply_to_bot=True)

    assert policy.should_process is True
    assert policy.reason is None
    assert policy.command is None


def test_telegram_webhook_ignores_group_freeform_message_without_reply(client, principal_override):
    principal = _make_principal(active_team_id=22, team_ids=[22], org_ids=[1])
    principal_override(principal)
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=22,
        bot_token="123:abc",
        webhook_secret="secret-123",
    )

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret-123"},
        json={
            "update_id": 9001,
            "message": {
                "message_id": 5,
                "chat": {"id": 123, "type": "group"},
                "from": {"id": 77, "username": "unknown"},
                "text": "hello everyone",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "ignored"}


def test_telegram_webhook_ignores_group_command_for_other_bot(client, principal_override):
    principal = _make_principal(active_team_id=22, team_ids=[22], org_ids=[1])
    principal_override(principal)
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=22,
        bot_token="123:abc",
        webhook_secret="secret-123",
    )

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret-123"},
        json={
            "update_id": 9003,
            "message": {
                "message_id": 7,
                "chat": {"id": 123, "type": "group"},
                "from": {"id": 77, "username": "unknown"},
                "text": "/ask@otherbot summarize the last report",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "ignored"}


def test_replayed_ignored_group_freeform_message_is_deduped(client, principal_override):
    principal = _make_principal(active_team_id=22, team_ids=[22], org_ids=[1])
    principal_override(principal)
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=22,
        bot_token="123:abc",
        webhook_secret="secret-123",
    )

    payload = {
        "update_id": 9002,
        "message": {
            "message_id": 6,
            "chat": {"id": 123, "type": "group"},
            "from": {"id": 77, "username": "unknown"},
            "text": "hello everyone",
        },
    }
    headers = {"X-Telegram-Bot-Api-Secret-Token": "secret-123"}

    first = client.post("/api/v1/telegram/webhook", headers=headers, json=payload)
    second = client.post("/api/v1/telegram/webhook", headers=headers, json=payload)

    assert first.status_code == 200
    assert first.json() == {"ok": True, "status": "ignored"}
    assert second.status_code == 200
    assert second.json() == {"ok": True, "status": "duplicate"}


def test_telegram_webhook_ignores_group_reply_to_different_bot(client, principal_override):
    principal = _make_principal(active_team_id=22, team_ids=[22], org_ids=[1])
    principal_override(principal)
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=22,
        bot_token="123:abc",
        webhook_secret="secret-123",
    )

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret-123"},
        json={
            "update_id": 9004,
            "message": {
                "message_id": 8,
                "chat": {"id": 123, "type": "group"},
                "from": {"id": 77, "username": "unknown"},
                "reply_to_message": {
                    "message_id": 2,
                    "from": {"id": 88, "is_bot": True, "username": "otherbot"},
                    "text": "previous bot message",
                },
                "text": "please summarize this",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "ignored"}
