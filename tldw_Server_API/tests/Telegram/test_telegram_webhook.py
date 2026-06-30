from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
import tldw_Server_API.app.api.v1.endpoints.telegram_support as telegram_support
from tldw_Server_API.app.api.v1.endpoints.telegram_support import (
    _reset_telegram_webhook_state_for_tests,
    telegram_webhook_impl,
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
        user_id=202,
        api_key_id=None,
        subject="telegram-webhook-test",
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


def _build_request(body: bytes, *, secret: str | None) -> Request:
    headers: list[tuple[bytes, bytes]] = [(b"content-type", b"application/json")]
    if secret is not None:
        headers.append((b"x-telegram-bot-api-secret-token", secret.encode("utf-8")))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/telegram/webhook",
        "query_string": b"",
        "headers": headers,
    }

    async def _receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, _receive)


@pytest.fixture()
def client(client_user_only, monkeypatch):
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"t"))
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
    client, principal_override, *, scope_type: str, scope_id: int, bot_token: str, webhook_secret: str
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


def test_telegram_webhook_accepts_and_dedupes_replayed_update(client, principal_override):
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=22,
        bot_token="123:abc",
        webhook_secret="secret-123",
    )

    payload = {
        "update_id": 5001,
        "message": {
            "message_id": 10,
            "chat": {"id": 222, "type": "private"},
            "text": "hello",
        },
    }
    headers = {"X-Telegram-Bot-Api-Secret-Token": "secret-123"}

    first = client.post("/api/v1/telegram/webhook", json=payload, headers=headers)
    second = client.post("/api/v1/telegram/webhook", json=payload, headers=headers)

    assert first.status_code == 200
    assert first.json() == {"ok": True, "status": "accepted"}
    assert second.status_code == 200
    assert second.json() == {"ok": True, "status": "duplicate"}


@pytest.mark.asyncio
async def test_telegram_webhook_rejects_short_secret_before_body_parse():
    async def _failing_repo():
        raise AssertionError("repo lookup should not happen before secret validation")

    request = _build_request(b"not-json", secret="short")
    response = await telegram_webhook_impl(request=request, get_org_secret_repo=_failing_repo)

    assert response.status_code == 401
    assert json.loads(response.body) == {"ok": False, "error": "invalid_secret"}


@pytest.mark.asyncio
async def test_telegram_webhook_rejects_missing_secret_before_body_parse():
    async def _failing_repo():
        raise AssertionError("repo lookup should not happen before secret validation")

    request = _build_request(b"not-json", secret=None)
    response = await telegram_webhook_impl(request=request, get_org_secret_repo=_failing_repo)

    assert response.status_code == 401
    assert json.loads(response.body) == {"ok": False, "error": "invalid_secret"}


def test_telegram_webhook_rejects_ambiguous_secret_collision(client, principal_override):
    shared_secret = "shared-secret"
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=55,
        bot_token="123:jkl",
        webhook_secret=shared_secret,
    )
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="org",
        scope_id=66,
        bot_token="123:mno",
        webhook_secret=shared_secret,
    )

    response = client.post(
        "/api/v1/telegram/webhook",
        json={"update_id": 5003, "message": {"message_id": 12, "chat": {"id": 444, "type": "private"}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": shared_secret},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_secret"


def test_telegram_webhook_rejects_invalid_secret(client, principal_override):
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="org",
        scope_id=33,
        bot_token="123:def",
        webhook_secret="secret-456",
    )

    response = client.post(
        "/api/v1/telegram/webhook",
        json={"update_id": 5002, "message": {"message_id": 11, "chat": {"id": 333, "type": "private"}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_secret"


def test_telegram_webhook_rejects_invalid_secret_before_job_manager_resolution(client, principal_override):
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="org",
        scope_id=34,
        bot_token="123:ghi",
        webhook_secret="secret-790",
    )

    def _failing_job_manager():
        raise AssertionError("job manager should not be resolved before secret validation")

    client.app.dependency_overrides[get_job_manager] = _failing_job_manager
    try:
        response = client.post(
            "/api/v1/telegram/webhook",
            json={"update_id": 5004, "message": {"message_id": 13, "chat": {"id": 334, "type": "private"}}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        )
    finally:
        client.app.dependency_overrides.pop(get_job_manager, None)

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_secret"


@pytest.mark.asyncio
async def test_telegram_webhook_rejects_wrong_secret_before_body_parse(monkeypatch):
    class _FakeRepo:
        async def list_secrets(self, provider: str):
            assert provider == "telegram"
            return [{"scope_type": "team", "scope_id": 77}]

        async def fetch_secret(self, scope_type: str, scope_id: int, provider: str):
            assert provider == "telegram"
            assert scope_type == "team"
            assert scope_id == 77
            return {"encrypted_blob": "ignored"}

    monkeypatch.setattr(
        telegram_support,
        "_decrypt_telegram_payload",
        lambda _encrypted_blob: {
            "provider": "telegram",
            "credential_version": 1,
            "bot_username": "example_bot",
            "enabled": True,
            "bot_token": "123:abc",
            "webhook_secret": "stored-secret-123",
        },
    )

    async def _receive() -> dict[str, object]:
        raise AssertionError("body should not be read before auth failure")

    async def _get_repo() -> _FakeRepo:
        return _FakeRepo()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/telegram/webhook",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-telegram-bot-api-secret-token", b"wrong-but-long-enough"),
            ],
        },
        _receive,
    )

    response = await telegram_webhook_impl(request=request, get_org_secret_repo=_get_repo)

    assert response.status_code == 401
    assert json.loads(response.body) == {"ok": False, "error": "invalid_secret"}


def test_telegram_webhook_rejects_malformed_payload(client, principal_override):
    _seed_telegram_bot(
        client,
        principal_override,
        scope_type="team",
        scope_id=44,
        bot_token="123:ghi",
        webhook_secret="secret-789",
    )

    response = client.post(
        "/api/v1/telegram/webhook",
        data="not-json",
        headers={
            "content-type": "application/json",
            "X-Telegram-Bot-Api-Secret-Token": "secret-789",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_json"


@pytest.mark.asyncio
async def test_telegram_webhook_execution_identity_failure_log_is_sanitized(monkeypatch):
    class _FakeSecretRepo:
        async def list_secrets(self, provider: str):
            assert provider == "telegram"
            return [{"scope_type": "team", "scope_id": 22}]

        async def fetch_secret(self, scope_type: str, scope_id: int, provider: str):
            assert provider == "telegram"
            assert scope_type == "team"
            assert scope_id == 22
            return {"encrypted_blob": "telegram-secret-envelope"}

    class _FakeRuntimeRepo:
        async def store_webhook_receipt(self, **kwargs):
            assert kwargs["scope_type"] == "team"
            assert kwargs["scope_id"] == 22
            return True

        async def get_actor_link(self, **kwargs):
            assert kwargs["scope_type"] == "team"
            assert kwargs["scope_id"] == 22
            assert kwargs["telegram_user_id"] == 123456
            return {"auth_user_id": 202}

    class _FailingExecutionIdentityService:
        def mint_telegram_identity(self, **kwargs):
            raise RuntimeError("telegram identity token at /private/identity.pem")

    async def _get_secret_repo():
        return _FakeSecretRepo()

    async def _get_runtime_repo():
        return _FakeRuntimeRepo()

    async def _get_execution_identity_service():
        return _FailingExecutionIdentityService()

    monkeypatch.setattr(
        telegram_support,
        "_decrypt_telegram_payload",
        lambda _encrypted_blob: {
            "provider": "telegram",
            "credential_version": 1,
            "bot_username": "example_bot",
            "enabled": True,
            "bot_token": "123:abc",
            "webhook_secret": "stored-secret-123",
        },
    )
    fake_logger = MagicMock()
    monkeypatch.setattr(telegram_support, "logger", fake_logger)
    request = _build_request(
        json.dumps(
            {
                "update_id": 6001,
                "message": {
                    "message_id": 14,
                    "from": {"id": 123456, "username": "alice"},
                    "chat": {"id": 555, "type": "private"},
                    "text": "/ask hello",
                },
            }
        ).encode("utf-8"),
        secret="stored-secret-123",
    )

    response = await telegram_webhook_impl(
        request=request,
        job_manager=object(),
        get_org_secret_repo=_get_secret_repo,
        get_telegram_runtime_repo=_get_runtime_repo,
        get_execution_identity_service=_get_execution_identity_service,
    )

    assert response.status_code == 503
    assert json.loads(response.body) == {"ok": False, "error": "execution_identity_unavailable"}
    fake_logger.error.assert_called_once_with("Failed to mint Telegram execution identity")
