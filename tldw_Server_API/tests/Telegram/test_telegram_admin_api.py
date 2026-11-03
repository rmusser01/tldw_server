from __future__ import annotations

import base64

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import telegram_support
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _make_principal(
    *,
    user_id: int = 101,
    active_org_id: int | None = None,
    active_team_id: int | None = None,
    org_ids: list[int] | None = None,
    team_ids: list[int] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        api_key_id=None,
        subject="test-admin",
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
def auth_header(auth_headers):
    return auth_headers


@pytest.fixture()
def client(client_user_only, monkeypatch):
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"t"))
    monkeypatch.delenv("BYOK_SECONDARY_ENCRYPTION_KEY", raising=False)
    reset_settings()
    return client_user_only


@pytest.fixture()
def principal_override(client):
    def _install(principal: AuthPrincipal) -> None:
        _override_principal(client, principal)

    yield _install
    client.app.dependency_overrides.pop(get_auth_principal, None)


async def _prepare_telegram_scope(pool: DatabasePool) -> tuple[int, int, int]:
    await pool.initialize()
    user_id = await ensure_test_user(pool, "telegram-api-admin", role="admin")
    scopes = AuthnzOrgsTeamsRepo(pool)
    org = await scopes.create_organization(name="Telegram API org", owner_user_id=None)
    team = await scopes.create_team(org_id=int(org["id"]), name="Telegram API team")
    return user_id, int(org["id"]), int(team["id"])


@pytest.mark.parametrize("path", (
    "/api/v1/telegram/admin/bot", "/api/v1/integrations/workspace/telegram/bot",
))
def test_put_and_get_telegram_bot_config_uses_shared_scope(
    client, auth_header, principal_override, monkeypatch, tmp_path, path,
):
    pool = DatabasePool(Settings(
        AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'telegram_api.db'}",
        JWT_SECRET_KEY="x" * 32,
    ))
    user_id, org_id, team_id = client.portal.call(_prepare_telegram_scope, pool)

    async def get_pool():
        return pool

    monkeypatch.setattr(telegram_support, "get_db_pool", get_pool)
    principal = _make_principal(
        user_id=user_id,
        active_team_id=team_id,
        active_org_id=org_id,
        team_ids=[team_id, team_id + 1000],
        org_ids=[org_id, org_id + 1000],
    )
    principal_override(principal)

    payload = {
        "bot_token": ":".join(["123", "abc"]),
        "webhook_secret": "-".join(["secret", "123"]),
        "bot_username": "@ResearchBot",
        "enabled": True,
    }
    try:
        put_res = client.put(path, json=payload, headers=auth_header)
        assert put_res.status_code == 200
        put_body = put_res.json()
        assert put_body["scope_type"] == "team"
        assert put_body["scope_id"] == team_id
        assert put_body["bot_username"] == "researchbot"
        assert put_body["enabled"] is True

        get_res = client.get(path, headers=auth_header)
        assert get_res.status_code == 200
        body = get_res.json()
        assert body["scope_type"] == "team"
        assert body["scope_id"] == team_id
        assert body["bot_username"] == "researchbot"
        assert body["enabled"] is True
    finally:
        client.portal.call(pool.close)


def test_put_telegram_bot_config_rejects_missing_active_scope(client, auth_header, principal_override):
    principal = _make_principal(
        org_ids=[11, 12],
        team_ids=[22],
    )
    principal_override(principal)

    payload = {
        "bot_token": ":".join(["123", "abc"]),
        "webhook_secret": "-".join(["secret", "123"]),
        "enabled": True,
    }
    res = client.put("/api/v1/telegram/admin/bot", json=payload, headers=auth_header)
    assert res.status_code == 400
    assert "active org/team scope" in res.json()["detail"]


@pytest.mark.parametrize(
    ("active_org_id", "active_team_id", "org_ids", "team_ids"),
    [
        (999, None, [11], []),
        (None, 999, [], [22]),
    ],
)
def test_put_telegram_bot_config_rejects_stale_active_scope_claim(
    client,
    auth_header,
    principal_override,
    active_org_id,
    active_team_id,
    org_ids,
    team_ids,
):
    principal = _make_principal(
        active_org_id=active_org_id,
        active_team_id=active_team_id,
        org_ids=org_ids,
        team_ids=team_ids,
    )
    principal_override(principal)

    payload = {
        "bot_token": ":".join(["123", "abc"]),
        "webhook_secret": "-".join(["secret", "123"]),
        "enabled": True,
    }
    res = client.put("/api/v1/telegram/admin/bot", json=payload, headers=auth_header)
    assert res.status_code == 400
    assert "active org/team scope" in res.json()["detail"]


def test_put_telegram_bot_config_rejects_whitespace_credentials(client, auth_header, principal_override):
    principal = _make_principal(active_org_id=11, org_ids=[11], team_ids=[])
    principal_override(principal)

    payload = {
        "bot_token": "   ",
        "webhook_secret": "\t",
        "enabled": True,
    }
    res = client.put("/api/v1/telegram/admin/bot", json=payload, headers=auth_header)
    assert res.status_code == 422
