from __future__ import annotations

import base64

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.utils.chatops.telegram_support import (
    _reset_telegram_link_state_for_tests,
    _reset_telegram_webhook_state_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.telegram_runtime_repo import get_telegram_runtime_repo
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
        user_id=303,
        api_key_id=None,
        subject="telegram-link-test",
        token_type="access",  # nosec B106 - auth principal test fixture token type
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
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"l"))
    monkeypatch.delenv("BYOK_SECONDARY_ENCRYPTION_KEY", raising=False)
    reset_settings()
    _reset_telegram_webhook_state_for_tests()
    _reset_telegram_link_state_for_tests()
    return client_user_only


@pytest.fixture()
def principal_override(client):
    def _install(principal: AuthPrincipal) -> None:
        _override_principal(client, principal)

    yield _install
    client.app.dependency_overrides.pop(get_auth_principal, None)
    _reset_telegram_webhook_state_for_tests()
    _reset_telegram_link_state_for_tests()


@pytest.mark.asyncio
async def test_admin_can_list_and_revoke_linked_telegram_actors(client, auth_headers, principal_override):
    principal = _make_principal(
        active_team_id=22,
        active_org_id=11,
        team_ids=[22, 23],
        org_ids=[11, 12],
    )
    principal_override(principal)

    runtime_repo = await get_telegram_runtime_repo()
    await runtime_repo.upsert_actor_link(
        scope_type="team",
        scope_id=22,
        telegram_user_id=123456,
        auth_user_id=77,
        telegram_username="linked",
    )

    list_response = client.get("/api/v1/telegram/admin/links", headers=auth_headers)
    assert list_response.status_code == 200, list_response.text  # nosec B101 - pytest assertion
    list_body = list_response.json()
    assert list_body["scope_type"] == "team"  # nosec B101 - pytest assertion
    assert list_body["scope_id"] == 22  # nosec B101 - pytest assertion
    assert len(list_body["items"]) == 1  # nosec B101 - pytest assertion
    item = list_body["items"][0]
    assert item["telegram_user_id"] == 123456  # nosec B101 - pytest assertion
    assert item["telegram_username"] == "linked"  # nosec B101 - pytest assertion

    delete_response = client.delete(f"/api/v1/telegram/admin/links/{item['id']}", headers=auth_headers)
    assert delete_response.status_code == 200, delete_response.text  # nosec B101 - pytest assertion
    delete_body = delete_response.json()
    assert delete_body["deleted"] is True  # nosec B101 - pytest assertion
    assert delete_body["id"] == item["id"]  # nosec B101 - pytest assertion
    assert delete_body["scope_type"] == "team"  # nosec B101 - pytest assertion
    assert delete_body["scope_id"] == 22  # nosec B101 - pytest assertion

    empty_response = client.get("/api/v1/telegram/admin/links", headers=auth_headers)
    assert empty_response.status_code == 200, empty_response.text  # nosec B101 - pytest assertion
    assert empty_response.json()["items"] == []  # nosec B101 - pytest assertion


@pytest.mark.asyncio
async def test_admin_cannot_revoke_linked_actor_from_other_scope(client, auth_headers, principal_override):
    principal = _make_principal(
        active_team_id=22,
        active_org_id=11,
        team_ids=[22, 23],
        org_ids=[11, 12],
    )
    principal_override(principal)

    runtime_repo = await get_telegram_runtime_repo()
    current_scope = await runtime_repo.upsert_actor_link(
        scope_type="team",
        scope_id=22,
        telegram_user_id=123456,
        auth_user_id=77,
        telegram_username="linked",
    )
    other_scope = await runtime_repo.upsert_actor_link(
        scope_type="team",
        scope_id=23,
        telegram_user_id=654321,
        auth_user_id=78,
        telegram_username="elsewhere",
    )

    delete_response = client.delete(f"/api/v1/telegram/admin/links/{other_scope['id']}", headers=auth_headers)
    assert delete_response.status_code == 404  # nosec B101 - pytest assertion
    assert delete_response.json()["detail"] == "linked_actor_not_found"  # nosec B101 - pytest assertion

    rows = await runtime_repo.list_actor_links(scope_type="team", scope_id=23)
    assert rows[0]["id"] == other_scope["id"]  # nosec B101 - pytest assertion

    current_rows = await runtime_repo.list_actor_links(scope_type="team", scope_id=22)
    assert current_rows[0]["id"] == current_scope["id"]  # nosec B101 - pytest assertion
