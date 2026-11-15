from __future__ import annotations

import base64
from datetime import datetime, timezone
import json

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import integrations_control_plane as integrations_module
from tldw_Server_API.app.api.v1.endpoints.integrations_control_plane import (
    get_integrations_control_plane_service,
)
from tldw_Server_API.app.api.v1.utils.chatops.discord_support import (
    _decrypt_discord_payload,
    _encrypt_discord_payload,
)
from tldw_Server_API.app.api.v1.utils.chatops.discord_support import _reset_discord_state_for_tests
from tldw_Server_API.app.api.v1.utils.chatops.slack_support import (
    _decrypt_slack_payload,
    _encrypt_slack_payload,
)
from tldw_Server_API.app.api.v1.utils.chatops.slack_support import _reset_slack_state_for_tests
from tldw_Server_API.app.api.v1.utils.chatops.telegram_support import _reset_telegram_link_state_for_tests
from tldw_Server_API.app.api.v1.schemas.integrations_control_plane_schemas import (
    IntegrationConnection,
    IntegrationOverviewResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos import get_workspace_provider_installations_repo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.telegram_runtime_repo import get_telegram_runtime_repo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


class _FakeIntegrationsControlPlaneService:
    async def build_personal_overview(self, *, user_id: int) -> IntegrationOverviewResponse:
        assert user_id == 303  # nosec B101 - pytest assertion
        return IntegrationOverviewResponse(
            scope="personal",
            items=[
                IntegrationConnection(
                    id="personal:slack",
                    provider="slack",
                    scope="personal",
                    display_name="Slack",
                    status="connected",
                    enabled=True,
                    actions=["disconnect"],
                ),
                IntegrationConnection(
                    id="personal:discord",
                    provider="discord",
                    scope="personal",
                    display_name="Discord",
                    status="disconnected",
                    enabled=False,
                    actions=["connect"],
                ),
            ],
        )

    async def build_workspace_overview(
        self,
        *,
        org_id: int,
        scope_type: str = "org",
        scope_id: int | None = None,
    ) -> IntegrationOverviewResponse:
        assert org_id == 11  # nosec B101 - pytest assertion
        assert scope_type == "team"  # nosec B101 - pytest assertion
        assert scope_id == 22  # nosec B101 - pytest assertion
        return IntegrationOverviewResponse(
            scope="workspace",
            items=[
                IntegrationConnection(
                    id="workspace:slack",
                    provider="slack",
                    scope="workspace",
                    display_name="Slack",
                    status="connected",
                    enabled=True,
                    actions=["manage"],
                ),
                IntegrationConnection(
                    id="workspace:discord",
                    provider="discord",
                    scope="workspace",
                    display_name="Discord",
                    status="disabled",
                    enabled=False,
                    actions=["manage"],
                ),
                IntegrationConnection(
                    id="workspace:telegram",
                    provider="telegram",
                    scope="workspace",
                    display_name="@tldwbot",
                    status="connected",
                    enabled=True,
                    actions=["configure_bot", "generate_pairing_code", "manage_linked_actors"],
                ),
            ],
        )


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _make_principal(
    *,
    user_id: int = 303,
    active_org_id: int | None = None,
    active_team_id: int | None = None,
    org_ids: list[int] | None = None,
    team_ids: list[int] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        api_key_id=None,
        subject="integrations-control-plane-test",
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
        request.state.org_ids = list(principal.org_ids)
        request.state.team_ids = list(principal.team_ids)
        return principal

    client.app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal


@pytest.fixture()
def client(client_user_only):
    _reset_slack_state_for_tests()
    _reset_discord_state_for_tests()
    _reset_telegram_link_state_for_tests()
    return client_user_only


@pytest.fixture()
def principal_override(client):
    def _install(principal: AuthPrincipal) -> None:
        _override_principal(client, principal)

    yield _install
    client.app.dependency_overrides.pop(get_auth_principal, None)
    client.app.dependency_overrides.pop(get_integrations_control_plane_service, None)
    _reset_slack_state_for_tests()
    _reset_discord_state_for_tests()
    _reset_telegram_link_state_for_tests()


@pytest.fixture()
def request_user_override(client):
    def _install(user_id: int = 303) -> None:
        async def _fake_get_request_user() -> User:
            return User(id=user_id, username="integrations-tester", is_active=True)

        client.app.dependency_overrides[get_request_user] = _fake_get_request_user

    yield _install
    client.app.dependency_overrides.pop(get_request_user, None)


@pytest.fixture()
def service_override(client):
    client.app.dependency_overrides[get_integrations_control_plane_service] = lambda: _FakeIntegrationsControlPlaneService()
    yield
    client.app.dependency_overrides.pop(get_integrations_control_plane_service, None)


async def _get_user_secret_repo():
    pool = await get_db_pool()
    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.ensure_tables()
    return repo


def test_get_personal_integrations_returns_normalized_payload(client, auth_headers, principal_override, service_override):
    principal_override(_make_principal(active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))

    response = client.get("/api/v1/integrations/personal", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["scope"] == "personal"  # nosec B101 - pytest assertion
    assert [item["provider"] for item in body["items"]] == ["slack", "discord"]  # nosec B101 - pytest assertion


def test_get_workspace_integrations_returns_normalized_payload(client, auth_headers, principal_override, service_override):
    principal_override(_make_principal(active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))

    response = client.get("/api/v1/integrations/workspace", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["scope"] == "workspace"  # nosec B101 - pytest assertion
    assert [item["provider"] for item in body["items"]] == ["slack", "discord", "telegram"]  # nosec B101


def test_personal_slack_connect_route_returns_auth_url(
    client,
    auth_headers,
    principal_override,
    request_user_override,
    monkeypatch,
):
    principal_override(_make_principal(active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))
    request_user_override(303)

    async def _fake_start(**kwargs):
        assert int(kwargs["workspace_org_id"]) == 11  # nosec B101 - pytest assertion
        assert int(kwargs["user"].id) == 303  # nosec B101 - pytest assertion
        return {
            "ok": True,
            "status": "ready",
            "auth_url": "https://slack.example.test/oauth",
            "auth_session_id": "session-123",
            "expires_at": "2026-03-20T22:00:00+00:00",
        }

    monkeypatch.setattr(integrations_module, "slack_oauth_start_impl", _fake_start)

    response = client.post("/api/v1/integrations/personal/slack/connect", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["provider"] == "slack"  # nosec B101 - pytest assertion
    assert body["connection_id"] == "personal:slack"  # nosec B101 - pytest assertion
    assert body["auth_url"] == "https://slack.example.test/oauth"  # nosec B101 - pytest assertion


@pytest.mark.asyncio
async def test_personal_slack_update_route_disables_all_provider_installations(
    client,
    auth_headers,
    principal_override,
    request_user_override,
    monkeypatch,
):
    principal_override(_make_principal(user_id=1, active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))
    request_user_override(1)
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    now = datetime.now(timezone.utc)
    user_repo = await _get_user_secret_repo()
    await user_repo.upsert_secret(
        user_id=1,
        provider="slack",
        encrypted_blob=_encrypt_slack_payload(
            {
                "provider": "slack",
                "credential_version": 1,
                "installations": {
                    "T-11": {
                        "team_id": "T-11",
                        "team_name": "Slack Org 11",
                        "access_token": "xoxb-one",
                        "installed_at": "2026-03-20T18:30:00+00:00",
                        "installed_by": 1,
                        "disabled": False,
                    },
                    "T-12": {
                        "team_id": "T-12",
                        "team_name": "Slack Org 12",
                        "access_token": "xoxb-two",
                        "installed_at": "2026-03-20T18:45:00+00:00",
                        "installed_by": 1,
                        "disabled": False,
                    },
                },
            }
        ),
        key_hint="hint",
        metadata={"installation_count": 2, "active_installation_count": 2},
        updated_at=now,
        created_by=None,
        updated_by=None,
    )

    workspace_repo = await get_workspace_provider_installations_repo()
    await workspace_repo.upsert_installation(
        org_id=11,
        provider="slack",
        external_id="T-11",
        display_name="Slack Org 11",
        installed_by_user_id=1,
        disabled=False,
    )
    await workspace_repo.upsert_installation(
        org_id=11,
        provider="slack",
        external_id="T-12",
        display_name="Slack Org 12",
        installed_by_user_id=1,
        disabled=False,
    )

    response = client.patch(
        "/api/v1/integrations/personal/slack/personal:slack",
        headers=auth_headers,
        json={"enabled": False},
    )

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["provider"] == "slack"  # nosec B101 - pytest assertion
    assert body["status"] == "disabled"  # nosec B101 - pytest assertion
    assert body["enabled"] is False  # nosec B101 - pytest assertion

    stored_row = await user_repo.fetch_secret_for_user(1, "slack")
    payload = _decrypt_slack_payload(stored_row["encrypted_blob"])
    assert payload is not None  # nosec B101 - pytest assertion
    assert all(installation["disabled"] is True for installation in payload["installations"].values())  # nosec B101
    stored_metadata = stored_row["metadata"]
    if isinstance(stored_metadata, str):
        stored_metadata = json.loads(stored_metadata)
    assert stored_metadata["active_installation_count"] == 0  # nosec B101 - pytest assertion

    workspace_rows = await workspace_repo.list_installations(org_id=11, provider="slack", include_disabled=True)
    assert all(bool(row["disabled"]) is True for row in workspace_rows)  # nosec B101 - pytest assertion


@pytest.mark.asyncio
async def test_personal_discord_delete_route_removes_provider_secret_and_workspace_rows(
    client,
    auth_headers,
    principal_override,
    request_user_override,
    monkeypatch,
):
    principal_override(_make_principal(user_id=1, active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))
    request_user_override(1)
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    now = datetime.now(timezone.utc)
    user_repo = await _get_user_secret_repo()
    await user_repo.upsert_secret(
        user_id=1,
        provider="discord",
        encrypted_blob=_encrypt_discord_payload(
            {
                "provider": "discord",
                "credential_version": 1,
                "installations": {
                    "G-11": {
                        "guild_id": "G-11",
                        "guild_name": "Discord Org 11",
                        "access_token": "discord-one",
                        "installed_at": "2026-03-20T18:30:00+00:00",
                        "installed_by": 1,
                        "disabled": False,
                    }
                },
            }
        ),
        key_hint="hint",
        metadata={"installation_count": 1, "active_installation_count": 1},
        updated_at=now,
        created_by=None,
        updated_by=None,
    )

    workspace_repo = await get_workspace_provider_installations_repo()
    await workspace_repo.upsert_installation(
        org_id=11,
        provider="discord",
        external_id="G-11",
        display_name="Discord Org 11",
        installed_by_user_id=1,
        disabled=False,
    )

    response = client.delete(
        "/api/v1/integrations/personal/discord/personal:discord",
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["deleted"] is True  # nosec B101 - pytest assertion
    assert body["provider"] == "discord"  # nosec B101 - pytest assertion
    assert body["connection_id"] == "personal:discord"  # nosec B101 - pytest assertion

    stored_row = await user_repo.fetch_secret_for_user(1, "discord")
    assert stored_row is None  # nosec B101 - pytest assertion

    workspace_rows = await workspace_repo.list_installations(org_id=11, provider="discord", include_disabled=True)
    assert workspace_rows == []  # nosec B101 - pytest assertion


@pytest.mark.asyncio
async def test_workspace_slack_policy_routes_apply_to_registry_installations(client, auth_headers, principal_override):
    principal_override(_make_principal(active_org_id=17, active_team_id=22, org_ids=[17], team_ids=[22]))

    workspace_repo = await get_workspace_provider_installations_repo()
    await workspace_repo.upsert_installation(
        org_id=17,
        provider="slack",
        external_id="T-11",
        display_name="Slack Org 11",
        installed_by_user_id=303,
        disabled=False,
    )

    get_response = client.get("/api/v1/integrations/workspace/slack/policy", headers=auth_headers)

    assert get_response.status_code == 200, get_response.text  # nosec B101 - pytest assertion
    assert get_response.json()["installation_ids"] == ["T-11"]  # nosec B101 - pytest assertion

    put_response = client.put(
        "/api/v1/integrations/workspace/slack/policy",
        headers=auth_headers,
        json={
            "allowed_commands": ["help", "status"],
            "default_response_mode": "thread",
            "status_scope": "workspace_and_user",
        },
    )

    assert put_response.status_code == 200, put_response.text  # nosec B101 - pytest assertion
    body = put_response.json()
    assert body["provider"] == "slack"  # nosec B101 - pytest assertion
    assert body["policy"]["allowed_commands"] == ["help", "status"]  # nosec B101 - pytest assertion
    assert body["policy"]["default_response_mode"] == "thread"  # nosec B101 - pytest assertion


@pytest.mark.asyncio
async def test_workspace_discord_policy_routes_apply_to_registry_installations(client, auth_headers, principal_override):
    principal_override(_make_principal(active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))

    workspace_repo = await get_workspace_provider_installations_repo()
    await workspace_repo.upsert_installation(
        org_id=11,
        provider="discord",
        external_id="G-11",
        display_name="Discord Org 11",
        installed_by_user_id=303,
        disabled=False,
    )

    put_response = client.put(
        "/api/v1/integrations/workspace/discord/policy",
        headers=auth_headers,
        json={
            "allowed_commands": ["help"],
            "default_response_mode": "channel",
            "status_scope": "guild_and_user",
        },
    )

    assert put_response.status_code == 200, put_response.text  # nosec B101 - pytest assertion
    body = put_response.json()
    assert body["provider"] == "discord"  # nosec B101 - pytest assertion
    assert body["installation_ids"] == ["G-11"]  # nosec B101 - pytest assertion
    assert body["policy"]["allowed_commands"] == ["help"]  # nosec B101 - pytest assertion
    assert body["policy"]["default_response_mode"] == "channel"  # nosec B101 - pytest assertion


@pytest.mark.asyncio
async def test_workspace_telegram_linked_actor_routes_proxy_admin_management(client, auth_headers, principal_override):
    principal_override(_make_principal(active_org_id=11, active_team_id=22, org_ids=[11], team_ids=[22]))

    runtime_repo = await get_telegram_runtime_repo()
    seeded = await runtime_repo.upsert_actor_link(
        scope_type="team",
        scope_id=22,
        telegram_user_id=123456,
        auth_user_id=77,
        telegram_username="linked",
    )

    list_response = client.get("/api/v1/integrations/workspace/telegram/linked-actors", headers=auth_headers)
    assert list_response.status_code == 200, list_response.text  # nosec B101 - pytest assertion
    assert list_response.json()["items"][0]["id"] == seeded["id"]  # nosec B101 - pytest assertion

    delete_response = client.delete(
        f"/api/v1/integrations/workspace/telegram/linked-actors/{seeded['id']}",
        headers=auth_headers,
    )
    assert delete_response.status_code == 200, delete_response.text  # nosec B101 - pytest assertion
    assert delete_response.json()["deleted"] is True  # nosec B101 - pytest assertion
