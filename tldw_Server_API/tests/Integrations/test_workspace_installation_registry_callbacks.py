from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import discord as discord_endpoint
from tldw_Server_API.app.api.v1.endpoints import slack as slack_endpoint
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


class _FakeOAuthStateRepo:
    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    async def create_state(
        self,
        *,
        state: str,
        user_id: int,
        provider: str,
        auth_session_id: str,
        redirect_uri: str,
        pkce_verifier_encrypted: str,
        expires_at,
        return_path=None,
        created_at=None,
    ) -> dict:
        record = {
            "state": state,
            "user_id": int(user_id),
            "provider": provider,
            "auth_session_id": auth_session_id,
            "redirect_uri": redirect_uri,
            "pkce_verifier_encrypted": pkce_verifier_encrypted,
            "expires_at": expires_at,
            "return_path": return_path,
            "created_at": created_at or datetime.now(timezone.utc),
        }
        self.states[state] = record
        return record

    async def consume_state(
        self,
        *,
        state: str,
        provider: str,
        consumed_at=None,
    ) -> dict | None:
        record = self.states.pop(state, None)
        if not record:
            return None
        if record.get("provider") != provider:
            return None
        return record


class _FakeUserSecretRepo:
    def __init__(self) -> None:
        self.row: dict | None = None

    async def fetch_secret_for_user(
        self,
        user_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict | None:
        if not self.row:
            return None
        return dict(self.row)

    async def upsert_secret(
        self,
        *,
        user_id: int,
        provider: str,
        encrypted_blob: str,
        key_hint: str | None,
        metadata: dict | None,
        updated_at,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict:
        self.row = {
            "user_id": int(user_id),
            "provider": provider,
            "encrypted_blob": encrypted_blob,
            "key_hint": key_hint,
            "metadata": metadata,
            "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at),
            "created_by": created_by,
            "updated_by": updated_by,
        }
        return dict(self.row)

    async def delete_secret(
        self,
        user_id: int,
        provider: str,
        *,
        revoked_by: int | None = None,
        revoked_at=None,
    ) -> bool:
        self.row = None
        return True


async def _make_registry_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos import get_workspace_provider_installations_repo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))
    repo = await get_workspace_provider_installations_repo()
    return repo


def _install_request_org_context(request: Request, *, active_org_id: int, org_ids: list[int]) -> None:
    request.state.active_org_id = active_org_id
    request.state.org_ids = list(org_ids)


def _extract_state_from_auth_url(auth_url: str) -> str:
    parsed = urlparse(auth_url)
    query = parse_qs(parsed.query)
    state_values = query.get("state") or []
    if not state_values:
        raise AssertionError("expected state value in auth url")
    return str(state_values[0])


@pytest.fixture()
async def slack_oauth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_services) -> tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo, Any]:
    state_repo = _FakeOAuthStateRepo()
    user_repo = _FakeUserSecretRepo()
    registry_repo = await _make_registry_repo(tmp_path, monkeypatch)
    selected_org_id = 11
    fallback_org_id = 7

    monkeypatch.setenv("SLACK_CLIENT_ID", "C123")
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "S123")
    monkeypatch.setenv("SLACK_OAUTH_REDIRECT_URI", "https://example.com/api/v1/slack/oauth/callback")
    monkeypatch.setenv("SLACK_OAUTH_AUTH_URL", "https://slack.test/oauth/v2/authorize")
    monkeypatch.setenv("SLACK_OAUTH_TOKEN_URL", "https://slack.test/api/oauth.v2.access")  # nosec B105
    monkeypatch.setenv("SLACK_OAUTH_SCOPES", "commands,chat:write")

    async def _get_state_repo() -> _FakeOAuthStateRepo:
        return state_repo

    async def _get_user_repo() -> _FakeUserSecretRepo:
        return user_repo

    async def _get_registry_repo():
        return registry_repo

    async def _get_request_user(request: Request):
        _install_request_org_context(request, active_org_id=selected_org_id, org_ids=[selected_org_id, fallback_org_id])
        return SimpleNamespace(id=1, org_ids=[selected_org_id, fallback_org_id], active_org_id=selected_org_id)

    async def _get_admin_principal(request: Request):
        _install_request_org_context(request, active_org_id=selected_org_id, org_ids=[selected_org_id, fallback_org_id])
        return SimpleNamespace(
            user_id=1,
            roles=["admin"],
            permissions=["*"],
            org_ids=[selected_org_id, fallback_org_id],
            active_org_id=selected_org_id,
            team_ids=[],
            kind="user",
            subject="user",
        )

    async def _token_exchange(*, token_url: str, form_data: dict) -> dict:
        if token_url != "https://slack.test/api/oauth.v2.access":  # nosec B105
            raise AssertionError(f"unexpected token_url: {token_url}")
        if form_data["client_id"] != "C123":
            raise AssertionError(f"unexpected client_id: {form_data['client_id']}")
        if form_data["client_secret"] != "S123":
            raise AssertionError(f"unexpected client_secret: {form_data['client_secret']}")
        return {
            "ok": True,
            "access_token": "xoxb-test-token",  # nosec B105
            "scope": "commands,chat:write",
            "bot_user_id": "B123",
            "team": {"id": "T123", "name": "Team 123"},
            "authed_user": {"id": "U456"},
        }

    async def _resolve_org_id_for_user(user_id: int) -> int:
        return fallback_org_id

    async def _list_org_memberships_for_user(user_id: int) -> list[dict[str, Any]]:
        return [
            {"org_id": fallback_org_id, "status": "active"},
            {"org_id": selected_org_id, "status": "active"},
        ]

    monkeypatch.setattr(slack_endpoint, "_get_oauth_state_repo", _get_state_repo)
    monkeypatch.setattr(slack_endpoint, "_get_user_secret_repo", _get_user_repo)
    monkeypatch.setattr(slack_endpoint, "_get_workspace_provider_installations_repo", _get_registry_repo, raising=False)
    monkeypatch.setattr(slack_endpoint, "_slack_oauth_token_exchange", _token_exchange)
    monkeypatch.setattr(slack_endpoint, "list_org_memberships_for_user", _list_org_memberships_for_user, raising=False)
    monkeypatch.setattr(slack_endpoint, "_encrypt_slack_payload", lambda payload: json.dumps(payload))
    monkeypatch.setattr(
        slack_endpoint,
        "_decrypt_slack_payload",
        lambda encrypted_blob: json.loads(encrypted_blob) if encrypted_blob else None,
    )

    app = FastAPI()
    app.include_router(slack_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = _get_request_user
    app.dependency_overrides[get_auth_principal] = _get_admin_principal
    return TestClient(app), state_repo, user_repo, registry_repo, selected_org_id


@pytest.fixture()
async def discord_oauth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_services) -> tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo, Any]:
    state_repo = _FakeOAuthStateRepo()
    user_repo = _FakeUserSecretRepo()
    registry_repo = await _make_registry_repo(tmp_path, monkeypatch)
    selected_org_id = 11
    fallback_org_id = 7

    monkeypatch.setenv("DISCORD_CLIENT_ID", "D123")
    monkeypatch.setenv("DISCORD_CLIENT_SECRET", "S123")
    monkeypatch.setenv("DISCORD_OAUTH_REDIRECT_URI", "https://example.com/api/v1/discord/oauth/callback")
    monkeypatch.setenv("DISCORD_OAUTH_AUTH_URL", "https://discord.test/oauth2/authorize")
    monkeypatch.setenv("DISCORD_OAUTH_TOKEN_URL", "https://discord.test/api/oauth2/token")  # nosec B105
    monkeypatch.setenv("DISCORD_OAUTH_SCOPE", "bot applications.commands")

    async def _get_state_repo() -> _FakeOAuthStateRepo:
        return state_repo

    async def _get_user_repo() -> _FakeUserSecretRepo:
        return user_repo

    async def _get_registry_repo():
        return registry_repo

    async def _get_request_user(request: Request):
        _install_request_org_context(request, active_org_id=selected_org_id, org_ids=[selected_org_id, fallback_org_id])
        return SimpleNamespace(id=1, org_ids=[selected_org_id, fallback_org_id], active_org_id=selected_org_id)

    async def _get_admin_principal(request: Request):
        _install_request_org_context(request, active_org_id=selected_org_id, org_ids=[selected_org_id, fallback_org_id])
        return SimpleNamespace(
            user_id=1,
            roles=["admin"],
            permissions=["*"],
            org_ids=[selected_org_id, fallback_org_id],
            active_org_id=selected_org_id,
            team_ids=[],
            kind="user",
            subject="user",
        )

    async def _token_exchange(*, token_url: str, form_data: dict) -> dict:
        if token_url != "https://discord.test/api/oauth2/token":  # nosec B105
            raise AssertionError(f"unexpected token_url: {token_url}")
        if form_data["client_id"] != "D123":
            raise AssertionError(f"unexpected client_id: {form_data['client_id']}")
        if form_data["client_secret"] != "S123":
            raise AssertionError(f"unexpected client_secret: {form_data['client_secret']}")
        return {
            "access_token": "discord-access-token",  # nosec B105
            "refresh_token": "discord-refresh-token",  # nosec B105
            "scope": "bot applications.commands",
            "guild": {"id": "G123", "name": "Guild 123"},
        }

    async def _resolve_org_id_for_user(user_id: int) -> int:
        return fallback_org_id

    async def _list_org_memberships_for_user(user_id: int) -> list[dict[str, Any]]:
        return [
            {"org_id": fallback_org_id, "status": "active"},
            {"org_id": selected_org_id, "status": "active"},
        ]

    monkeypatch.setattr(discord_endpoint, "_get_oauth_state_repo", _get_state_repo)
    monkeypatch.setattr(discord_endpoint, "_get_user_secret_repo", _get_user_repo)
    monkeypatch.setattr(discord_endpoint, "_get_workspace_provider_installations_repo", _get_registry_repo, raising=False)
    monkeypatch.setattr(discord_endpoint, "_discord_oauth_token_exchange", _token_exchange)
    monkeypatch.setattr(discord_endpoint, "list_org_memberships_for_user", _list_org_memberships_for_user, raising=False)
    monkeypatch.setattr(discord_endpoint, "_encrypt_discord_payload", lambda payload: json.dumps(payload))
    monkeypatch.setattr(
        discord_endpoint,
        "_decrypt_discord_payload",
        lambda encrypted_blob: json.loads(encrypted_blob) if encrypted_blob else None,
    )

    app = FastAPI()
    app.include_router(discord_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = _get_request_user
    app.dependency_overrides[get_auth_principal] = _get_admin_principal
    return TestClient(app), state_repo, user_repo, registry_repo, selected_org_id


@pytest.mark.asyncio
async def test_slack_oauth_callback_persists_registry_row(slack_oauth_client):
    client, state_repo, user_repo, registry_repo, selected_org_id = slack_oauth_client
    start = client.post("/api/v1/slack/oauth/start")
    if start.status_code != 200:
        raise AssertionError(f"unexpected status: {start.status_code}")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    stored_state = json.loads(state_repo.states[state]["pkce_verifier_encrypted"])
    if stored_state.get("org_id") != selected_org_id:
        raise AssertionError(f"unexpected org_id in state: {stored_state}")

    callback = client.get("/api/v1/slack/oauth/callback", params={"code": "abc", "state": state})
    if callback.status_code != 200:
        raise AssertionError(f"unexpected status: {callback.status_code}")
    if callback.json()["status"] != "installed":
        raise AssertionError(f"unexpected status payload: {callback.json()}")
    if callback.json()["team_id"] != "T123":
        raise AssertionError(f"unexpected team_id payload: {callback.json()}")

    rows = await registry_repo.list_installations(org_id=selected_org_id, provider="slack")
    if not rows:
        raise AssertionError("expected registry row")
    if rows[0]["provider"] != "slack":
        raise AssertionError(f"unexpected provider: {rows[0]['provider']}")
    if rows[0]["org_id"] != selected_org_id:
        raise AssertionError(f"unexpected org_id: {rows[0]['org_id']}")
    if rows[0]["external_id"] != "T123":
        raise AssertionError(f"unexpected external_id: {rows[0]['external_id']}")
    if rows[0]["display_name"] != "Team 123":
        raise AssertionError(f"unexpected display_name: {rows[0]['display_name']}")
    if rows[0]["installed_by_user_id"] != 1:
        raise AssertionError(f"unexpected installed_by_user_id: {rows[0]['installed_by_user_id']}")
    if rows[0]["disabled"] is not False:
        raise AssertionError(f"unexpected disabled state: {rows[0]['disabled']}")


@pytest.mark.asyncio
async def test_slack_admin_toggle_and_delete_updates_registry_state(slack_oauth_client):
    client, _, _, registry_repo, selected_org_id = slack_oauth_client
    start = client.post("/api/v1/slack/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    callback = client.get("/api/v1/slack/oauth/callback", params={"code": "abc", "state": state})
    if callback.status_code != 200:
        raise AssertionError(f"unexpected status: {callback.status_code}")

    toggle = client.put("/api/v1/slack/admin/installations/T123", json={"disabled": True})
    if toggle.status_code != 200:
        raise AssertionError(f"unexpected toggle status: {toggle.status_code}")
    if toggle.json()["disabled"] is not True:
        raise AssertionError(f"unexpected toggle payload: {toggle.json()}")

    rows = await registry_repo.list_installations(org_id=selected_org_id, provider="slack")
    if rows[0]["disabled"] is not True:
        raise AssertionError(f"expected disabled row, found {rows[0]}")

    deleted = client.delete("/api/v1/slack/admin/installations/T123")
    if deleted.status_code != 200:
        raise AssertionError(f"unexpected delete status: {deleted.status_code}")
    if deleted.json()["status"] != "deleted":
        raise AssertionError(f"unexpected delete payload: {deleted.json()}")

    remaining = await registry_repo.list_installations(org_id=selected_org_id, provider="slack")
    if remaining != []:
        raise AssertionError(f"expected empty registry, found {remaining}")


@pytest.mark.asyncio
async def test_discord_oauth_callback_persists_registry_row(discord_oauth_client):
    client, state_repo, user_repo, registry_repo, selected_org_id = discord_oauth_client
    start = client.post("/api/v1/discord/oauth/start")
    if start.status_code != 200:
        raise AssertionError(f"unexpected status: {start.status_code}")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    stored_state = json.loads(state_repo.states[state]["pkce_verifier_encrypted"])
    if stored_state.get("org_id") != selected_org_id:
        raise AssertionError(f"unexpected org_id in state: {stored_state}")

    callback = client.get("/api/v1/discord/oauth/callback", params={"code": "abc", "state": state})
    if callback.status_code != 200:
        raise AssertionError(f"unexpected status: {callback.status_code}")
    if callback.json()["status"] != "installed":
        raise AssertionError(f"unexpected status payload: {callback.json()}")
    if callback.json()["guild_id"] != "G123":
        raise AssertionError(f"unexpected guild_id payload: {callback.json()}")

    rows = await registry_repo.list_installations(org_id=selected_org_id, provider="discord")
    if not rows:
        raise AssertionError("expected registry row")
    if rows[0]["provider"] != "discord":
        raise AssertionError(f"unexpected provider: {rows[0]['provider']}")
    if rows[0]["org_id"] != selected_org_id:
        raise AssertionError(f"unexpected org_id: {rows[0]['org_id']}")
    if rows[0]["external_id"] != "G123":
        raise AssertionError(f"unexpected external_id: {rows[0]['external_id']}")
    if rows[0]["display_name"] != "Guild 123":
        raise AssertionError(f"unexpected display_name: {rows[0]['display_name']}")
    if rows[0]["installed_by_user_id"] != 1:
        raise AssertionError(f"unexpected installed_by_user_id: {rows[0]['installed_by_user_id']}")
    if rows[0]["disabled"] is not False:
        raise AssertionError(f"unexpected disabled state: {rows[0]['disabled']}")


@pytest.mark.asyncio
async def test_discord_admin_toggle_and_delete_updates_registry_state(discord_oauth_client):
    client, _, _, registry_repo, selected_org_id = discord_oauth_client
    start = client.post("/api/v1/discord/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    callback = client.get("/api/v1/discord/oauth/callback", params={"code": "abc", "state": state})
    if callback.status_code != 200:
        raise AssertionError(f"unexpected status: {callback.status_code}")

    toggle = client.put("/api/v1/discord/admin/installations/G123", json={"disabled": True})
    if toggle.status_code != 200:
        raise AssertionError(f"unexpected toggle status: {toggle.status_code}")
    if toggle.json()["disabled"] is not True:
        raise AssertionError(f"unexpected toggle payload: {toggle.json()}")

    rows = await registry_repo.list_installations(org_id=selected_org_id, provider="discord")
    if rows[0]["disabled"] is not True:
        raise AssertionError(f"expected disabled row, found {rows[0]}")

    deleted = client.delete("/api/v1/discord/admin/installations/G123")
    if deleted.status_code != 200:
        raise AssertionError(f"unexpected delete status: {deleted.status_code}")
    if deleted.json()["status"] != "deleted":
        raise AssertionError(f"unexpected delete payload: {deleted.json()}")

    remaining = await registry_repo.list_installations(org_id=selected_org_id, provider="discord")
    if remaining != []:
        raise AssertionError(f"expected empty registry, found {remaining}")
