from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import discord as discord_endpoint
from tldw_Server_API.app.api.v1.endpoints import discord_support
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.tests._chatops_helpers.fake_oauth_repos import (
    FakeOAuthStateRepo as _FakeOAuthStateRepo,
    FakeUserSecretRepo as _FakeUserSecretRepo,
)



@pytest.fixture()
def discord_oauth_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]:
    state_repo = _FakeOAuthStateRepo()
    user_repo = _FakeUserSecretRepo()

    monkeypatch.setenv("DISCORD_CLIENT_ID", "D123")
    monkeypatch.setenv("DISCORD_CLIENT_SECRET", "S123")
    monkeypatch.setenv("DISCORD_OAUTH_REDIRECT_URI", "https://example.com/api/v1/discord/oauth/callback")
    monkeypatch.setenv("DISCORD_OAUTH_AUTH_URL", "https://discord.test/oauth2/authorize")
    monkeypatch.setenv("DISCORD_OAUTH_TOKEN_URL", "https://discord.test/api/oauth2/token")
    monkeypatch.setenv("DISCORD_OAUTH_SCOPE", "bot applications.commands")

    async def _get_state_repo() -> _FakeOAuthStateRepo:
        return state_repo

    async def _get_user_repo() -> _FakeUserSecretRepo:
        return user_repo

    async def _token_exchange(*, token_url: str, form_data: dict) -> dict:
        assert token_url == "https://discord.test/api/oauth2/token"
        assert form_data["client_id"] == "D123"
        assert form_data["client_secret"] == "S123"
        return {
            "access_token": "discord-access-token",
            "refresh_token": "discord-refresh-token",
            "scope": "bot applications.commands",
            "guild": {"id": "G123", "name": "Guild 123"},
        }

    monkeypatch.setattr(discord_endpoint, "_get_oauth_state_repo", _get_state_repo)
    monkeypatch.setattr(discord_endpoint, "_get_user_secret_repo", _get_user_repo)
    monkeypatch.setattr(discord_endpoint, "_discord_oauth_token_exchange", _token_exchange)
    monkeypatch.setattr(discord_endpoint, "_encrypt_discord_payload", lambda payload: json.dumps(payload))
    monkeypatch.setattr(
        discord_endpoint,
        "_decrypt_discord_payload",
        lambda encrypted_blob: json.loads(encrypted_blob) if encrypted_blob else None,
    )

    app = FastAPI()
    app.include_router(discord_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    return TestClient(app), state_repo, user_repo


def _extract_state_from_auth_url(auth_url: str) -> str:
    parsed = urlparse(auth_url)
    query = parse_qs(parsed.query)
    state_values = query.get("state") or []
    assert state_values
    return str(state_values[0])


def test_discord_oauth_start_returns_auth_url_and_state(discord_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, state_repo, _ = discord_oauth_client
    response = client.post("/api/v1/discord/oauth/start")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "https://discord.test/oauth2/authorize?" in payload["auth_url"]

    state = _extract_state_from_auth_url(payload["auth_url"])
    stored = state_repo.states.get(state)
    assert stored is not None
    assert stored["provider"] == "discord"
    assert stored["user_id"] == 1


def test_discord_oauth_callback_persists_installation_and_lists(discord_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, state_repo, user_repo = discord_oauth_client
    start = client.post("/api/v1/discord/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    assert state in state_repo.states

    callback = client.get("/api/v1/discord/oauth/callback", params={"code": "abc", "state": state})
    assert callback.status_code == 200
    assert callback.json()["status"] == "installed"
    assert callback.json()["guild_id"] == "G123"
    assert state not in state_repo.states

    assert user_repo.row is not None
    secret_payload = json.loads(user_repo.row["encrypted_blob"])
    assert secret_payload["installations"]["G123"]["access_token"] == "discord-access-token"

    listed = client.get("/api/v1/discord/admin/installations")
    assert listed.status_code == 200
    items = listed.json()["installations"]
    assert len(items) == 1
    assert items[0]["guild_id"] == "G123"
    assert items[0]["disabled"] is False
    assert "access_token" not in items[0]


def test_discord_oauth_callback_rejects_invalid_state(discord_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, _, _ = discord_oauth_client
    response = client.get("/api/v1/discord/oauth/callback", params={"code": "abc", "state": "missing"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid or expired OAuth state"


def test_discord_admin_toggle_and_delete(discord_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, _, _ = discord_oauth_client
    start = client.post("/api/v1/discord/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    callback = client.get("/api/v1/discord/oauth/callback", params={"code": "abc", "state": state})
    assert callback.status_code == 200

    toggle = client.put("/api/v1/discord/admin/installations/G123", json={"disabled": True})
    assert toggle.status_code == 200
    assert toggle.json()["disabled"] is True

    listed = client.get("/api/v1/discord/admin/installations")
    assert listed.status_code == 200
    assert listed.json()["installations"][0]["disabled"] is True

    deleted = client.delete("/api/v1/discord/admin/installations/G123")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    listed_after_delete = client.get("/api/v1/discord/admin/installations")
    assert listed_after_delete.status_code == 200
    assert listed_after_delete.json()["installations"] == []


@pytest.mark.asyncio
async def test_discord_oauth_token_exchange_sanitizes_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 400

        def json(self) -> dict[str, str]:
            return {"error_description": "client authentication failed"}

        async def aclose(self) -> None:
            return None

    async def _fake_http_afetch(**_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(discord_support, "_http_afetch", _fake_http_afetch)

    with pytest.raises(HTTPException) as exc_info:
        await discord_endpoint._discord_oauth_token_exchange(
            token_url="https://discord.test/api/oauth2/token",
            form_data={"code": "bad-code"},
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Discord OAuth token exchange failed"
