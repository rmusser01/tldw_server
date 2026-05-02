from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import discord as discord_endpoint
from tldw_Server_API.app.api.v1.endpoints import discord_support
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

    async def fetch_secret_for_user(self, user_id: int, provider: str, *, include_revoked: bool = False) -> dict | None:
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
