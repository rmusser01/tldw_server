from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import slack as slack_endpoint
from tldw_Server_API.app.api.v1.endpoints import slack_support
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.tests._chatops_helpers.fake_oauth_repos import (
    FakeOAuthStateRepo as _FakeOAuthStateRepo,
    FakeUserSecretRepo as _FakeUserSecretRepo,
)



@pytest.fixture()
def slack_oauth_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]:
    state_repo = _FakeOAuthStateRepo()
    user_repo = _FakeUserSecretRepo()

    monkeypatch.setenv("SLACK_CLIENT_ID", "C123")
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "S123")
    monkeypatch.setenv("SLACK_OAUTH_REDIRECT_URI", "https://example.com/api/v1/slack/oauth/callback")
    monkeypatch.setenv("SLACK_OAUTH_AUTH_URL", "https://slack.test/oauth/v2/authorize")
    monkeypatch.setenv("SLACK_OAUTH_TOKEN_URL", "https://slack.test/api/oauth.v2.access")
    monkeypatch.setenv("SLACK_OAUTH_SCOPES", "commands,chat:write")

    async def _get_state_repo() -> _FakeOAuthStateRepo:
        return state_repo

    async def _get_user_repo() -> _FakeUserSecretRepo:
        return user_repo

    async def _token_exchange(*, token_url: str, form_data: dict) -> dict:
        assert token_url == "https://slack.test/api/oauth.v2.access"
        assert form_data["client_id"] == "C123"
        assert form_data["client_secret"] == "S123"
        return {
            "ok": True,
            "access_token": "xoxb-test-token",
            "scope": "commands,chat:write",
            "bot_user_id": "B123",
            "team": {"id": "T123", "name": "Team 123"},
            "authed_user": {"id": "U456"},
        }

    monkeypatch.setattr(slack_endpoint, "_get_oauth_state_repo", _get_state_repo)
    monkeypatch.setattr(slack_endpoint, "_get_user_secret_repo", _get_user_repo)
    monkeypatch.setattr(slack_endpoint, "_slack_oauth_token_exchange", _token_exchange)
    monkeypatch.setattr(slack_endpoint, "_encrypt_slack_payload", lambda payload: json.dumps(payload))
    monkeypatch.setattr(
        slack_endpoint,
        "_decrypt_slack_payload",
        lambda encrypted_blob: json.loads(encrypted_blob) if encrypted_blob else None,
    )

    app = FastAPI()
    app.include_router(slack_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    return TestClient(app), state_repo, user_repo


def _extract_state_from_auth_url(auth_url: str) -> str:
    parsed = urlparse(auth_url)
    query = parse_qs(parsed.query)
    state_values = query.get("state") or []
    assert state_values
    return str(state_values[0])


def test_slack_oauth_start_returns_auth_url_and_state(slack_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, state_repo, _ = slack_oauth_client
    response = client.post("/api/v1/slack/oauth/start")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "https://slack.test/oauth/v2/authorize?" in payload["auth_url"]

    state = _extract_state_from_auth_url(payload["auth_url"])
    stored = state_repo.states.get(state)
    assert stored is not None
    assert stored["provider"] == "slack"
    assert stored["user_id"] == 1


def test_slack_oauth_callback_persists_installation_and_lists(slack_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, state_repo, user_repo = slack_oauth_client
    start = client.post("/api/v1/slack/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    assert state in state_repo.states

    callback = client.get("/api/v1/slack/oauth/callback", params={"code": "abc", "state": state})
    assert callback.status_code == 200
    assert callback.json()["status"] == "installed"
    assert callback.json()["team_id"] == "T123"
    assert state not in state_repo.states

    assert user_repo.row is not None
    secret_payload = json.loads(user_repo.row["encrypted_blob"])
    assert secret_payload["installations"]["T123"]["access_token"] == "xoxb-test-token"

    listed = client.get("/api/v1/slack/admin/installations")
    assert listed.status_code == 200
    items = listed.json()["installations"]
    assert len(items) == 1
    assert items[0]["team_id"] == "T123"
    assert items[0]["disabled"] is False
    assert "access_token" not in items[0]


def test_slack_oauth_callback_rejects_invalid_state(slack_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, _, _ = slack_oauth_client
    response = client.get("/api/v1/slack/oauth/callback", params={"code": "abc", "state": "missing"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid or expired OAuth state"


def test_slack_oauth_callback_sanitizes_provider_error(
    slack_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = slack_oauth_client

    async def _failed_token_exchange(*, token_url: str, form_data: dict) -> dict:
        assert token_url == "https://slack.test/api/oauth.v2.access"
        assert form_data["client_id"] == "C123"
        return {"ok": False, "error": "invalid_code"}

    monkeypatch.setattr(slack_endpoint, "_slack_oauth_token_exchange", _failed_token_exchange)

    start = client.post("/api/v1/slack/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])

    response = client.get("/api/v1/slack/oauth/callback", params={"code": "bad-code", "state": state})

    assert response.status_code == 502
    assert response.json()["detail"] == "Slack OAuth token exchange failed"


def test_slack_admin_toggle_and_delete(slack_oauth_client: tuple[TestClient, _FakeOAuthStateRepo, _FakeUserSecretRepo]) -> None:
    client, _, _ = slack_oauth_client
    start = client.post("/api/v1/slack/oauth/start")
    state = _extract_state_from_auth_url(start.json()["auth_url"])
    callback = client.get("/api/v1/slack/oauth/callback", params={"code": "abc", "state": state})
    assert callback.status_code == 200

    toggle = client.put("/api/v1/slack/admin/installations/T123", json={"disabled": True})
    assert toggle.status_code == 200
    assert toggle.json()["disabled"] is True

    listed = client.get("/api/v1/slack/admin/installations")
    assert listed.status_code == 200
    assert listed.json()["installations"][0]["disabled"] is True

    deleted = client.delete("/api/v1/slack/admin/installations/T123")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    listed_after_delete = client.get("/api/v1/slack/admin/installations")
    assert listed_after_delete.status_code == 200
    assert listed_after_delete.json()["installations"] == []


@pytest.mark.asyncio
async def test_slack_oauth_token_exchange_sanitizes_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 400

        def json(self) -> dict[str, str]:
            return {"error": "invalid_client"}

        async def aclose(self) -> None:
            return None

    async def _fake_http_afetch(**_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(slack_support, "_http_afetch", _fake_http_afetch)

    with pytest.raises(HTTPException) as exc_info:
        await slack_endpoint._slack_oauth_token_exchange(
            token_url="https://slack.test/api/oauth.v2.access",
            form_data={"code": "bad-code"},
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Slack OAuth token exchange failed"
