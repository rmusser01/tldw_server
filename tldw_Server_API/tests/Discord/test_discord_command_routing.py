from __future__ import annotations

import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import discord as discord_endpoint


def _make_test_signer() -> tuple[Ed25519PrivateKey, str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_key, public_key.hex()


def _sign(private_key: Ed25519PrivateKey, timestamp: int, body: bytes) -> str:
    return private_key.sign(str(timestamp).encode("utf-8") + body).hex()


def _as_web_user_in_installing_org(monkeypatch: pytest.MonkeyPatch, module, app: FastAPI, provider: str, external_id: str) -> None:
    """The job-status route needs a logged-in user in the org that installed the tenant."""
    from types import SimpleNamespace

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user

    async def _memberships(_user_id: int):
        return [{"org_id": 5, "status": "active"}]

    installed = {(5, provider): [external_id]}

    class _Repo:
        async def list_installations(self, *, org_id: int, provider: str | None = None, **_kwargs):
            return [{"external_id": ext} for ext in installed.get((org_id, provider), [])]

    async def _repo():
        return _Repo()

    monkeypatch.setattr(module, "list_org_memberships_for_user", _memberships)
    monkeypatch.setattr(module, "_get_workspace_provider_installations_repo", _repo)
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(AUTH_MODE="multi_user"))
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)


@pytest.fixture()
def discord_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Ed25519PrivateKey]:
    private_key, public_key_hex = _make_test_signer()
    monkeypatch.setenv("DISCORD_PUBLIC_KEY", public_key_hex)
    monkeypatch.setenv("DISCORD_REPLAY_WINDOW_SECONDS", "300")
    monkeypatch.setenv("DISCORD_INGRESS_RATE_LIMIT_PER_MINUTE", "1000")
    discord_endpoint._reset_discord_state_for_tests()

    app = FastAPI()
    app.include_router(discord_endpoint.router, prefix="/api/v1")
    _as_web_user_in_installing_org(monkeypatch, discord_endpoint, app, "discord", "guild-1")
    return TestClient(app), private_key


def test_discord_command_parse_rag_route(discord_client: tuple[TestClient, Ed25519PrivateKey]) -> None:
    client, signer = discord_client
    payload = {
        "type": 2,
        "id": "cmd-1",
        "application_id": "app-1",
        "guild_id": "guild-1",
        "data": {
            "name": "tldw",
            "options": [
                {"name": "rag", "value": "release notes"},
            ],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    ts = int(time.time())
    headers = {
        "x-signature-timestamp": str(ts),
        "x-signature-ed25519": _sign(signer, ts, body),
        "content-type": "application/json",
    }

    response = client.post("/api/v1/discord/interactions", data=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["parsed"]["action"] == "rag"
    assert data["parsed"]["route"] == "rag.search"
    assert data["parsed"]["input"] == "release notes"
    job_id = data["job_id"]
    assert isinstance(job_id, int)
    assert data["response_mode"] == "ephemeral"

    job_status = client.get(f"/api/v1/discord/jobs/{job_id}")
    assert job_status.status_code == 200
    assert job_status.json()["job"]["id"] == job_id


def test_discord_command_status_queries_job(discord_client: tuple[TestClient, Ed25519PrivateKey]) -> None:
    client, signer = discord_client
    queue_payload = {
        "type": 2,
        "id": "cmd-status-queue",
        "application_id": "app-1",
        "guild_id": "guild-1",
        "data": {
            "name": "tldw",
            "options": [
                {"name": "ask", "value": "hello"},
            ],
        },
    }
    queue_body = json.dumps(queue_payload).encode("utf-8")
    queue_ts = int(time.time())
    queue_headers = {
        "x-signature-timestamp": str(queue_ts),
        "x-signature-ed25519": _sign(signer, queue_ts, queue_body),
        "content-type": "application/json",
    }
    queued = client.post("/api/v1/discord/interactions", data=queue_body, headers=queue_headers)
    assert queued.status_code == 200
    queued_payload = queued.json()
    job_id = queued_payload["job_id"]
    assert isinstance(job_id, int)

    status_payload = {
        "type": 2,
        "id": "cmd-status-query",
        "application_id": "app-1",
        "guild_id": "guild-1",
        "data": {
            "name": "tldw",
            "options": [
                {"name": "status", "value": str(job_id)},
            ],
        },
    }
    status_body = json.dumps(status_payload).encode("utf-8")
    status_ts = int(time.time())
    status_headers = {
        "x-signature-timestamp": str(status_ts),
        "x-signature-ed25519": _sign(signer, status_ts, status_body),
        "content-type": "application/json",
    }
    status_response = client.post("/api/v1/discord/interactions", data=status_body, headers=status_headers)
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "accepted"
    assert status_data["job"]["id"] == job_id


def test_discord_command_unknown_returns_usage(discord_client: tuple[TestClient, Ed25519PrivateKey]) -> None:
    client, signer = discord_client
    payload = {
        "type": 2,
        "id": "cmd-2",
        "application_id": "app-1",
        "guild_id": "guild-1",
        "data": {"name": "foobar"},
    }
    body = json.dumps(payload).encode("utf-8")
    ts = int(time.time())
    headers = {
        "x-signature-timestamp": str(ts),
        "x-signature-ed25519": _sign(signer, ts, body),
        "content-type": "application/json",
    }

    response = client.post("/api/v1/discord/interactions", data=body, headers=headers)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "unknown_command"
    assert "Supported commands" in data["usage"]


def test_discord_command_defaults_to_ask_when_missing_data(discord_client: tuple[TestClient, Ed25519PrivateKey]) -> None:
    client, signer = discord_client
    payload = {
        "type": 2,
        "id": "cmd-3",
        "application_id": "app-1",
        "guild_id": "guild-1",
    }
    body = json.dumps(payload).encode("utf-8")
    ts = int(time.time())
    headers = {
        "x-signature-timestamp": str(ts),
        "x-signature-ed25519": _sign(signer, ts, body),
        "content-type": "application/json",
    }

    response = client.post("/api/v1/discord/interactions", data=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["parsed"]["action"] == "ask"
    assert data["parsed"]["route"] == "chat.ask"
