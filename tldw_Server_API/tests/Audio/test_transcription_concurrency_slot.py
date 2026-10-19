"""POST /audio/transcriptions holds a per-user slot (limits.audio_concurrent_transcriptions) for the whole request (TASK-13435)."""

import asyncio
import io
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor, RGRequest
from tldw_Server_API.app.core.Usage import concurrency_quota

pytestmark = pytest.mark.unit
TEST_API_KEY = "test-api-key-1234567890"
KEY = "limits.audio_concurrent_transcriptions"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, bypass_api_limits: Any):
    """The audio router for single user 1, a limit of 1 transcription at a time, and a fresh lease store."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _user() -> User:
        """The single-user principal."""
        return User(id=1, username="single_user")

    async def _one_at_a_time(_user_id: int, key: str) -> object:
        """A limit of 1 for direct transcription; every other key unset."""
        return 1 if key == KEY else None

    store = MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies())
    monkeypatch.setattr(concurrency_quota, "user_quota", _one_at_a_time)
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    app = FastAPI()
    app.dependency_overrides[get_request_user] = _user
    app.include_router(audio_ep.router, prefix="/api/v1/audio")
    with bypass_api_limits(app), TestClient(app) as test_client:
        test_client.store = store
        yield test_client


def _post(client: TestClient):
    """Upload a small file for transcription."""
    return client.post(
        "/api/v1/audio/transcriptions",
        headers={"X-API-KEY": TEST_API_KEY},
        files={"file": ("sample.wav", io.BytesIO(b"RIFF" + b"\0" * 64), "audio/wav")},
        data={"model": "external:stub", "response_format": "json"},
    )


async def _heartbeat_loops() -> list[asyncio.Task]:
    """The request job-heartbeat loops still running on the app's event loop."""
    return [task for task in asyncio.all_tasks() if task.get_coro().__name__ == "_hb_loop"]


def test_refused_with_429_while_the_users_slot_is_held(client: TestClient) -> None:
    """With the user's one slot taken by another request, a transcription is refused with 429 and Retry-After."""
    held = RGRequest(
        entity="user:1",
        categories={"jobs": {"units": 1, "max_concurrent": 1, "ttl_sec": 120}},
        tags={"policy_id": KEY},
    )
    decision, _handle = asyncio.run(client.store.reserve(held))
    assert decision.allowed
    resp = _post(client)
    assert resp.status_code == 429, resp.text
    assert "Concurrent transcription limit reached (1)" in resp.text
    assert int(resp.headers["Retry-After"]) >= 1
    assert client.portal.call(_heartbeat_loops) == []


def test_a_failed_request_gives_its_slot_back(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that fails after admission (413 here) releases its slot: the next one fails on its own merits, not 429."""
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    monkeypatch.setattr(audio_tx, "_max_upload_bytes", lambda _limits: 8)
    first, second = _post(client), _post(client)
    assert (first.status_code, second.status_code) == (413, 413)
    assert second.json()["detail"] == "File size exceeds maximum of 0MB"
    assert "Concurrent transcription limit" not in second.text
    assert client.portal.call(_heartbeat_loops) == []


def test_a_policy_resolution_failure_leaves_no_heartbeat_task(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that fails before its slot is taken starts no job-heartbeat task, so none is left running."""
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    async def _broken(**_kwargs: Any) -> None:
        """Fail STT policy resolution."""
        raise RuntimeError("policy failed")

    monkeypatch.setattr(audio_tx, "resolve_effective_stt_policy", _broken)
    with pytest.raises(RuntimeError, match="policy failed"):
        _post(client)
    assert client.portal.call(_heartbeat_loops) == []
