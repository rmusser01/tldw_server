"""GET /api/v1/{discord,slack}/jobs/{job_id} is scoped to the web user (TASK-13364).

It used to answer anyone, for any job of that integration. Now: the job's owner, or an
active member of an org that installed the job's guild/workspace -- unless that
tenant's policy limits status to the job owner. Everything else is 404.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import discord as discord_endpoint
from tldw_Server_API.app.api.v1.endpoints import slack as slack_endpoint

_JOBS = {
    11: {"id": 11, "domain": "discord", "status": "queued", "owner_user_id": "7", "payload": {"guild_id": "g-other"}},
    12: {"id": 12, "domain": "discord", "status": "queued", "owner_user_id": "d-123", "payload": {"guild_id": "g-1"}},
    13: {"id": 13, "domain": "discord", "status": "queued", "owner_user_id": "d-123", "payload": {"guild_id": "g-2"}},
    14: {"id": 14, "domain": "slack", "status": "queued", "owner_user_id": "7", "payload": {"team_id": "T1"}},
    21: {"id": 21, "domain": "slack", "status": "queued", "owner_user_id": "s-9", "payload": {"team_id": "T1"}},
}


class _JobManager:
    def get_job(self, job_id: int) -> dict[str, Any] | None:
        return _JOBS.get(int(job_id))


class _InstallationsRepo:
    async def list_installations(self, *, org_id: int, provider: str | None = None, **_: Any) -> list[dict[str, Any]]:
        installed = {(5, "discord"): ["g-1"], (5, "slack"): ["T1"]}
        return [{"external_id": ext} for ext in installed.get((org_id, provider), [])]


@pytest.fixture()
def client_for(monkeypatch: pytest.MonkeyPatch):
    async def _memberships(_user_id: int) -> list[dict[str, Any]]:
        return [{"org_id": 5, "status": "active"}]

    async def _repo() -> _InstallationsRepo:
        return _InstallationsRepo()

    for module in (discord_endpoint, slack_endpoint):
        monkeypatch.setattr(module, "_get_job_manager", lambda: _JobManager())
        monkeypatch.setattr(module, "list_org_memberships_for_user", _memberships)
        monkeypatch.setattr(module, "_get_workspace_provider_installations_repo", _repo)
        monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(AUTH_MODE="multi_user"))
    discord_endpoint._reset_discord_state_for_tests()
    slack_endpoint._reset_slack_state_for_tests()

    def _make(user_id: int | None) -> TestClient:
        app = FastAPI()
        app.include_router(discord_endpoint.router, prefix="/api/v1")
        app.include_router(slack_endpoint.router, prefix="/api/v1")
        if user_id is not None:
            app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=user_id)
        return TestClient(app)

    return _make


def test_unauthenticated_caller_is_rejected(client_for) -> None:
    assert client_for(None).get("/api/v1/discord/jobs/12").status_code == 401


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/v1/discord/jobs/11", 200),  # owner, even outside their orgs' installations
        ("/api/v1/discord/jobs/12", 200),  # guild g-1 is installed by the user's org 5
        ("/api/v1/discord/jobs/13", 404),  # guild g-2 belongs to nobody the user is in
        ("/api/v1/discord/jobs/14", 404),  # a Slack job via the Discord route
        ("/api/v1/discord/jobs/99", 404),  # missing
        ("/api/v1/slack/jobs/21", 200),  # workspace T1 installed by org 5
    ],
)
def test_owner_or_tenant_member_only(client_for, path: str, expected: int) -> None:
    response = client_for(7).get(path)
    assert response.status_code == expected, response.text
    if expected == 200:
        assert response.json()["job"]["id"] == int(path.rsplit("/", 1)[1])


def test_owner_only_policy_hides_other_members_jobs(client_for) -> None:
    discord_endpoint._set_discord_policy("g-1", {"status_scope": "guild_and_user"})
    client = client_for(7)
    assert client.get("/api/v1/discord/jobs/12").status_code == 404  # tenant member, not owner
    assert client.get("/api/v1/discord/jobs/11").status_code == 200  # still the owner


def test_non_member_sees_nothing_but_their_own(client_for, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_memberships(_user_id: int) -> list[dict[str, Any]]:
        return [{"org_id": 5, "status": "invited"}]

    monkeypatch.setattr(discord_endpoint, "list_org_memberships_for_user", _no_memberships)
    client = client_for(7)
    assert client.get("/api/v1/discord/jobs/12").status_code == 404
    assert client.get("/api/v1/discord/jobs/11").status_code == 200


def test_single_user_mode_sees_every_job_of_the_integration(client_for, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discord_endpoint, "get_settings", lambda: SimpleNamespace(AUTH_MODE="single_user"))
    client = client_for(1)
    assert client.get("/api/v1/discord/jobs/13").status_code == 200
    assert client.get("/api/v1/discord/jobs/14").status_code == 404  # still domain-checked
