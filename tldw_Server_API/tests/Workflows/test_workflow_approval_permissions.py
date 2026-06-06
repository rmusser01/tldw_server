"""Integration tests for workflow approval permissions."""

import itertools
import time
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import (
    WORKFLOWS_RUNS_CONTROL,
    WORKFLOWS_RUNS_READ,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration

_NAME_COUNTER = itertools.count(1)

@pytest.fixture()
def client_with_user_switch(
    tmp_path: Path,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, dict[str, User]], None, None]:
    """Provide a TestClient and mutable user state for auth switching."""
    base = tmp_path / "user_databases"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base))
    monkeypatch.setenv("TEST_MODE", "1")
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    permissions = [WORKFLOWS_RUNS_READ, WORKFLOWS_RUNS_CONTROL]
    state = {
        "user": User(
            id=1,
            username="owner",
            email="owner@example.com",
            is_active=True,
            is_admin=False,
            tenant_id="default",
            roles=[],
            permissions=permissions,
        ),
    }

    def current_user() -> User:
        user = state["user"]
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_superuser=user.is_superuser,
            is_admin=user.is_admin,
            tenant_id="default",
            roles=getattr(user, "roles", None) or (["admin"] if user.is_admin else []),
            permissions=permissions,
        )

    async def override_user():
        return current_user()

    async def override_principal():
        user = current_user()
        return AuthPrincipal(
            kind="user",
            user_id=user.id,
            username=user.username,
            email=user.email,
            roles=getattr(user, "roles", None) or (["admin"] if getattr(user, "is_admin", False) else []),
            permissions=permissions,
        )

    def override_db():
        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client, state

    app.dependency_overrides.clear()


def _wait_for_status(client: TestClient, run_id: str, statuses: set[str], timeout_s: float = 5.0) -> dict:
    """Poll run status until it matches one of the expected statuses."""
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.05)
    return last


def _wait_for_terminal(client: TestClient, run_id: str, timeout_s: float = 5.0) -> dict:
    """Wait until a run reaches a terminal status."""
    return _wait_for_status(client, run_id, {"succeeded", "failed", "cancelled"}, timeout_s)


def _create_waiting_run(client: TestClient, definition: dict) -> str:
    """Create a run that blocks on a human/approval step."""
    payload = dict(definition)
    payload["name"] = f"{definition.get('name', 'workflow')}-{next(_NAME_COUNTER)}"
    r = client.post("/api/v1/workflows", json=payload)
    assert r.status_code == 201, r.text
    wid = r.json()["id"]
    r = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    data = _wait_for_status(client, run_id, {"waiting_human", "waiting_approval"})
    assert data.get("status") in {"waiting_human", "waiting_approval"}
    return run_id


def test_approve_requires_assigned_user(
    client_with_user_switch: tuple[TestClient, dict[str, User]],
) -> None:
    """Verify assigned users can approve approval steps."""
    client, state = client_with_user_switch
    definition = {
        "name": "approval-perms",
        "version": 1,
        "steps": [
            {"id": "l1", "type": "log", "config": {"message": "Start", "level": "info"}},
            {
                "id": "w1",
                "type": "wait_for_approval",
                "config": {"instructions": "Approve to continue", "assigned_to_user_id": 2},
            },
            {"id": "l2", "type": "log", "config": {"message": "Approved", "level": "info"}},
        ],
    }
    run_id = _create_waiting_run(client, definition)

    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/approve", json={"comment": "nope"})
    assert r.status_code == 404, r.text

    state["user"] = User(id=2, username="assignee", email="assignee@example.com", is_active=True, is_admin=False)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/approve", json={"comment": "ok"})
    assert r.status_code == 200, r.text
    state["user"] = User(id=1, username="owner", email="owner@example.com", is_active=True, is_admin=False)
    data = _wait_for_terminal(client, run_id)
    assert data["status"] == "succeeded"


def test_reject_allows_admin_override(
    client_with_user_switch: tuple[TestClient, dict[str, User]],
) -> None:
    """Verify admins can reject approval steps."""
    client, state = client_with_user_switch
    definition = {
        "name": "approval-reject-perms",
        "version": 1,
        "steps": [
            {"id": "l1", "type": "log", "config": {"message": "Start", "level": "info"}},
            {
                "id": "w1",
                "type": "wait_for_approval",
                "config": {"instructions": "Approve to continue", "assigned_to_user_id": 2},
            },
            {"id": "l2", "type": "log", "config": {"message": "After reject", "level": "info"}},
        ],
    }

    run_id = _create_waiting_run(client, definition)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/reject", json={"comment": "owner no"})
    assert r.status_code == 404, r.text

    state["user"] = User(id=2, username="assignee", email="assignee@example.com", is_active=True, is_admin=False)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/reject", json={"comment": "assignee ok"})
    assert r.status_code == 200, r.text
    state["user"] = User(id=1, username="owner", email="owner@example.com", is_active=True, is_admin=False)
    data = _wait_for_terminal(client, run_id)
    assert data["status"] == "failed"

    run_id = _create_waiting_run(client, definition)
    state["user"] = User(id=3, username="admin", email="admin@example.com", is_active=True, is_admin=True)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/reject", json={"comment": "admin ok"})
    assert r.status_code == 200, r.text
    data = _wait_for_terminal(client, run_id)
    assert data["status"] == "failed"
