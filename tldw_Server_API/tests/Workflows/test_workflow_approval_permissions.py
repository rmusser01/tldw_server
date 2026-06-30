"""Integration tests for workflow approval permissions."""

import itertools
import time
from pathlib import Path
from typing import Generator
from uuid import uuid4

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
) -> Generator[tuple[TestClient, dict[str, User], WorkflowsDatabase], None, None]:
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
        yield client, state, db

    app.dependency_overrides.clear()


def _wait_for_status(client: TestClient, run_id: str, statuses: set[str], timeout_s: float = 15.0) -> dict:
    """Poll run status until it matches one of the expected statuses."""
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        if response.status_code != 200:
            pytest.fail(f"workflow run {run_id} status poll returned {response.status_code}: {response.text}")
        last = response.json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.1)
    pytest.fail(
        f"workflow run {run_id} did not reach {sorted(statuses)} "
        f"within {timeout_s}s; last={last}"
    )


def _wait_for_terminal(client: TestClient, run_id: str, timeout_s: float = 15.0) -> dict:
    """Wait until a run reaches a terminal status."""
    return _wait_for_status(client, run_id, {"succeeded", "failed", "cancelled"}, timeout_s)


def _create_waiting_run(client: TestClient, definition: dict, db: WorkflowsDatabase) -> str:
    """Create a run that blocks on a human/approval step."""
    payload = dict(definition)
    payload["name"] = f"{definition.get('name', 'workflow')}-{next(_NAME_COUNTER)}"
    r = client.post("/api/v1/workflows", json=payload)
    if r.status_code != 201:
        pytest.fail(f"workflow create returned {r.status_code}: {r.text}")
    wid = r.json()["id"]

    wait_step = next(
        (
            step
            for step in payload.get("steps", [])
            if step.get("type") in {"wait_for_human", "wait_for_approval"}
        ),
        None,
    )
    if wait_step is None:
        pytest.fail("test definition does not include a human wait step")
    wait_config = wait_step.get("config") or {}
    assigned_to = str(wait_config.get("assigned_to_user_id") or "")
    if not assigned_to:
        pytest.fail("test wait step does not include assigned_to_user_id")

    run_id = str(uuid4())
    wait_status = "waiting_human" if wait_step.get("type") == "wait_for_human" else "waiting_approval"
    wait_payload = {"__status__": wait_status, "assigned_to": assigned_to}
    db.create_run(
        run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=wid,
        definition_version=payload.get("version", 1),
        definition_snapshot=payload,
    )
    step_run_id = f"{run_id}:{wait_step.get('id', 'w1')}:{int(time.time() * 1000)}"
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id=str(wait_step.get("id") or "w1"),
        name=str(wait_step.get("name") or wait_step.get("id") or "w1"),
        step_type=str(wait_step.get("type") or "wait_for_approval"),
        status=wait_status,
        inputs={"config": wait_config},
        assigned_to=assigned_to,
    )
    db.complete_step_run(step_run_id=step_run_id, status=wait_status, outputs=wait_payload)
    db.update_run_status(
        run_id,
        status=wait_status,
        status_reason="awaiting_review",
        outputs=wait_payload,
    )
    return run_id


def test_approve_requires_assigned_user(
    client_with_user_switch: tuple[TestClient, dict[str, User], WorkflowsDatabase],
) -> None:
    """Verify assigned users can approve approval steps."""
    client, state, db = client_with_user_switch
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
    run_id = _create_waiting_run(client, definition, db)

    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/approve", json={"comment": "nope"})
    assert r.status_code == 404, r.text

    state["user"] = User(id=2, username="assignee", email="assignee@example.com", is_active=True, is_admin=False)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/approve", json={"comment": "ok"})
    assert r.status_code == 200, r.text
    state["user"] = User(id=1, username="owner", email="owner@example.com", is_active=True, is_admin=False)
    data = _wait_for_terminal(client, run_id)
    assert data["status"] == "succeeded"


def test_reject_allows_admin_override(
    client_with_user_switch: tuple[TestClient, dict[str, User], WorkflowsDatabase],
) -> None:
    """Verify admins can reject approval steps."""
    client, state, db = client_with_user_switch
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

    run_id = _create_waiting_run(client, definition, db)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/reject", json={"comment": "owner no"})
    assert r.status_code == 404, r.text

    state["user"] = User(id=2, username="assignee", email="assignee@example.com", is_active=True, is_admin=False)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/reject", json={"comment": "assignee ok"})
    assert r.status_code == 200, r.text
    state["user"] = User(id=1, username="owner", email="owner@example.com", is_active=True, is_admin=False)
    data = _wait_for_terminal(client, run_id)
    assert data["status"] == "failed"

    run_id = _create_waiting_run(client, definition, db)
    state["user"] = User(id=3, username="admin", email="admin@example.com", is_active=True, is_admin=True)
    r = client.post(f"/api/v1/workflows/runs/{run_id}/steps/w1/reject", json={"comment": "admin ok"})
    assert r.status_code == 200, r.text
    data = _wait_for_terminal(client, run_id)
    assert data["status"] == "failed"
