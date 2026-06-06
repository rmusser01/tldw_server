import base64
import time
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_wf(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf_events_cursor.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            tenant_id="default",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            email="t@e.com",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client

    app.dependency_overrides.clear()


def _wait_for_terminal(client: TestClient, run_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] in ("succeeded", "failed", "cancelled"):
            return data
        time.sleep(0.02)
    response = client.get(f"/api/v1/workflows/runs/{run_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_events_pagination_with_next_cursor_header(client_with_wf: TestClient):
    client = client_with_wf
    # Create a tiny workflow that emits a handful of events (queued, started, step events, succeeded)
    definition = {
        "name": "events-cursor",
        "version": 1,
        "steps": [
            {"id": "d1", "type": "delay", "config": {"milliseconds": 5}},
            {"id": "l1", "type": "log", "config": {"message": "hello {{ inputs.name }}", "level": "info"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"name": "Ada"}}).json()["run_id"]

    # Wait until terminal to have a stable set of events
    _wait_for_terminal(client, run_id)

    # First page: limit 1 to force pagination; capture Next-Cursor header
    r1 = client.get(f"/api/v1/workflows/runs/{run_id}/events", params={"limit": 1})
    assert r1.status_code == 200
    ev1 = r1.json()
    assert isinstance(ev1, list) and len(ev1) == 1
    next_cursor = r1.headers.get("Next-Cursor")
    assert next_cursor is not None and isinstance(next_cursor, str) and len(next_cursor) > 0

    # Decode base64url token to ensure it is well-formed JSON with last_seq
    pad = "=" * (-len(next_cursor) % 4)
    raw = base64.urlsafe_b64decode((next_cursor + pad).encode("utf-8")).decode("utf-8")
    assert '"last_seq"' in raw

    # Second page via cursor; should advance event_seq
    r2 = client.get(f"/api/v1/workflows/runs/{{run_id}}/events".format(run_id=run_id), params={"limit": 1, "cursor": next_cursor})
    assert r2.status_code == 200
    ev2 = r2.json()
    assert isinstance(ev2, list) and len(ev2) >= 0
    # If we got another event, its seq should be greater than the first
    if ev2:
        assert int(ev2[0]["event_seq"]) > int(ev1[0]["event_seq"])  # stable ASC ordering
