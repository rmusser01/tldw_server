"""A malformed pagination cursor is a 400 "Invalid cursor", never a silent restart.

Both workflows list routes used to swallow a bad cursor and serve page 1, so a client
with a corrupted cursor looped forever over the same page without an error.
Contract: Docs/API-related/Pagination_Cursors.md.
"""

import base64

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.integration


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


MALFORMED = [
    "not!a!cursor",               # out of the base64url alphabet
    _b64(b"not json"),            # decodes, but not JSON
    _b64(b"[1, 2]"),              # JSON, but not an object
    _b64(b'{"unrelated": 1}'),    # object without the position fields
]


@pytest.fixture()
def client(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf_bad_cursor.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            tenant_id="default",
            roles=["admin"],
            permissions=["*"],
        )

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
        )

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = lambda: db
    try:
        with TestClient(app, headers=auth_headers) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        db.close()


@pytest.mark.parametrize("cursor", MALFORMED)
def test_runs_list_rejects_malformed_cursor(client: TestClient, cursor: str) -> None:
    response = client.get("/api/v1/workflows/runs", params={"cursor": cursor})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Invalid cursor"


def test_events_list_rejects_malformed_cursor(client: TestClient) -> None:
    definition = {
        "name": "bad-cursor",
        "version": 1,
        "steps": [{"id": "l1", "type": "log", "config": {"message": "ok"}}],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run = client.post(f"/api/v1/workflows/{wid}/run", params={"mode": "sync"}, json={"inputs": {}})
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]

    for cursor in MALFORMED:
        response = client.get(f"/api/v1/workflows/runs/{run_id}/events", params={"cursor": cursor})
        assert response.status_code == 400, (cursor, response.text)
        assert response.json()["detail"] == "Invalid cursor"
