import base64
import json
import time
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


@pytest.fixture()
def client(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
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

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as c:
        yield c

    app.dependency_overrides.clear()


def _create_def(client: TestClient) -> int:
    definition = {
        "name": "cursor-demo",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "log", "config": {"message": "ok"}},
        ],
    }
    r = client.post("/api/v1/workflows", json=definition)
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _start_run(client: TestClient, wid: int) -> str:
    rr = client.post(f"/api/v1/workflows/{wid}/run?mode=async", json={"inputs": {}})
    assert rr.status_code == 200, rr.text
    return rr.json()["run_id"]


def _wait_status(client: TestClient, run_id: str, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert s.status_code == 200
        if s.json().get("status") in ("succeeded", "failed", "cancelled"):
            return s.json()
        time.sleep(0.05)
    return s.json()


@pytest.mark.integration
def test_runs_cursor_pagination_flow(client: TestClient):
    wid = _create_def(client)
    # Start a few runs
    ids = []
    for _ in range(3):
        rid = _start_run(client, wid)
        ids.append(rid)
    # Ensure they reach terminal (order_by created_at desc)
    for rid in ids:
        _wait_status(client, rid)

    # First page (offset-style)
    r1 = client.get("/api/v1/workflows/runs", params={"limit": 2, "order_by": "created_at", "order": "desc"})
    assert r1.status_code == 200
    p1 = r1.json()
    assert isinstance(p1.get("runs"), list)
    assert len(p1["runs"]) <= 2
    assert p1["pagination"]["mode"] == "offset"
    assert p1["pagination"]["limit"] == 2
    assert p1["pagination"]["offset"] == 0
    assert p1["pagination"]["has_more"] is True
    assert p1["pagination"]["next_offset"] == 2
    assert p1["has_more"] is True
    assert p1["next_offset"] == 2
    next_cursor = p1.get("next_cursor")
    assert isinstance(next_cursor, str) and next_cursor, "expected next_cursor token"

    # Cursor token structure sanity: base64url JSON with required fields
    pad = "=" * (-len(next_cursor) % 4)
    raw = base64.urlsafe_b64decode((next_cursor + pad).encode("utf-8")).decode("utf-8")
    tok = json.loads(raw)
    assert set(["order_by", "order_desc", "last_ts", "last_id"]).issubset(set(tok.keys()))

    # Next page using cursor (offset is ignored)
    r2 = client.get("/api/v1/workflows/runs", params={"limit": 2, "cursor": next_cursor, "offset": 999})
    assert r2.status_code == 200
    p2 = r2.json()
    assert isinstance(p2.get("runs"), list)
    # No overlap with p1 last item
    if p1["runs"]:
        last_id = p1["runs"][ -1 ]["run_id"]
        assert all(it["run_id"] != last_id for it in p2["runs"])
