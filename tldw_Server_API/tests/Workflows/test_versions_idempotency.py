import time
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


@pytest.fixture()
def client_with_wf(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            tenant_id="default",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ, WORKFLOWS_RUNS_CONTROL],
        )

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            email="t@e.com",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ, WORKFLOWS_RUNS_CONTROL],
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client

    app.dependency_overrides.clear()


def test_versioning_and_soft_delete(client_with_wf: TestClient):
    client = client_with_wf

    base_def = {
        "name": "ver-demo",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "A"}}],
    }
    wid = client.post("/api/v1/workflows", json=base_def).json()["id"]

    v2 = {**base_def, "version": 2}
    r = client.post(f"/api/v1/workflows/{wid}/versions", json=v2)
    assert r.status_code == 201

    lst = client.get("/api/v1/workflows").json()
    names = [d["name"] for d in lst]
    assert "ver-demo" in names

    # Soft delete
    ddel = client.delete(f"/api/v1/workflows/{wid}")
    assert ddel.status_code == 200
    lst2 = client.get("/api/v1/workflows").json()
    # Deleted definition should be excluded
    assert not any(d["id"] == wid and d["is_active"] for d in lst2)


def test_idempotent_runs(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "idem",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "X"}}],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]

    body = {"inputs": {}, "idempotency_key": "key-1"}
    r1 = client.post(f"/api/v1/workflows/{wid}/run", json=body).json()
    r2 = client.post(f"/api/v1/workflows/{wid}/run", json=body).json()
    assert r1["run_id"] == r2["run_id"]


def test_retry_from_failed_step(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "retry-fail-then-continue",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "bad", "force_error": True}},
            {"id": "s2", "type": "prompt", "config": {"template": "OK"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Wait until failed
    for _ in range(200):
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] in ("failed", "succeeded"):
            break
        time.sleep(0.02)
    assert data["status"] == "failed"

    # Retry should continue after s1 and run s2
    rr = client.post(f"/api/v1/workflows/runs/{run_id}/retry")
    assert rr.status_code == 200

    # Poll until completion
    saw_retry_transition = False
    for _ in range(200):
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert response.status_code == 200, response.text
        data2 = response.json()
        if data2["status"] == "running":
            saw_retry_transition = True
        if data2["status"] == "succeeded":
            break
        if saw_retry_transition and data2["status"] == "failed":
            break
        time.sleep(0.02)
    assert data2["status"] == "succeeded"
    assert (data2.get("outputs") or {}).get("text") == "OK"
