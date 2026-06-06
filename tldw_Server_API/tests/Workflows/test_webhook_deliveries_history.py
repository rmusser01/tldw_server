import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.DB_Manager import create_workflows_database, get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


@pytest.fixture()
def client_with_wf_db(tmp_path, auth_headers):
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
        yield client, db

    app.dependency_overrides.clear()


def test_get_webhook_deliveries_history(client_with_wf_db):


    client, db = client_with_wf_db
    # Create and run a trivial workflow, then append synthetic webhook events
    definition = {"name": "x", "version": 1, "steps": [{"id": "l1", "type": "log", "config": {"message": "hi"}}]}
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Append directly via the same DB used by the app override
    db.append_event("default", run_id, "webhook_delivery", {"host": "example.com", "status": "delivered", "code": 200})
    db.append_event("default", run_id, "webhook_delivery", {"host": "example.com", "status": "failed", "code": 500})

    r = client.get(f"/api/v1/workflows/runs/{run_id}/webhooks/deliveries")
    assert r.status_code == 200
    deliveries = r.json().get("deliveries") or []
    assert len(deliveries) >= 2
    hosts = {d.get("host") for d in deliveries}
    assert "example.com" in hosts
