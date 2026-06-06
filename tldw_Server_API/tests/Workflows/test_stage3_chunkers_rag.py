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
        yield client

    app.dependency_overrides.clear()


def test_chunker_options_endpoint(client_with_wf: TestClient):
    client = client_with_wf
    r = client.get("/api/v1/workflows/options/chunkers")
    assert r.status_code == 200
    data = r.json()
    assert data.get("name") == "core_chunking"
    assert isinstance(data.get("methods"), list) and "words" in data["methods"]
    assert isinstance(data.get("defaults"), dict)
    assert isinstance(data.get("parameter_schema"), dict)


def test_rag_search_with_citations_returns_citations(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "rag-citations",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "rag_search",
                "config": {
                    "query": "test",
                    "top_k": 1,
                    "enable_citations": True,
                    "enable_reranking": False
                },
            },
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    # Poll until completion
    import time
    for _ in range(100):
        resp = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    # Citations may be empty but key should exist when enabled
    if out.get("documents"):
        assert "citations" in out or out.get("generated_answer") is not None


def test_rag_search_rejects_unknown_fields(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "rag-invalid-extra",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "rag_search",
                "config": {
                    "query": "test",
                    "unknown_field": True,
                },
            },
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_rag_search_rejects_invalid_bounds(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "rag-invalid-bounds",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "rag_search",
                "config": {
                    "query": "test",
                    "top_k": 1000,
                },
            },
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_media_ingest_rejects_unknown_chunker(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "ingest-invalid-chunker",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "media_ingest",
                "config": {
                    "sources": [{"uri": "file:///tmp/does-not-matter.txt"}],
                    "chunking": {"strategy": "not_a_method"},
                },
            },
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422
