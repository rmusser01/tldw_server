import time
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


pytestmark = pytest.mark.integration


def _wait_for_run_status(
    client: TestClient,
    run_id: str,
    terminal_statuses: set[str],
    timeout_s: float = 30.0,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_data: dict | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        if response.status_code != 200:
            pytest.fail(
                f"workflow run {run_id} status poll returned "
                f"{response.status_code}: {response.text}"
            )
        last_data = response.json()
        status = last_data.get("status")
        if status in terminal_statuses:
            return last_data
        time.sleep(0.1)
    pytest.fail(
        f"workflow run {run_id} did not reach {sorted(terminal_statuses)} "
        f"within {timeout_s}s; last={last_data}"
    )


@pytest.fixture()
def client_with_wf(tmp_path, monkeypatch, auth_headers):
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    async def override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test",
            token_type="test",
            jti=None,
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
        return principal

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client

    app.dependency_overrides.clear()


def test_media_ingest_local_text_chunking(client_with_wf: TestClient, tmp_path):
    client = client_with_wf
    # Create a small local text file
    sample = "# Title\n\nThis is a test document. It has multiple sentences. Another line here."
    fpath = tmp_path / "doc.txt"
    fpath.write_text(sample, encoding="utf-8")

    definition = {
        "name": "ingest-local",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "media_ingest",
                "config": {
                    "sources": [{"uri": f"file://{fpath}"}],
                    "extraction": {"extract_text": True},
                    "chunking": {"strategy": "sentences", "max_tokens": 50, "overlap": 0},
                },
            }
        ],
    }
    create_response = client.post("/api/v1/workflows", json=definition)
    if create_response.status_code != 201:
        pytest.fail(f"workflow create returned {create_response.status_code}: {create_response.text}")
    wid = create_response.json()["id"]
    run_response = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}})
    if run_response.status_code != 200:
        pytest.fail(f"workflow run returned {run_response.status_code}: {run_response.text}")
    run_id = run_response.json()["run_id"]

    data = _wait_for_run_status(client, run_id, {"succeeded", "failed"})
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "text" in out and isinstance(out["text"], str)
    assert "chunks" in out and isinstance(out["chunks"], list) and len(out["chunks"]) > 0
    # basic shape
    first = out["chunks"][0]
    assert "chunker_name" in first and "metadata" in first
