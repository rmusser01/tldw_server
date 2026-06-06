import os
import time
from typing import Optional

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
def client_with_wf(tmp_path, monkeypatch, auth_headers):
    monkeypatch.setenv("TEST_MODE", "1")
    base = tmp_path / "user_databases"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base))

    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@example.com",
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
            email="t@example.com",
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


def _get_wait_timeout() -> float:
    value = os.getenv("TEST_WAIT_TIMEOUT")
    if not value:
        return 5.0
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("TEST_WAIT_TIMEOUT must be a valid float value") from exc
    if timeout <= 0:
        raise ValueError("TEST_WAIT_TIMEOUT must be positive")
    return timeout


@pytest.fixture()
def wait_timeout() -> float:
    return _get_wait_timeout()


def _wait_terminal(client: TestClient, run_id: str, timeout: Optional[float] = None):
    if timeout is None:
        timeout = _get_wait_timeout()
    deadline = time.time() + timeout
    last_data = None
    while time.time() < deadline:
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        response.raise_for_status()
        last_data = response.json()
        if last_data.get("status") in ("succeeded", "failed", "cancelled"):
            return last_data
        time.sleep(0.05)
    if last_data is None:
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        response.raise_for_status()
        last_data = response.json()
    data = last_data
    data["_timeout"] = True
    return data


def test_mcp_tool_allowlist_blocks(client_with_wf: TestClient, wait_timeout: float):
    client = client_with_wf
    definition = {
        "name": "mcp-block",
        "version": 1,
        "metadata": {"mcp": {"allowlist": ["media.search"]}},
        "steps": [
            {"id": "m1", "type": "mcp_tool", "config": {"tool_name": "echo", "arguments": {"message": "hi"}}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id, timeout=wait_timeout)
    assert not data.get("_timeout"), f"Run {run_id} did not finish within {wait_timeout:.2f}s."
    assert data["status"] == "failed"
    assert "mcp_tool_not_allowed" in (data.get("error") or data.get("status_reason") or "")


def test_mcp_tool_allowlist_allows(client_with_wf: TestClient, wait_timeout: float):
    client = client_with_wf
    definition = {
        "name": "mcp-allow",
        "version": 1,
        "metadata": {"mcp": {"allowlist": ["echo"]}},
        "steps": [
            {"id": "m1", "type": "mcp_tool", "config": {"tool_name": "echo", "arguments": {"message": "hi"}}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id, timeout=wait_timeout)
    assert not data.get("_timeout"), f"Run {run_id} did not finish within {wait_timeout:.2f}s."
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("result") == "hi"
