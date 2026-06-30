from __future__ import annotations

import asyncio
import time
import types
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
from tldw_Server_API.app.core.exceptions import AdapterError


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_wf(tmp_path, auth_headers, monkeypatch):
    monkeypatch.setenv("WORKFLOWS_SQLITE_POOL_SIZE", "4")
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

    try:
        with TestClient(app, headers=auth_headers) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        db.close()


def _wait_terminal(client: TestClient, run_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        d = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if d["status"] in ("succeeded", "failed", "cancelled"):
            return d
        time.sleep(0.02)
    return client.get(f"/api/v1/workflows/runs/{run_id}").json()


def test_cancel_during_delay_step(client_with_wf: TestClient):
    client = client_with_wf
    # Single delay step long enough to issue cancel in-between
    definition = {
        "name": "cancel-delay",
        "version": 1,
        "steps": [
            {"id": "d1", "type": "delay", "config": {"milliseconds": 1500}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    # Cancel very soon after start
    r = client.post(f"/api/v1/workflows/runs/{run_id}/cancel")
    assert r.status_code == 200
    d = _wait_terminal(client, run_id)
    assert d["status"] == "cancelled"
    # Ensure a run_cancelled event exists
    ev = client.get(f"/api/v1/workflows/runs/{run_id}/events").json()
    assert any(e.get("event_type") == "run_cancelled" for e in ev)


def test_cancel_records_ack_event(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "cancel-ack",
        "version": 1,
        "steps": [
            {"id": "d1", "type": "delay", "config": {"milliseconds": 1200}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    r = client.post(f"/api/v1/workflows/runs/{run_id}/cancel")
    assert r.status_code == 200
    terminal = _wait_terminal(client, run_id)
    assert terminal["status"] == "cancelled"

    events = client.get(f"/api/v1/workflows/runs/{run_id}/events").json()
    assert any(e.get("event_type") == "cancel_acknowledged" for e in events)


def test_retry_backoff_persists_attempts(client_with_wf: TestClient):
    client = client_with_wf
    # Configure a quick timeout and one retry; start async to avoid blocking
    definition = {
        "name": "retry-persist",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "retry": 1, "timeout_seconds": 0.02, "config": {"template": "Y", "simulate_delay_ms": 100}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run?mode=async", json={"inputs": {}}).json()["run_id"]

    # Poll DB for attempts (>=2: initial + retry) without requiring terminal state
    db: WorkflowsDatabase = app.dependency_overrides[wf_mod._get_db]()
    deadline = time.time() + 3.0
    attempt = 0
    while time.time() < deadline:
        row = db._conn.cursor().execute(
            "SELECT attempt FROM workflow_step_runs WHERE run_id = ? ORDER BY started_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row:
            attempt = int(row[0])
            if attempt >= 2:
                break
        time.sleep(0.02)
    assert attempt >= 2


def test_backoff_cap_env_applied(monkeypatch, client_with_wf: TestClient):
    client = client_with_wf
    # Cap the backoff to 1s and intercept asyncio.sleep to capture durations
    monkeypatch.setenv("WORKFLOWS_BACKOFF_CAP_SECONDS", "1")

    sleep_calls = []
    orig_sleep = asyncio.sleep

    async def _fake_sleep(dur: float):
        sleep_calls.append(float(dur))
        await orig_sleep(0)
        return None

    monkeypatch.setattr("asyncio.sleep", _fake_sleep)

    # Force a timeout to trigger retry backoff in the engine
    definition = {
        "name": "retry-cap",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "retry": 1, "timeout_seconds": 0.01, "config": {"template": "Y", "simulate_delay_ms": 100}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run?mode=async", json={"inputs": {}}).json()["run_id"]

    # Wait briefly for one retry cycle to occur
    import time
    t0 = time.time()
    while time.time() - t0 < 1.0 and len(sleep_calls) == 0:
        time.sleep(0.02)

    assert len(sleep_calls) >= 1
    # The first backoff should be <= cap(1) + jitter(~<=0.75). Assert sanity bound 2.0s
    assert sleep_calls[0] <= 2.0


def test_non_retriable_error_skips_retry_attempts(monkeypatch, client_with_wf: TestClient):
    client = client_with_wf

    async def _blocked_adapter(_config, _context):
        raise AdapterError("acp_governance_blocked")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.engine.get_adapter",
        lambda _step_type: _blocked_adapter,
    )

    definition = {
        "name": "retry-classifier-governance",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "retry": 3, "config": {"template": "Y"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run?mode=async", json={"inputs": {}}).json()["run_id"]

    terminal = _wait_terminal(client, run_id, timeout_s=5.0)
    assert terminal["status"] == "failed"
    assert terminal.get("status_reason") == "acp_governance_blocked"

    db: WorkflowsDatabase = app.dependency_overrides[wf_mod._get_db]()
    row = db._conn.cursor().execute(
        "SELECT MAX(attempt) FROM workflow_step_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert row is not None
    assert int(row[0] or 0) == 1
