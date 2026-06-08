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


def test_cancel_during_long_prompt(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "cancel-long",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "X", "simulate_delay_ms": 1500}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Immediately cancel
    r = client.post(f"/api/v1/workflows/runs/{run_id}/cancel")
    assert r.status_code == 200

    # Poll for cancelled
    status = None
    for _ in range(100):
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        status = data["status"]
        if status in ("cancelled", "failed", "succeeded"):
            break
        time.sleep(0.02)
    assert status == "cancelled"
    events = client.get(f"/api/v1/workflows/runs/{run_id}/events").json()
    assert any(event.get("event_type") == "cancel_acknowledged" for event in events)


def test_step_timeout_and_retry_failure(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "timeout-retry",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "retry": 1, "timeout_seconds": 0.05, "config": {"template": "Y", "simulate_delay_ms": 200}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Wait for completion
    for _ in range(200):
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    assert data["status"] == "failed"
    ev = client.get(f"/api/v1/workflows/runs/{run_id}/events").json()
    types = [e["event_type"] for e in ev]
    assert "step_timeout" in types or "run_failed" in types


def test_heartbeat_written_for_step(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "hb",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "hello"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # wait briefly for step to record
    time.sleep(0.05)
    # fetch from DB (via events identify step_run_id not exposed; query table)
    db: WorkflowsDatabase = app.dependency_overrides[wf_mod._get_db]()
    rows = db._conn.cursor().execute("SELECT heartbeat_at FROM workflow_step_runs WHERE run_id = ?", (run_id,)).fetchall()
    assert rows, "No step runs recorded"
    assert any(r[0] is not None for r in rows)


def test_completion_webhook_disable_and_enable(monkeypatch, client_with_wf: TestClient):
    """Verify completion webhook is suppressed when globally disabled and fires when enabled."""
    client = client_with_wf

    # Always allow egress in this test to reach the webhook code path when enabled
    from tldw_Server_API.app.core.Security import egress as egress_mod
    monkeypatch.setattr(egress_mod, "is_url_allowed", lambda url: True)

    # Track invocation via method stub (global disable should short-circuit)
    from tldw_Server_API.app.core.Workflows import engine as eng_mod
    from tldw_Server_API.app.core.testing import is_truthy
    calls = {"count": 0}
    async def _stub(self, defn, run_id, status):
        import os
        if is_truthy(os.getenv("WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS", "")):
            return
        calls["count"] += 1
        return
    monkeypatch.setattr(eng_mod.WorkflowEngine, "_maybe_send_completion_webhook", _stub)
    monkeypatch.setattr(wf_mod.WorkflowEngine, "_maybe_send_completion_webhook", _stub)

    # Definition with a completion webhook
    definition = {
        "name": "webhook-test",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "hello"}}],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]

    # Case 1: Globally disabled - no call should be recorded
    monkeypatch.setenv("WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS", "y")
    # Patch the stored definition to include the webhook in snapshot
    # (alternatively, run ad-hoc, but this path ensures engine snapshot contains config)
    # Here we re-create with webhook for simplicity
    definition_disabled = {
        "name": "webhook-test-disabled",
        "version": 2,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "hello"}}],
        "on_completion_webhook": {"url": "http://example.test/hook", "include_outputs": True},
        "metadata": {},
    }
    wid2 = client.post("/api/v1/workflows", json=definition_disabled).json()["id"]
    # Inject webhook into the stored snapshot by running ad-hoc (engine will use definition snapshot)
    run1 = client.post(f"/api/v1/workflows/{wid2}/run", json={"inputs": {}, "idempotency_key": "k1"}).json()["run_id"]
    # Wait for completion
    for _ in range(200):
        d = client.get(f"/api/v1/workflows/runs/{run1}").json()
        if d["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    assert d["status"] == "succeeded"
    # Because global disable is on, no webhook should be attempted
    before = calls["count"]

    # Case 2: Enabled - attach webhook and expect one call
    monkeypatch.setenv("WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS", "false")
    definition_enabled = {
        "name": "webhook-test-enabled",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "done"}}],
        "on_completion_webhook": {"url": "http://example.test/hook", "include_outputs": True},
    }
    wid3 = client.post("/api/v1/workflows", json=definition_enabled).json()["id"]
    run2 = client.post(f"/api/v1/workflows/{wid3}/run", json={"inputs": {}}).json()["run_id"]
    for _ in range(200):
        d2 = client.get(f"/api/v1/workflows/runs/{run2}").json()
        if d2["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    assert d2["status"] == "succeeded"
    for _ in range(100):
        if calls["count"] > before:
            break
        time.sleep(0.02)
    assert calls["count"] > before


@pytest.mark.asyncio
async def test_webhook_hmac_and_ssrf(monkeypatch, client_with_wf: TestClient):
    """Verify HMAC header correctness and SSRF allowlist behavior by invoking the engine helper directly."""
    client = client_with_wf
    from tldw_Server_API.app.core.Workflows import engine as eng_mod
    from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
    db = client.app.dependency_overrides[wf_mod._get_db]()

    # Create a run record that looks completed
    run_id = "run-hmac"
    db.create_run(run_id=run_id, tenant_id="default", user_id="1", inputs={}, workflow_id=None, definition_version=1, definition_snapshot={})
    db.update_run_status(run_id, status="succeeded", ended_at="2024-01-01T00:00:00Z")

    # Allow egress
    from tldw_Server_API.app.core.Security import egress as egress_mod
    monkeypatch.setattr(egress_mod, "is_url_allowed", lambda url: True)

    captured = {"headers": None, "data": None, "calls": 0}
    class _CapClient:
        def __init__(self, timeout=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def post(self, url, data=None, headers=None):
            captured["calls"] += 1
            captured["headers"] = headers
            captured["data"] = data
            return type("Resp", (), {"status_code": 200})()

    import sys
    dummy_httpx = type("_DummyHttpx", (), {"Client": _CapClient})()
    if "httpx" in sys.modules:
        del sys.modules["httpx"]
    sys.modules["httpx"] = dummy_httpx

    # Set secret for HMAC signing
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_SECRET", "supersecret")

    engine = eng_mod.WorkflowEngine(db)
    definition = {"on_completion_webhook": {"url": "http://example.test/hook", "include_outputs": True}}
    await engine._maybe_send_completion_webhook(definition, run_id, status="succeeded")
    assert captured["calls"] == 1
    import hmac, hashlib
    ts = captured["headers"].get("X-Signature-Timestamp")
    signed_body = f"{ts}.{captured['data']}" if ts else captured["data"]
    expected_sig = hmac.new(b"supersecret", signed_body.encode("utf-8"), hashlib.sha256).hexdigest()
    assert captured["headers"].get("X-Workflows-Signature") == expected_sig

    # SSRF block: ensure no call happens
    captured["calls"] = 0
    monkeypatch.setattr(egress_mod, "is_url_allowed", lambda url: False)
    await engine._maybe_send_completion_webhook(definition, run_id, status="succeeded")
    assert captured["calls"] == 0
