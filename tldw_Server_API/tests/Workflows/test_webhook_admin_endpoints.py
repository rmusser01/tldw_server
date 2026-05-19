import json
import sqlite3
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.DB_Manager import create_workflows_database, get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from starlette.requests import Request


def test_dlq_list_and_replay_simulated(monkeypatch, auth_headers):


     # Simulate admin user via single-user mode and enable test-mode replay short-circuit
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("WORKFLOWS_TEST_REPLAY_SUCCESS", "true")
    # Override deps to simulate admin
    db_for_app = WorkflowsDatabase("Databases/test_wf_dlq.db")
    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)
    async def override_principal(request: Request):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="tester",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["workflows.runs.control"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
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
        return db_for_app
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    client = TestClient(app, headers=auth_headers)
    try:
        # Seed a DLQ row into the same DB the app uses
        db_for_app.enqueue_webhook_dlq(tenant_id="default", run_id="r1", url="https://example.com/hook", body={"ok": True}, last_error="init")
        # List DLQ
        resp = client.get("/api/v1/workflows/webhooks/dlq?limit=10")
        assert resp.status_code == 200
        items = resp.json().get("items") or []
        assert len(items) >= 1
        dlq_id = items[0]["id"]
        # Replay simulated (deletes row)
        r2 = client.post(f"/api/v1/workflows/webhooks/dlq/{dlq_id}/replay")
        assert r2.status_code == 200
        assert r2.json().get("ok") is True
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_dlq_replay_appends_delivery_event(monkeypatch, auth_headers, tmp_path):
    db_for_app = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    async def override_principal(request: Request):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="tester",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["workflows.runs.control"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
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
        return db_for_app

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    workflow_id = db_for_app.create_definition(
        tenant_id="default",
        name="replay-workflow",
        version=1,
        owner_id="1",
        visibility="private",
        description=None,
        tags=[],
        definition={"name": "replay", "version": 1, "steps": []},
    )
    run_id = "wf-dlq-event-run"
    db_for_app.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=workflow_id,
        definition_version=1,
        definition_snapshot={"name": "replay", "version": 1, "steps": []},
    )
    db_for_app.enqueue_webhook_dlq(
        tenant_id="default",
        run_id=run_id,
        url="https://example.com/hook",
        body={"ok": True},
        last_error="init",
    )

    class _DummyResp:
        status_code = 200

        async def aclose(self):
            return None

    async def _fake_http_afetch(**kwargs):  # noqa: ANN003
        return _DummyResp()

    monkeypatch.setattr(wf_mod, "_http_afetch", _fake_http_afetch)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.is_webhook_url_allowed_for_tenant",
        lambda url, tenant_id: True,
        raising=True,
    )

    client = TestClient(app, headers=auth_headers)
    try:
        resp = client.get("/api/v1/workflows/webhooks/dlq?limit=10")
        assert resp.status_code == 200
        dlq_id = resp.json()["items"][0]["id"]
        replay_resp = client.post(f"/api/v1/workflows/webhooks/dlq/{dlq_id}/replay")
        assert replay_resp.status_code == 200, replay_resp.text
        assert replay_resp.json()["ok"] is True
    finally:
        client.close()
        app.dependency_overrides.clear()

    events = db_for_app.get_events(run_id)
    delivery_events = [e for e in events if e.get("event_type") == "webhook_delivery"]
    assert delivery_events, "Expected replay to append a webhook_delivery event"
    payload = delivery_events[-1].get("payload_json") or {}
    assert payload.get("status") == "delivered"
    assert payload.get("source") == "dlq_replay"



def test_dlq_replay_deletes_successful_delivery_when_evidence_append_fails(monkeypatch, auth_headers, tmp_path):
    db_for_app = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    async def override_principal(request: Request):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="tester",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["workflows.runs.control"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    def override_db():
        return db_for_app

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    run_id = "wf-dlq-evidence-fails"
    db_for_app.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "replay", "version": 1, "steps": []},
    )
    db_for_app.enqueue_webhook_dlq(
        tenant_id="default",
        run_id=run_id,
        url="https://example.com/hook?sig=secret",
        body={"ok": True},
        last_error="init",
    )

    class _DummyResp:
        status_code = 200

        async def aclose(self):
            return None

    async def _fake_http_afetch(**kwargs):  # noqa: ANN003
        return _DummyResp()

    def _failing_append_event(*args, **kwargs):  # noqa: ANN002, ANN003
        raise sqlite3.OperationalError("event store unavailable")

    monkeypatch.setattr(wf_mod, "_http_afetch", _fake_http_afetch)
    monkeypatch.setattr(db_for_app, "append_event", _failing_append_event)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.is_webhook_url_allowed_for_tenant",
        lambda url, tenant_id: True,
        raising=True,
    )

    client = TestClient(app, headers=auth_headers)
    try:
        resp = client.get("/api/v1/workflows/webhooks/dlq?limit=10")
        assert resp.status_code == 200
        dlq_id = resp.json()["items"][0]["id"]
        replay_resp = client.post(f"/api/v1/workflows/webhooks/dlq/{dlq_id}/replay")
        assert replay_resp.status_code == 200, replay_resp.text
        assert replay_resp.json()["ok"] is True

        after = client.get("/api/v1/workflows/webhooks/dlq?limit=10")
        assert after.status_code == 200
        assert after.json()["items"] == []
    finally:
        client.close()
        app.dependency_overrides.clear()
