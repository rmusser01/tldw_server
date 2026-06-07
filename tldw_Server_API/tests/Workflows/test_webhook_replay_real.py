import threading
import http.server
import socketserver
import json
import time
import os
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from starlette.requests import Request


class _Recorder:
    def __init__(self):
        self.requests = []

    def add(self, item):

        self.requests.append(item)


def _start_test_server():


    rec = _Recorder()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get('content-length') or 0)
            except Exception:
                length = 0
            body = self.rfile.read(length) if length else b""
            rec.add({
                'path': self.path,
                'headers': {k.lower(): v for k, v in self.headers.items()},
                'body': body.decode('utf-8', errors='replace')
            })
            self.send_response(200)
            self.send_header('content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *args, **kwargs):  # noqa: D401
            return

    # Bind to ephemeral port; skip test gracefully if sandbox forbids binding
    try:
        httpd = socketserver.TCPServer(('127.0.0.1', 0), Handler)
    except PermissionError:
        pytest.skip("Cannot bind local test webhook server in this environment")
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, port, rec


@pytest.fixture()
def admin_client(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

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

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client, db

    app.dependency_overrides.clear()


def test_dlq_replay_real_allowed(monkeypatch, admin_client):


    client, db = admin_client
    httpd, thread, port, rec = _start_test_server()
    try:
        # Allow localhost and chosen port; do not block private; set signing secret
        monkeypatch.setenv('WORKFLOWS_EGRESS_BLOCK_PRIVATE', 'false')
        monkeypatch.setenv('WORKFLOWS_EGRESS_ALLOWLIST', '127.0.0.1,localhost')
        monkeypatch.delenv('WORKFLOWS_EGRESS_DENYLIST', raising=False)
        monkeypatch.setenv('WORKFLOWS_WEBHOOK_ALLOWLIST', '127.0.0.1')
        monkeypatch.delenv('WORKFLOWS_WEBHOOK_DENYLIST', raising=False)
        monkeypatch.setenv('WORKFLOWS_EGRESS_ALLOWED_PORTS', f'80,443,{port}')
        monkeypatch.setenv('WORKFLOWS_WEBHOOK_SECRET', 'testsecret')
        # Seed DLQ row in same DB used by the app
        url = f'http://127.0.0.1:{port}/hook'
        db.enqueue_webhook_dlq(tenant_id='default', run_id='r1', url=url, body={'ok': True}, last_error='seed')
        # Fetch DLQ id then replay
        resp = client.get('/api/v1/workflows/webhooks/dlq?limit=10')
        assert resp.status_code == 200, resp.text
        items = resp.json().get('items') or []
        assert items, 'Expected at least one DLQ item'
        dlq_id = items[0]['id']
        r2 = client.post(f'/api/v1/workflows/webhooks/dlq/{dlq_id}/replay')
        assert r2.status_code == 200, r2.text
        assert r2.json().get('ok') is True
        # Give server a moment to process
        deadline = time.time() + 2
        while time.time() < deadline and not rec.requests:
            time.sleep(0.05)
        assert rec.requests, 'Server did not receive the replay'
        last = rec.requests[-1]
        # Signature headers should be present
        hdrs = last['headers']
        assert 'x-signature-timestamp' in hdrs
        assert 'x-workflows-signature' in hdrs or 'x-hub-signature-256' in hdrs
    finally:
        httpd.shutdown()


def test_dlq_replay_real_denied(monkeypatch, admin_client):


    client, db = admin_client
    httpd, thread, port, rec = _start_test_server()
    try:
        # Block private or deny host explicitly
        monkeypatch.setenv('WORKFLOWS_EGRESS_BLOCK_PRIVATE', 'true')
        monkeypatch.delenv('WORKFLOWS_WEBHOOK_ALLOWLIST', raising=False)
        # OR add explicit deny
        monkeypatch.setenv('WORKFLOWS_WEBHOOK_DENYLIST', '127.0.0.1')
        url = f'http://127.0.0.1:{port}/hook'
        db.enqueue_webhook_dlq(tenant_id='default', run_id='r2', url=url, body={'ok': True}, last_error='seed')
        resp = client.get('/api/v1/workflows/webhooks/dlq?limit=10')
        assert resp.status_code == 200, resp.text
        items = resp.json().get('items') or []
        target = None
        for it in items:
            if it['url'] == url:
                target = it
                break
        assert target is not None
        dlq_id = target['id']
        r2 = client.post(f'/api/v1/workflows/webhooks/dlq/{dlq_id}/replay')
        assert r2.status_code == 400, r2.text
        assert 'Denied' in r2.text or 'policy' in r2.text
        # Ensure server did not receive a request
        time.sleep(0.1)
        assert not rec.requests
    finally:
        httpd.shutdown()
