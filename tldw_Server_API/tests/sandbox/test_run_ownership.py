from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _user(uid: int, *, admin: bool = False) -> User:
    return User(id=uid, username=f"user-{uid}", is_active=True, is_admin=admin)


def _force_docker_preflight_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SandboxService,
        "_collect_runtime_preflights",
        lambda self, network_policy=None: {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                execution_mode="mocked",
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        },
    )


def test_run_owner_enforced(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    _force_docker_preflight_available(monkeypatch)

    client = TestClient(app)
    app.dependency_overrides[get_request_user] = lambda: _user(1)
    try:
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["python", "-c", "print('hello')"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]
    finally:
        app.dependency_overrides.pop(get_request_user, None)

    app.dependency_overrides[get_request_user] = lambda: _user(2)
    try:
        r2 = client.get(f"/api/v1/sandbox/runs/{run_id}")
        assert r2.status_code == 404
    finally:
        app.dependency_overrides.pop(get_request_user, None)


@pytest.mark.sandbox_ws_auth
@pytest.mark.sandbox_no_auth
def test_ws_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_WS_SIGNED_URLS", "false")
    monkeypatch.delenv("SANDBOX_WS_SIGNING_SECRET", raising=False)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    _force_docker_preflight_available(monkeypatch)

    client = TestClient(app)
    app.dependency_overrides[get_request_user] = lambda: _user(1)
    try:
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo run"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]
    finally:
        app.dependency_overrides.pop(get_request_user, None)

    with pytest.raises(Exception):
        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream"):
            pass
