from __future__ import annotations

import os
from typing import Any

from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:


     # Speed up and stabilize sandbox WS behavior in tests
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    # Ensure sandbox routes are enabled
    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    from tldw_Server_API.app.main import app  # import after env is set
    return TestClient(app)


def _admin_user_dep():


    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    return User(id=1, username="admin", roles=["admin"], is_admin=True)


def test_admin_details_includes_resource_usage(monkeypatch) -> None:


     # Override dependency for admin route using the app from TestClient
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    with _client(monkeypatch) as client:
        client.app.dependency_overrides[get_request_user] = _admin_user_dep
        body: dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["python", "-c", "print('ok')"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Fetch admin details and verify resource_usage structure
        rd = client.get(f"/api/v1/sandbox/admin/runs/{run_id}")
        assert rd.status_code == 200
        j = rd.json()
        assert j.get("id") == run_id
        assert j.get("status_reason_code") == "completed"
        assert j.get("status_reason_details", {}).get("code") == "completed"
        assert j.get("status_reason_details", {}).get("category") == "success"
        ru = j.get("resource_usage")
        assert isinstance(ru, dict)
        # Expect keys; values are ints (may be 0 in fake exec)
        for k in ("cpu_time_sec", "wall_time_sec", "peak_rss_mb", "log_bytes", "artifact_bytes"):
            assert k in ru
            assert isinstance(ru[k], int)
        # Clear overrides to avoid leaking into other tests
        client.app.dependency_overrides.clear()
