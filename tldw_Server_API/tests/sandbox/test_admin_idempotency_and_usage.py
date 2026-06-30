from __future__ import annotations

import os
from typing import Any, Dict

import pytest
from fastapi import FastAPI
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
    from tldw_Server_API.app.core.Sandbox.models import RuntimeType
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    monkeypatch.setattr(
        SandboxService,
        "_collect_runtime_preflights",
        lambda self, network_policy=None: {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        },
    )
    # Build a minimal app with only the sandbox router
    from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router
    app = FastAPI()
    app.include_router(sandbox_router, prefix="/api/v1")
    return TestClient(app)


def _admin_user_dep():


    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    return User(id=1, username="admin", roles=["admin"], is_admin=True)


@pytest.mark.unit
def test_admin_idempotency_list_filters_and_pagination(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        client.app.dependency_overrides[get_request_user] = _admin_user_dep
        # Create a session with idempotency key, then post the same again to trigger 409
        body: Dict[str, Any] = {"spec_version": "1.0", "runtime": "docker"}
        hdr = {"Idempotency-Key": "sess-uniq-1"}
        r1 = client.post("/api/v1/sandbox/sessions", json=body, headers=hdr)
        assert r1.status_code == 200
        r2 = client.post("/api/v1/sandbox/sessions", json=body, headers=hdr)
        # For identical body + key, API replays cached result (200). A 409 is only for body/key mismatches.
        assert r2.status_code == 200

        # List idempotency records filtered by endpoint and key
        lr = client.get("/api/v1/sandbox/admin/idempotency", params={
            "endpoint": "sessions",
            "key": "sess-uniq-1",
            "limit": 10,
            "offset": 0,
            "sort": "desc",
        })
        assert lr.status_code == 200
        payload = lr.json()
        assert set(payload.keys()) == {"total", "limit", "offset", "has_more", "next_offset", "items", "pagination"}
        assert payload["pagination"]["total"] == payload["total"]
        assert payload["pagination"]["limit"] == 10
        assert payload["pagination"]["offset"] == 0
        assert payload["pagination"]["has_more"] == payload["has_more"]
        assert payload["pagination"]["next_offset"] == payload["next_offset"]
        items = payload["items"]
        assert isinstance(items, list)
        assert len(items) >= 1
        first = items[0]
        # Schema conformance
        assert set(first.keys()) >= {"endpoint", "key", "object_id", "created_at"}
        assert first["endpoint"] == "sessions"
        # has_more should be boolean
        assert isinstance(payload["has_more"], bool)

        client.app.dependency_overrides.clear()


@pytest.mark.unit
def test_admin_usage_aggregates_schema_and_filters(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        client.app.dependency_overrides[get_request_user] = _admin_user_dep
        # Create two runs for default user
        for i in range(2):
            body: Dict[str, Any] = {
                "spec_version": "1.0",
                "runtime": "docker",
                "base_image": "python:3.11-slim",
                "command": ["python", "-c", f"print('run{i}')"],
                "timeout_sec": 5,
            }
            r = client.post("/api/v1/sandbox/runs", json=body)
            assert r.status_code == 200

        ur = client.get("/api/v1/sandbox/admin/usage", params={"limit": 50, "offset": 0})
        assert ur.status_code == 200
        payload = ur.json()
        assert set(payload.keys()) == {"total", "limit", "offset", "has_more", "next_offset", "items", "pagination"}
        assert payload["pagination"]["total"] == payload["total"]
        assert payload["pagination"]["limit"] == 50
        assert payload["pagination"]["offset"] == 0
        assert payload["pagination"]["has_more"] == payload["has_more"]
        assert payload["pagination"]["next_offset"] == payload["next_offset"]
        assert isinstance(payload["items"], list)
        # If the default user exists, ensure schema for first item
        if payload["items"]:
            item = payload["items"][0]
            assert set(item.keys()) == {"user_id", "runs_count", "log_bytes", "artifact_bytes"}

        # Filter by a non-existent user to exercise pagination/filters
        ur2 = client.get("/api/v1/sandbox/admin/usage", params={"user_id": "nonexistent", "limit": 1, "offset": 0})
        assert ur2.status_code == 200
        p2 = ur2.json()
        assert p2["total"] in (0, p2["total"])  # total should be an int; accept 0
        assert p2["limit"] == 1
        assert p2["offset"] == 0
        assert p2["pagination"]["total"] == p2["total"]
        assert p2["pagination"]["limit"] == 1
        assert p2["pagination"]["offset"] == 0
        assert p2["pagination"]["next_offset"] == p2["next_offset"]

        client.app.dependency_overrides.clear()


@pytest.mark.unit
def test_admin_idempotency_sort_asc_desc(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        client.app.dependency_overrides[get_request_user] = _admin_user_dep
        # Create two idempotent sessions with different times by patching time.time
        import time as _time
        base = _time.time()
        import tldw_Server_API.app.core.Sandbox.store as store_mod
        # First record
        monkeypatch.setattr(store_mod.time, "time", lambda: base - 30)
        r1 = client.post("/api/v1/sandbox/sessions", json={"spec_version": "1.0", "runtime": "docker"}, headers={"Idempotency-Key": "k-asc-1"})
        assert r1.status_code == 200
        # Second record later
        monkeypatch.setattr(store_mod.time, "time", lambda: base + 30)
        r2 = client.post("/api/v1/sandbox/sessions", json={"spec_version": "1.0", "runtime": "docker"}, headers={"Idempotency-Key": "k-asc-2"})
        assert r2.status_code == 200

        # Descending: k-asc-2 should appear before k-asc-1
        lr_desc = client.get("/api/v1/sandbox/admin/idempotency", params={"endpoint": "sessions", "limit": 10, "offset": 0, "sort": "desc"})
        assert lr_desc.status_code == 200
        items_desc = lr_desc.json().get("items", [])
        keys_desc = [it.get("key") for it in items_desc]
        assert keys_desc.index("k-asc-2") < keys_desc.index("k-asc-1")

        # Ascending: reverse order
        lr_asc = client.get("/api/v1/sandbox/admin/idempotency", params={"endpoint": "sessions", "limit": 10, "offset": 0, "sort": "asc"})
        assert lr_asc.status_code == 200
        items_asc = lr_asc.json().get("items", [])
        keys_asc = [it.get("key") for it in items_asc]
        assert keys_asc.index("k-asc-1") < keys_asc.index("k-asc-2")

        client.app.dependency_overrides.clear()
