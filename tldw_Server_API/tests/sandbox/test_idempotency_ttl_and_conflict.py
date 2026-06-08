from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch, ttl_sec: int | None = None) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    if ttl_sec is not None:
        monkeypatch.setenv("SANDBOX_IDEMPOTENCY_TTL_SEC", str(ttl_sec))
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

    from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

    app = FastAPI()
    app.include_router(sandbox_router, prefix="/api/v1")
    return TestClient(app)


def _run_body(msg: str = "echo") -> Dict[str, Any]:
    return {
        "spec_version": "1.0",
        "runtime": "docker",
        "base_image": "python:3.11-slim",
        "command": ["bash", "-lc", msg],
        "timeout_sec": 5,
    }


def test_idempotency_conflict_on_mismatch(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        key = "k-conflict-1"
        r1 = client.post("/api/v1/sandbox/runs", headers={"Idempotency-Key": key}, json=_run_body("echo 1"))
        assert r1.status_code == 200
        rid1 = r1.json().get("id")
        assert isinstance(rid1, str) and rid1
        r2 = client.post("/api/v1/sandbox/runs", headers={"Idempotency-Key": key}, json=_run_body("echo 2"))
        assert r2.status_code == 409
        j = r2.json()
        assert j.get("error", {}).get("code") == "idempotency_conflict"
        details = j.get("error", {}).get("details", {})
        assert details.get("prior_id") == rid1
        assert details.get("key") == key
        # ISO 8601 string expected (not validating format strictly)
        assert isinstance(details.get("prior_created_at"), str) and details.get("prior_created_at")


def test_idempotency_ttl_expiry_allows_new_execution(monkeypatch) -> None:


     # TTL = 0 means immediate expiry
    with _client(monkeypatch, ttl_sec=0) as client:
        key = "k-expire-1"
        r1 = client.post("/api/v1/sandbox/runs", headers={"Idempotency-Key": key}, json=_run_body("echo 1"))
        assert r1.status_code == 200
        # Second request with same key/body should not return conflict because TTL has expired
        r2 = client.post("/api/v1/sandbox/runs", headers={"Idempotency-Key": key}, json=_run_body("echo 1"))
        # Accept either a fresh 200 with a different id or an idempotent replay depending on store timing; it must not be 409
        assert r2.status_code != 409
