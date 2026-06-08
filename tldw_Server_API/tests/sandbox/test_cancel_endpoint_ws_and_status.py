from __future__ import annotations

import os
import time
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    # Disable real execution to keep run queued/non-terminal for cancel
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
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


@pytest.mark.unit
def test_cancel_endpoint_sends_single_end_and_sets_killed(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        # Start a run (will be queued due to execution disabled)
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo running"],
            "timeout_sec": 30,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Open WS stream and then cancel
        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream") as ws:
            # Issue cancel
            r2 = client.post(f"/api/v1/sandbox/runs/{run_id}/cancel")
            assert r2.status_code == 200
            assert r2.json().get("cancelled") is True

            # Read frames until end; ensure exactly one end
            end_count = 0
            deadline = time.time() + 3
            while time.time() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "event" and msg.get("event") == "end":
                    end_count += 1
                    break
            assert end_count == 1

        # Run status should be killed
        r3 = client.get(f"/api/v1/sandbox/runs/{run_id}")
        assert r3.status_code == 200
        assert r3.json().get("phase") == "killed"
