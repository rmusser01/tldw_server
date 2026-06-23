from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


def _force_docker_preflight_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Sandbox.models import RuntimeType
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    def _preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        del self, network_policy
        return {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                execution_mode="mocked",
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _preflights)


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    _force_docker_preflight_available(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

    app = FastAPI()
    app.include_router(sandbox_router, prefix="/api/v1")
    return TestClient(app)


pytestmark = pytest.mark.timeout(10)


def test_ws_stream_fake_exec_start_end(ws_flush, monkeypatch) -> None:
    with _client(monkeypatch) as client:
        # Start a run
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["python", "-c", "print('hello')"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]
        # Subscribe to WS
        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream") as ws:
            seen_start = False
            seen_end = False
            deadline = time.time() + 2
            while time.time() < deadline and not seen_end:
                msg = ws.receive_json()
                if msg.get("type") == "event" and msg.get("event") == "start":
                    seen_start = True
                if msg.get("type") == "event" and msg.get("event") == "end":
                    seen_end = True
            assert seen_start and seen_end
            ws_flush(run_id)
            ws.close()
