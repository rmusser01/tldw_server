from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi.testclient import TestClient
import pytest


pytestmark = pytest.mark.timeout(30)


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
    # Ensure sandbox router active
    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    _force_docker_preflight_available(monkeypatch)
    from tldw_Server_API.app.main import app as _app
    return TestClient(_app)


def test_ws_accepts_stdin_and_enforces_caps(ws_flush, monkeypatch) -> None:


    with _client(monkeypatch) as client:
        # Start a run with interactive caps
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["python", "-c", "print('hi')"],
            "timeout_sec": 5,
            "interactive": True,
            "stdin_max_bytes": 5,
            "stdin_max_frame_bytes": 3,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream") as ws:
            # Send a frame larger than per-frame cap
            ws.send_json({"type": "stdin", "encoding": "utf8", "data": "abcdef"})
            saw_trunc = False
            deadline = time.time() + 2
            while time.time() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "heartbeat":
                    continue
                if msg.get("type") == "truncated":
                    # Any truncated reason from stdin enforcement is acceptable
                    saw_trunc = True
                    break
            assert saw_trunc, "Expected a truncated frame due to stdin caps"
            ws_flush(run_id)
            ws.close()
