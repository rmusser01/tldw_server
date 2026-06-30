from __future__ import annotations

import os
import time
from typing import Any, Dict, List

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
    # Import app after env so settings pick up values
    from tldw_Server_API.app.main import app as _app
    return TestClient(_app)


pytestmark = pytest.mark.timeout(30)


def test_ws_frames_include_monotonic_seq(ws_flush, monkeypatch) -> None:


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
            seqs: List[int] = []
            deadline = time.time() + 2
            while time.time() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "heartbeat":
                    continue
                # Events and streams must include seq
                assert "seq" in msg and isinstance(msg["seq"], int)
                seqs.append(int(msg["seq"]))
                if msg.get("type") == "event" and msg.get("event") == "end":
                    break
            # Expect at least two frames: start and end
            assert len(seqs) >= 2
            # Monotonic increasing
            assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
            # Force-close to avoid lingering background tasks
            ws_flush(run_id)
            ws.close()
