from __future__ import annotations

import os
import asyncio as _asyncio
import time
from typing import Any, Dict
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Sandbox.streams import get_hub

pytestmark = pytest.mark.timeout(10)


def _force_docker_preflight_available(monkeypatch) -> None:
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
    # Disable synthetic frames for this test so heartbeats aren't suppressed by early end
    monkeypatch.setenv("SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS", "false")
    # Disable real execution so WS doesn't end immediately (to receive heartbeats)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    _force_docker_preflight_available(monkeypatch)
    return TestClient(app)


@pytest.mark.unit
def test_ws_heartbeats_include_seq(monkeypatch: pytest.MonkeyPatch, ws_flush) -> None:
    # Speed up heartbeats by monkeypatching the sandbox asyncio.sleep
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    _orig_sleep = _asyncio.sleep

    async def _fast_sleep(_n: float) -> None:  # pragma: no cover - trivial
        await _orig_sleep(0.01)

    monkeypatch.setattr(sb.asyncio, "sleep", _fast_sleep, raising=True)

    with _client(monkeypatch) as client:
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo waiting"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream") as ws:
            # Wait for a heartbeat (short deadline; heartbeat sleep patched to ~0.01s)
            deadline = time.time() + 1.0
            saw_heartbeat = False
            last_seq: int | None = None
            while time.time() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "heartbeat":
                    assert "seq" in msg and isinstance(msg["seq"], int)
                    if last_seq is not None:
                        assert msg["seq"] > last_seq
                    last_seq = msg["seq"]
                    saw_heartbeat = True
                    break
            assert saw_heartbeat, "Did not receive heartbeat with seq in time"
            ws_flush(run_id)
            ws.close()


@pytest.mark.unit
def test_ws_heartbeats_stop_after_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    _orig_sleep = _asyncio.sleep

    async def _fast_sleep(_n: float) -> None:  # pragma: no cover - trivial
        await _orig_sleep(0.01)

    monkeypatch.setattr(sb.asyncio, "sleep", _fast_sleep, raising=True)

    with _client(monkeypatch) as client:
        run_id = f"run-{uuid4()}"
        hub = get_hub()
        original_publish_heartbeat = hub.publish_heartbeat
        heartbeat_states: list[bool] = []

        def _tracked_publish_heartbeat(target_run_id: str) -> None:
            if target_run_id == run_id:
                heartbeat_states.append(target_run_id in hub._ended)
            original_publish_heartbeat(target_run_id)

        monkeypatch.setattr(hub, "publish_heartbeat", _tracked_publish_heartbeat, raising=True)

        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream") as ws:
            deadline = time.time() + 1.0
            hub.publish_event(run_id, "start", {"source": "heartbeat-stop"})
            while time.time() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "heartbeat":
                    break
            else:
                pytest.fail("Did not receive initial heartbeat")

            hub.publish_event(run_id, "end", {})
            time.sleep(0.1)

        assert True not in heartbeat_states
