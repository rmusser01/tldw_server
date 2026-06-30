from __future__ import annotations

import os
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient


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


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_WS_POLL_TIMEOUT_SEC", "1")
    monkeypatch.setenv("SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS", "false")
    monkeypatch.setenv("SANDBOX_WS_MAX_CONNECTIONS_PER_USER", "1")

    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    _force_docker_preflight_available(monkeypatch)

    from tldw_Server_API.app.main import app

    return TestClient(app)


def _create_run(client: TestClient) -> str:
    body: Dict[str, Any] = {
        "spec_version": "1.0",
        "runtime": "docker",
        "base_image": "python:3.11-slim",
        "command": ["echo", "ok"],
        "timeout_sec": 5,
    }
    resp = client.post("/api/v1/sandbox/runs", json=body)
    assert resp.status_code == 200
    return str(resp.json()["id"])


def test_sandbox_ws_per_user_quota_enforced_and_released(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        run_id = _create_run(client)

        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream"):
            with pytest.raises(Exception):
                with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream"):
                    pass

        # Ensure quota slot is released on disconnect.
        with client.websocket_connect(f"/api/v1/sandbox/runs/{run_id}/stream"):
            pass
