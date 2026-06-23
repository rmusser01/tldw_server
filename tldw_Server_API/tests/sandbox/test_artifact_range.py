from __future__ import annotations

import os
from typing import Any, Dict

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _force_docker_preflight_available(monkeypatch) -> None:
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
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    _force_docker_preflight_available(monkeypatch)
    return TestClient(app)


def test_artifact_download_range_support(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        # Create a run
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo done"],
            "timeout_sec": 5,
            "capture_patterns": ["out.txt"],
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Seed artifact via orchestrator (test helper)
        from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

        payload = b"0123456789"
        sb._service._orch.store_artifacts(run_id, {"out.txt": payload})  # type: ignore[attr-defined]

        # Range: bytes=0-4
        r2 = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/out.txt", headers={"Range": "bytes=0-4"})
        assert r2.status_code == 206
        assert r2.headers.get("Content-Range") == "bytes 0-4/10"
        assert r2.headers.get("Accept-Ranges") == "bytes"
        assert r2.headers.get("Content-Length") == "5"
        assert r2.content == b"01234"

        # Suffix range: last 3 bytes
        r3 = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/out.txt", headers={"Range": "bytes=-3"})
        assert r3.status_code == 206
        assert r3.headers.get("Content-Range") == "bytes 7-9/10"
        assert r3.headers.get("Content-Length") == "3"
        assert r3.content == b"789"


def test_artifact_download_range_avoids_full_artifact_read(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo done"],
            "timeout_sec": 5,
            "capture_patterns": ["out.txt"],
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

        payload = b"0123456789"
        sb._service._orch.store_artifacts(run_id, {"out.txt": payload})  # type: ignore[attr-defined]

        def _fail_get_artifact(*args, **kwargs):
            raise AssertionError("download path should stream from disk instead of loading full artifact")

        monkeypatch.setattr(sb._service._orch, "get_artifact", _fail_get_artifact)

        r2 = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/out.txt", headers={"Range": "bytes=0-4"})
        assert r2.status_code == 206
        assert r2.headers.get("Content-Range") == "bytes 0-4/10"
        assert r2.content == b"01234"
