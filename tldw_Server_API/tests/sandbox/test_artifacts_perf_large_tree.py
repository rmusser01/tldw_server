from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Sandbox.models import RuntimeType


def _force_docker_preflight_available(monkeypatch) -> None:
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
    # Minimal app with sandbox router enabled
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    _force_docker_preflight_available(monkeypatch)
    return TestClient(app)


def test_artifacts_list_perf_large_tree(tmp_path: Path, monkeypatch) -> None:
    # Use a shared artifacts dir under tmp to avoid polluting repo
    monkeypatch.setenv("SANDBOX_SHARED_ARTIFACTS_DIR", str(tmp_path))

    with _client(monkeypatch) as client:
        # Create a run
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo done"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Seed a moderately large nested tree of artifacts
        from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

        files: Dict[str, bytes] = {}
        # 300 small files across 6 directories
        for i in range(300):
            sub = f"d{i // 50}"
            rel = f"{sub}/file_{i}.txt"
            files[rel] = f"payload-{i}".encode("utf-8")
        sb._service._orch.store_artifacts(run_id, files)  # type: ignore[attr-defined]

        # List artifacts and assert it completes quickly and returns full set
        t0 = time.perf_counter()
        lr = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts")
        dt = time.perf_counter() - t0
        assert lr.status_code == 200
        items = lr.json().get("items", [])
        assert len(items) == len(files)
        # Generous threshold to avoid flakiness in CI
        assert dt < 5.0, f"artifact listing too slow: {dt:.3f}s for {len(files)} files"
