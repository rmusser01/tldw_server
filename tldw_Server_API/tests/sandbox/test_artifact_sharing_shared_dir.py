from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    # Ensure sandbox router is active
    existing = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing.split(",") if p.strip()]
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
                execution_mode="mocked",
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        },
    )
    # Build a minimal app with only the sandbox router
    from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router
    app = FastAPI()
    app.include_router(sandbox_router, prefix="/api/v1")
    return TestClient(app)


@pytest.mark.unit
def test_shared_artifacts_directory_persists_and_is_listed(tmp_path: Path, monkeypatch) -> None:
    # Point shared artifacts root at tmp_path
    monkeypatch.setenv("SANDBOX_SHARED_ARTIFACTS_DIR", str(tmp_path))
    with _client(monkeypatch) as client:
        # Create a run (fake exec)
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["python", "-c", "print('ok')"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Inject artifacts using orchestrator to simulate runner capture
        from tldw_Server_API.app.api.v1.endpoints.sandbox import _service  # type: ignore
        _service._orch.store_artifacts(run_id, {  # type: ignore[attr-defined]
            "results/a.txt": b"hello",
            "b.bin": b"\x00\x01",
        })

        # Create a new client (simulating a second worker process) and list artifacts
        with _client(monkeypatch) as client2:
            lr = client2.get(f"/api/v1/sandbox/runs/{run_id}/artifacts")
            assert lr.status_code == 200
            items = lr.json().get("items")
            paths = sorted([it.get("path") for it in items])
            assert paths == ["b.bin", "results/a.txt"]
            # Download one artifact
            dr = client2.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/results/a.txt")
            assert dr.status_code == 200
            assert dr.content == b"hello"
