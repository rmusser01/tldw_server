from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.testclient import TestClient


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
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    _force_docker_preflight_available(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

    app = FastAPI()
    app.include_router(sandbox_router, prefix="/api/v1")
    return TestClient(app)


def test_artifact_content_type_and_invalid_path(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        # Create a run
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo"],
            "timeout_sec": 5,
            "capture_patterns": ["out.txt"],
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Seed artifact
        from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
        payload = b"hello world"
        sb._service._orch.store_artifacts(run_id, {"out.txt": payload})  # type: ignore[attr-defined]

        # Full download: expect content-type text/plain
        r2 = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/out.txt")
        assert r2.status_code == 200
        assert r2.headers.get("content-type", "").startswith("text/plain")
        assert r2.content == payload

        # Invalid path: traversal (encoded to avoid client-side normalization)
        r3 = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/%2E%2E/secret.txt")
        assert r3.status_code == 400

        # Invalid path: double-encoded traversal
        r3_double = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/%252E%252E/secret.txt")
        assert r3_double.status_code == 400

        # Invalid path: traversal with encoded backslash
        r3_backslash = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts/..%5csecret.txt")
        assert r3_backslash.status_code == 400

        # Invalid path: absolute
        r4 = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts//etc/passwd")
        assert r4.status_code == 400

        # Invalid range header → 416
        r5 = client.get(
            f"/api/v1/sandbox/runs/{run_id}/artifacts/out.txt",
            headers={"Range": "bytes=100-50"},
        )
        assert r5.status_code == 416
