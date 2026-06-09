from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

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


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
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


@pytest.mark.unit
def test_queue_wait_metric_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    # Patch the observe_histogram reference in service module
    import tldw_Server_API.app.core.Sandbox.service as svc

    def _obs(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(svc, "observe_histogram", _obs, raising=True)

    with _client(monkeypatch) as client:
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo q"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200

    # Expect at least one call to sandbox_queue_wait_seconds
    names = [a[0] if a else None for (a, _k) in calls]
    assert any(name == "sandbox_queue_wait_seconds" for name in names)
    # Validate kwargs of first matching call
    for (a, k) in calls:
        if a and a[0] == "sandbox_queue_wait_seconds":
            assert "value" in k and isinstance(k["value"], float)
            assert "labels" in k and isinstance(k["labels"], dict)
            break
