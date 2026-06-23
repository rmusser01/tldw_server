from __future__ import annotations

import os
from urllib.parse import urlparse, parse_qs

from fastapi.testclient import TestClient
import pytest


pytestmark = pytest.mark.timeout(10)


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
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    # Ensure router enabled
    existing = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    _force_docker_preflight_available(monkeypatch)
    from tldw_Server_API.app.main import app as _app
    return TestClient(_app)


def test_post_runs_exposes_resume_from_seq_in_url(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo run"],
            "timeout_sec": 5,
            "resume_from_seq": 7,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        url = r.json().get("log_stream_url")
        assert isinstance(url, str)
        p = urlparse(url)
        qs = parse_qs(p.query)
        assert int(qs.get("from_seq", ["0"])[0]) == 7
