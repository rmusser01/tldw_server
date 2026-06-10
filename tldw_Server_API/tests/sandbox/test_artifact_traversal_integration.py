from __future__ import annotations

import os
import threading
import time

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


@pytest.mark.integration
def test_artifact_traversal_rejected_under_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only run if uvicorn is available
    try:
        import uvicorn  # type: ignore
    except ImportError:
        pytest.skip("uvicorn not installed")

    # Prepare environment for the app
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    api_key = os.environ.get("SINGLE_USER_API_KEY") or "test_sandbox_api_key_12345"
    monkeypatch.setenv("SINGLE_USER_API_KEY", api_key)
    _force_docker_preflight_available(monkeypatch)

    # Import app lazily after env is set
    from tldw_Server_API.app.main import app
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)

    # Start uvicorn server in background
    host = "127.0.0.1"
    port = 8809
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    th = threading.Thread(target=server.run, daemon=True)
    th.start()

    try:
        # Wait for server to start
        deadline = time.time() + 20
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        if not server.started:
            pytest.skip("uvicorn server did not start in time")

        # Drive API against real HTTP server so raw_path is preserved
        import requests
        # Use the same timeout for all HTTP calls to avoid hangs
        TIMEOUT = 10

        # Create a run
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo"],
            "timeout_sec": 5,
            "capture_patterns": ["out.txt"],
        }
        headers = {"X-API-KEY": api_key}
        r = requests.post(
            f"http://{host}:{port}/api/v1/sandbox/runs",
            json=body,
            headers=headers,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        run_id: str = r.json()["id"]

        # Traversal should be rejected with 400 using raw `..` segment
        r3 = requests.get(
            f"http://{host}:{port}/api/v1/sandbox/runs/{run_id}/artifacts/../secret.txt",
            headers=headers,
            timeout=TIMEOUT,
        )
        # Under servers that preserve raw_path (e.g., uvicorn+h11 without aggressive normalization),
        # the ASGI middleware and route guard return 400. Some uvicorn builds normalize the path
        # before ASGI, so the request is routed to the generic artifact handler and yields 404
        # (no artifact) rather than leaking. Treat both as acceptable denials.
        assert r3.status_code in (400, 404)
    finally:
        # Shutdown server (best-effort)
        try:
            server.should_exit = True
            th.join(timeout=2)
        except Exception as e:
            # Swallow shutdown errors to avoid masking test results
            _ = None
