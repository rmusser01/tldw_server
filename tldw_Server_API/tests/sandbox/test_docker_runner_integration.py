"""Integration tests for the Docker runner with real container lifecycle.

These tests require a working Docker daemon and are skipped in CI unless
``SANDBOX_ENABLE_EXECUTION=1`` is set and Docker is available.

Run with::

    SANDBOX_ENABLE_EXECUTION=1 python -m pytest \
        tldw_Server_API/tests/sandbox/test_docker_runner_integration.py \
        -v -m sandbox_real_docker
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

pytestmark = [
    pytest.mark.sandbox_real_docker,
    pytest.mark.skipif(
        not os.getenv("SANDBOX_ENABLE_EXECUTION"),
        reason="SANDBOX_ENABLE_EXECUTION not set",
    ),
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not found on PATH",
    ),
]


def _docker_available() -> bool:
    """Return True if the Docker daemon is reachable."""
    try:
        subprocess.check_call(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


@pytest.fixture(autouse=True)
def _skip_without_docker():
    if not _docker_available():
        pytest.skip("Docker daemon not reachable")


@pytest.fixture()
def sandbox_client(monkeypatch):
    """Yield a TestClient wired to the sandbox router with real Docker exec."""
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")

    from tldw_Server_API.app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_lifecycle(sandbox_client):
    """Create session → upload script → start run → verify exit 0 → collect artifacts."""
    client = sandbox_client

    # 1. Create a session
    session_body = {
        "spec_version": "1.0",
        "runtime": "docker",
        "base_image": "python:3.11-slim",
    }
    r = client.post("/api/v1/sandbox/sessions", json=session_body)
    assert r.status_code in (200, 201), f"session create failed: {r.text}"
    session_id = r.json().get("id") or r.json().get("session_id")
    assert session_id

    # 2. Upload a simple Python script via inline files
    run_body = {
        "session_id": session_id,
        "spec_version": "1.0",
        "runtime": "docker",
        "command": ["python", "/workspace/hello.py"],
        "files": [
            {
                "path": "hello.py",
                "content_b64": "aW1wb3J0IHN5cwpwcmludCgiSGVsbG8gZnJvbSBzYW5kYm94ISIpCnN5cy5leGl0KDAp",
            }
        ],
        "capture_patterns": ["*.py"],
        "timeout_sec": 30,
    }
    r = client.post("/api/v1/sandbox/runs", json=run_body)
    assert r.status_code in (200, 201, 202), f"run create failed: {r.text}"
    run_id = r.json().get("id") or r.json().get("run_id")
    assert run_id

    # 3. Poll until run completes (max 60s)
    deadline = time.monotonic() + 60
    final_status = None
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/sandbox/runs/{run_id}")
        if r.status_code == 200:
            data = r.json()
            phase = data.get("phase", "")
            if phase in ("completed", "finished", "failed", "killed", "timed_out", "cancelled"):
                final_status = data
                break
        time.sleep(1)

    assert final_status is not None, "Run did not complete within 60s"
    assert final_status.get("exit_code") == 0, f"Expected exit 0, got: {final_status}"

    # 4. Check artifacts
    r = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts")
    if r.status_code == 200:
        artifacts = r.json()
        if isinstance(artifacts, list):
            names = [a.get("path") or a.get("name", "") for a in artifacts]
            assert any("hello.py" in n for n in names), f"Expected hello.py in artifacts: {names}"

    # 5. Verify no orphaned containers
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--filter", f"label=tldw_run_id={run_id}", "-q"],
            text=True,
            timeout=5,
        )
        assert out.strip() == "", f"Orphaned container found: {out.strip()}"
    except subprocess.SubprocessError:
        pass  # best-effort check

    # 6. Verify no orphaned networks
    try:
        net_name = f"tldw_sbx_{run_id[:12]}"
        out = subprocess.check_output(
            ["docker", "network", "ls", "--filter", f"name={net_name}", "-q"],
            text=True,
            timeout=5,
        )
        assert out.strip() == "", f"Orphaned network found: {out.strip()}"
    except subprocess.SubprocessError:
        pass  # best-effort check
