from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.main import app

_NATIVE_ASYNCIO_SLEEP = asyncio.sleep


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

        files: dict[str, bytes] = {}
        # 300 small files across 6 directories
        for i in range(300):
            sub = f"d{i // 50}"
            rel = f"{sub}/file_{i}.txt"
            files[rel] = f"payload-{i}".encode()
        sb._service._orch.store_artifacts(run_id, files)  # type: ignore[attr-defined]

        # Keep periodic janitor work out of the cached-listing measurement.
        sb._service._orch._maybe_prune_expired_artifacts()  # type: ignore[attr-defined]
        # List artifacts and assert it completes quickly and returns full set
        t0 = time.perf_counter()
        lr = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts")
        dt = time.perf_counter() - t0
        assert lr.status_code == 200
        items = lr.json().get("items", [])
        assert len(items) == len(files)
        # Generous threshold to avoid flakiness in CI
        assert dt < 5.0, f"artifact listing too slow: {dt:.3f}s for {len(files)} files"


def test_artifacts_list_uses_cached_sizes_before_filesystem_walk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_SHARED_ARTIFACTS_DIR", str(tmp_path))

    with _client(monkeypatch) as client:
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

        from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
        from tldw_Server_API.app.core.Sandbox import orchestrator as orchestrator_module

        sb._service._orch.store_artifacts(  # type: ignore[attr-defined]
            run_id,
            {
                "nested/out.txt": b"hello",
                "summary.txt": b"ok",
            },
        )

        def _unexpected_walk(*_args, **_kwargs):
            raise AssertionError("cached artifact listing should not walk the filesystem")

        monkeypatch.setattr(orchestrator_module.os, "walk", _unexpected_walk)

        lr = client.get(f"/api/v1/sandbox/runs/{run_id}/artifacts")
        assert lr.status_code == 200
        items = lr.json().get("items", [])
        assert sorted((item["path"], item["size"]) for item in items) == [
            ("nested/out.txt", 5),
            ("summary.txt", 2),
        ]


def test_artifact_context_preserves_native_asyncio_sleep() -> None:
    """Artifact tests must retain Jobs backoff and other process-wide clocks."""
    assert asyncio.sleep is _NATIVE_ASYNCIO_SLEEP


def test_heartbeat_fixture_preserves_native_asyncio_sleep(
    patch_sandbox_heartbeat_sleep,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    assert asyncio.sleep is _NATIVE_ASYNCIO_SLEEP
    assert sb.asyncio is not asyncio
    assert sb.asyncio.create_task is asyncio.create_task
    assert sb.asyncio.wait_for is asyncio.wait_for
    assert sb.asyncio.to_thread is asyncio.to_thread
    assert sb.asyncio.Task is asyncio.Task
    assert sb.asyncio.TimeoutError is asyncio.TimeoutError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delay", "expected_delay"),
    [(0.05, 0.05), (0.5, 0.5), (2, 2), (300, 300), (10, 0.01)],
)
async def test_heartbeat_fixture_only_shortens_heartbeat_wait(
    monkeypatch: pytest.MonkeyPatch,
    delay: float,
    expected_delay: float,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    from tldw_Server_API.tests.sandbox import conftest as sandbox_fixtures

    requested_delays = []

    async def record_sleep(requested_delay: float, result: object = None) -> object:
        requested_delays.append(requested_delay)
        return result

    # Substitute only the fixture's clock reference, keeping process-wide
    # asyncio native even while checking a 300-second requested delay.
    test_clock = SimpleNamespace(**{**vars(asyncio), "sleep": record_sleep})
    monkeypatch.setattr(sandbox_fixtures, "_asyncio", test_clock)
    sandbox_fixtures.patch_sandbox_heartbeat_sleep.__wrapped__(monkeypatch)
    assert asyncio.sleep is _NATIVE_ASYNCIO_SLEEP
    await sb.asyncio.sleep(delay)
    assert requested_delays == [expected_delay]
