from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.orchestrator import SandboxOrchestrator
from tldw_Server_API.app.core.Sandbox.runners.docker_runner import DockerRunner
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService

pytestmark = pytest.mark.unit


def _configure_sqlite_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = str(tmp_path / "sandbox_store.db")
    root_dir = str(tmp_path / "sandbox_root")
    snapshot_dir = str(tmp_path / "snapshots")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("SANDBOX_STORE_DB_PATH", db_path)
    monkeypatch.setenv("SANDBOX_ROOT_DIR", root_dir)
    monkeypatch.setenv("SANDBOX_SNAPSHOT_PATH", snapshot_dir)
    if hasattr(app_settings, "SANDBOX_STORE_BACKEND"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_BACKEND", "sqlite")
    if hasattr(app_settings, "SANDBOX_STORE_DB_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_DB_PATH", db_path)
    if hasattr(app_settings, "SANDBOX_ROOT_DIR"):
        monkeypatch.setattr(app_settings, "SANDBOX_ROOT_DIR", root_dir)
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_PATH", snapshot_dir)
    clear_config_cache()


def _force_docker_preflight_available(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_claim_heartbeat_prevents_takeover_during_long_foreground_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    _force_docker_preflight_available(monkeypatch)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_RUN_CLAIM_LEASE_SEC", "1")

    intruder = SandboxOrchestrator()
    attempts: list[bool] = []

    def _fake_start_run(self, run_id: str, spec: RunSpec, workspace_path: str | None) -> RunStatus:
        # Hold execution long enough to exceed one lease interval. If heartbeat
        # renewal is missing, the intruder claim attempt will succeed.
        time.sleep(1.25)
        attempts.append(
            intruder.try_claim_run(run_id, worker_id="intruder-worker", lease_seconds=30) is not None
        )
        time.sleep(0.6)
        attempts.append(
            intruder.try_claim_run(run_id, worker_id="intruder-worker", lease_seconds=30) is not None
        )
        now = datetime.now(timezone.utc)
        return RunStatus(
            id=run_id,
            phase=RunPhase.completed,
            runtime=RuntimeType.docker,
            base_image=spec.base_image,
            exit_code=0,
            started_at=now,
            finished_at=now,
            message="ok",
            resource_usage={
                "cpu_time_sec": 1,
                "wall_time_sec": 1,
                "peak_rss_mb": 16,
                "log_bytes": 0,
                "artifact_bytes": 0,
            },
        )

    monkeypatch.setattr(DockerRunner, "start_run", _fake_start_run)

    svc = SandboxService()
    status = svc.start_run_scaffold(
        user_id="user-lease",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "lease-test"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "lease-test"]},
    )

    assert status.phase == RunPhase.completed
    assert attempts == [False, False]
    stored = svc.get_run(status.id)
    assert stored is not None
    assert stored.phase == RunPhase.completed
    assert stored.claim_owner is None
    assert stored.claim_expires_at is None
