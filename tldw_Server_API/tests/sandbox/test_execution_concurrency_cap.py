from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RunStatus, RuntimeType
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


def _wait_for_phase(
    svc: SandboxService,
    run_id: str,
    phase: RunPhase,
    timeout_sec: float = 3.0,
) -> RunStatus | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        st = svc.get_run(run_id)
        if st is not None and st.phase == phase:
            return st
        time.sleep(0.02)
    return None


def test_background_execution_respects_max_concurrent_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("SANDBOX_RUN_CLAIM_LEASE_SEC", "30")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    _force_docker_preflight_available(monkeypatch)

    lock = threading.Lock()
    allow_first_finish = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    active = 0
    peak = 0
    start_order: list[str] = []

    def _fake_start_run(self, run_id: str, spec: RunSpec, workspace_path: str | None) -> RunStatus:
        nonlocal active, peak
        with lock:
            start_order.append(run_id)
            active += 1
            peak = max(peak, active)
            index = len(start_order)
        if index == 1:
            first_started.set()
            allow_first_finish.wait(timeout=10.0)
        else:
            second_started.set()
        now = datetime.now(timezone.utc)
        with lock:
            active -= 1
        return RunStatus(
            id=run_id,
            phase=RunPhase.completed,
            runtime=RuntimeType.docker,
            base_image=spec.base_image,
            exit_code=0,
            started_at=now,
            finished_at=now,
            message="ok",
        )

    monkeypatch.setattr(DockerRunner, "start_run", _fake_start_run)

    svc = SandboxService()
    run1 = svc.start_run_scaffold(
        user_id="user-cap",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "one"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "one"]},
    )
    assert first_started.wait(timeout=1.0) is True

    run2 = svc.start_run_scaffold(
        user_id="user-cap",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "two"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "two"]},
    )

    time.sleep(0.15)
    assert second_started.is_set() is False

    allow_first_finish.set()
    assert second_started.wait(timeout=2.0) is True

    done1 = _wait_for_phase(svc, run1.id, RunPhase.completed)
    done2 = _wait_for_phase(svc, run2.id, RunPhase.completed)
    assert done1 is not None
    assert done2 is not None
    assert peak == 1


def test_global_active_cap_enforced_across_service_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("SANDBOX_RUN_CLAIM_LEASE_SEC", "30")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    _force_docker_preflight_available(monkeypatch)

    lock = threading.Lock()
    allow_first_finish = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    active = 0
    peak = 0
    start_order: list[str] = []

    def _fake_start_run(self, run_id: str, spec: RunSpec, workspace_path: str | None) -> RunStatus:
        nonlocal active, peak
        with lock:
            start_order.append(run_id)
            active += 1
            peak = max(peak, active)
            index = len(start_order)
        if index == 1:
            first_started.set()
            allow_first_finish.wait(timeout=10.0)
        else:
            second_started.set()
        now = datetime.now(timezone.utc)
        with lock:
            active -= 1
        return RunStatus(
            id=run_id,
            phase=RunPhase.completed,
            runtime=RuntimeType.docker,
            base_image=spec.base_image,
            exit_code=0,
            started_at=now,
            finished_at=now,
            message="ok",
        )

    monkeypatch.setattr(DockerRunner, "start_run", _fake_start_run)

    svc_a = SandboxService()
    svc_b = SandboxService()
    run1 = svc_a.start_run_scaffold(
        user_id="user-cap-a",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "one"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "one"]},
    )
    assert first_started.wait(timeout=1.0) is True

    run2 = svc_b.start_run_scaffold(
        user_id="user-cap-b",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "two"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "two"]},
    )

    time.sleep(0.15)
    assert second_started.is_set() is False

    allow_first_finish.set()
    assert second_started.wait(timeout=2.0) is True

    done1 = _wait_for_phase(svc_a, run1.id, RunPhase.completed)
    done2 = _wait_for_phase(svc_b, run2.id, RunPhase.completed)
    assert done1 is not None
    assert done2 is not None
    assert peak == 1


def test_per_user_active_cap_enforced_across_service_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_MAX_CONCURRENT_RUNS", "2")
    monkeypatch.setenv("SANDBOX_ACTIVE_MAX_PER_USER", "1")
    monkeypatch.setenv("SANDBOX_RUN_CLAIM_LEASE_SEC", "30")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    _force_docker_preflight_available(monkeypatch)

    lock = threading.Lock()
    allow_first_finish = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    active = 0
    peak = 0
    start_order: list[str] = []

    def _fake_start_run(self, run_id: str, spec: RunSpec, workspace_path: str | None) -> RunStatus:
        nonlocal active, peak
        with lock:
            start_order.append(run_id)
            active += 1
            peak = max(peak, active)
            index = len(start_order)
        if index == 1:
            first_started.set()
            allow_first_finish.wait(timeout=10.0)
        else:
            second_started.set()
        now = datetime.now(timezone.utc)
        with lock:
            active -= 1
        return RunStatus(
            id=run_id,
            phase=RunPhase.completed,
            runtime=RuntimeType.docker,
            base_image=spec.base_image,
            exit_code=0,
            started_at=now,
            finished_at=now,
            message="ok",
        )

    monkeypatch.setattr(DockerRunner, "start_run", _fake_start_run)

    svc_a = SandboxService()
    svc_b = SandboxService()
    run1 = svc_a.start_run_scaffold(
        user_id="user-same",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "one"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "one"]},
    )
    assert first_started.wait(timeout=1.0) is True

    run2 = svc_b.start_run_scaffold(
        user_id="user-same",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["echo", "two"],
        ),
        spec_version="1.0",
        idem_key=None,
        raw_body={"command": ["echo", "two"]},
    )

    time.sleep(0.15)
    assert second_started.is_set() is False

    allow_first_finish.set()
    assert second_started.wait(timeout=2.0) is True

    done1 = _wait_for_phase(svc_a, run1.id, RunPhase.completed)
    done2 = _wait_for_phase(svc_b, run2.id, RunPhase.completed)
    assert done1 is not None
    assert done2 is not None
    assert peak == 1
