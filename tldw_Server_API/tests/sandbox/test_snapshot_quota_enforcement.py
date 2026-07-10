from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.snapshots import SnapshotManager

pytestmark = pytest.mark.unit


def _force_docker_preflight_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult

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


@pytest.fixture(autouse=True)
def _snapshot_quota_docker_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_docker_preflight_available(monkeypatch)


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


def test_create_snapshot_enforces_count_quota_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_COUNT", "2")
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_SIZE_MB", "256")

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-snap",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc.get_session_workspace_path(session.id)
    assert ws is not None
    ws_path = Path(str(ws))

    (ws_path / "state.txt").write_text("v1", encoding="utf-8")
    snap1 = svc.create_snapshot(session.id)
    (ws_path / "state.txt").write_text("v2", encoding="utf-8")
    snap2 = svc.create_snapshot(session.id)
    (ws_path / "state.txt").write_text("v3", encoding="utf-8")
    snap3 = svc.create_snapshot(session.id)

    ids = [s.get("snapshot_id") for s in svc.list_snapshots(session.id)]
    assert len(ids) == 2
    assert snap1["snapshot_id"] not in ids
    assert snap2["snapshot_id"] in ids
    assert snap3["snapshot_id"] in ids


def test_create_snapshot_enforces_size_quota_under_large_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_COUNT", "10")
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_SIZE_MB", "1")

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-snap-size",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc.get_session_workspace_path(session.id)
    assert ws is not None
    ws_path = Path(str(ws))

    blob = ws_path / "blob.bin"
    blob.write_bytes(os.urandom(800_000))
    snap1 = svc.create_snapshot(session.id)
    blob.write_bytes(os.urandom(800_000))
    snap2 = svc.create_snapshot(session.id)

    snapshots = svc.list_snapshots(session.id)
    ids = [s.get("snapshot_id") for s in snapshots]
    assert len(ids) == 1
    assert snap1["snapshot_id"] not in ids
    assert snap2["snapshot_id"] in ids


def test_maintenance_enforces_snapshot_quota_for_existing_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_COUNT", "5")
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_SIZE_MB", "256")
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_MAX_COUNT"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_MAX_COUNT", 5)
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_MAX_SIZE_MB"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_MAX_SIZE_MB", 256)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-snap-maint",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc.get_session_workspace_path(session.id)
    assert ws is not None
    ws_path = Path(str(ws))

    (ws_path / "state.txt").write_text("v1", encoding="utf-8")
    svc.create_snapshot(session.id)
    (ws_path / "state.txt").write_text("v2", encoding="utf-8")
    svc.create_snapshot(session.id)
    (ws_path / "state.txt").write_text("v3", encoding="utf-8")
    svc.create_snapshot(session.id)
    assert len(svc.list_snapshots(session.id)) == 3

    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_COUNT", "1")
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_MAX_COUNT"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_MAX_COUNT", 1)

    summary = svc.run_artifact_maintenance_once(trigger="manual")
    snapshots = svc.list_snapshots(session.id)

    assert len(snapshots) == 1
    assert summary.get("snapshot_evicted_sessions", 0) >= 1
    assert summary.get("snapshot_deleted_snapshots", 0) >= 2


def test_global_quota_enforcement_handles_hashed_session_directories(tmp_path: Path) -> None:
    """Maintenance must not hash an already-hashed session directory name."""
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = workspace / "state.txt"
    session_id = "session-raw-id"

    for value in ("v1", "v2", "v3"):
        state.write_text(value, encoding="utf-8")
        manager.create_snapshot(session_id, str(workspace))

    summary = manager.enforce_quota_all_sessions(max_snapshots=1, max_size_mb=256)

    assert len(manager.list_snapshots(session_id)) == 1
    assert summary == {
        "scanned_sessions": 1,
        "evicted_sessions": 1,
        "deleted_snapshots": 2,
    }


def test_create_snapshot_is_serialized_per_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    svc = SandboxService()
    session = svc.create_session(
        user_id="user-snap-lock",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc.get_session_workspace_path(session.id)
    assert ws is not None
    (Path(str(ws)) / "state.txt").write_text("v1", encoding="utf-8")

    gate = threading.Lock()
    active = 0
    peak_active = 0
    base_create = svc._snapshots.create_snapshot  # type: ignore[attr-defined]

    def _wrapped_create(session_id: str, workspace_path: str):
        nonlocal active, peak_active
        with gate:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.05)
            return base_create(session_id, workspace_path)
        finally:
            with gate:
                active -= 1

    monkeypatch.setattr(svc._snapshots, "create_snapshot", _wrapped_create)  # type: ignore[attr-defined]

    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            svc.create_snapshot(session.id)
        except BaseException as e:  # pragma: no cover - asserted via errors list
            errors.append(e)

    t1 = threading.Thread(target=_worker, daemon=True)
    t2 = threading.Thread(target=_worker, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert not t1.is_alive()
    assert not t2.is_alive()
    assert not errors
    assert peak_active == 1
    assert len(svc.list_snapshots(session.id)) == 2
