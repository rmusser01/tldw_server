from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType, RunSpec, Session, SessionSpec, TrustLevel
from tldw_Server_API.app.core.Sandbox.orchestrator import SandboxOrchestrator, SessionActiveRunsConflict
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.store import get_store


def _configure_sqlite_store(monkeypatch, tmp_path: Path) -> None:
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


def test_session_metadata_rehydrates_across_orchestrator_instances(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    spec = SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim")
    orch_a = SandboxOrchestrator()
    source = orch_a.create_session(
        user_id="user-42",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        body={"spec_version": "1.0", "runtime": "docker"},
    )

    owner_a = orch_a.get_session_owner(source.id)
    ws_a = orch_a.get_session_workspace_path(source.id)
    assert owner_a == "user-42"
    assert ws_a is not None

    marker = Path(str(ws_a)) / "marker.txt"
    marker.write_text("hello", encoding="utf-8")

    orch_b = SandboxOrchestrator()
    owner_b = orch_b.get_session_owner(source.id)
    restored = orch_b.get_session(source.id)
    ws_b = orch_b.get_session_workspace_path(source.id)

    assert owner_b == "user-42"
    assert restored is not None
    assert restored.id == source.id
    assert restored.runtime == RuntimeType.docker
    assert ws_b == ws_a
    assert (Path(str(ws_b)) / "marker.txt").read_text(encoding="utf-8") == "hello"


def test_clone_session_works_after_service_restart(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    source_service = SandboxService()
    spec = SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim")
    source = source_service.create_session(
        user_id="user-7",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    source_ws = source_service._orch.get_session_workspace_path(source.id)
    assert source_ws is not None
    source_file = Path(str(source_ws)) / "notes.txt"
    source_file.write_text("stateful data", encoding="utf-8")

    restarted_service = SandboxService()
    cloned = restarted_service.clone_session(source.id)
    cloned_ws = restarted_service._orch.get_session_workspace_path(cloned.id)

    assert cloned.id != source.id
    assert cloned_ws is not None
    assert restarted_service._orch.get_session_owner(cloned.id) == "user-7"
    assert (Path(str(cloned_ws)) / "notes.txt").read_text(encoding="utf-8") == "stateful data"


def test_session_execution_defaults_roundtrip_and_clone(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    source_service = SandboxService()
    spec = SessionSpec(
        runtime=RuntimeType.docker,
        base_image="python:3.12-slim",
        cpu_limit=1.5,
        memory_mb=768,
        timeout_sec=77,
        network_policy="deny_all",
        env={"SESSION_TOKEN": "present"},
        labels={"team": "sandbox"},
        trust_level=TrustLevel.trusted,
        persona_id="persona-77",
        workspace_id="workspace-77",
        workspace_group_id="wg-77",
        scope_snapshot_id="scope-77",
    )
    source = source_service.create_session(
        user_id="user-7",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )

    restarted_service = SandboxService()
    restored = restarted_service._orch.get_session(source.id)
    assert restored is not None
    assert restored.base_image == "python:3.12-slim"
    assert restored.cpu_limit == 1.5
    assert restored.memory_mb == 768
    assert restored.timeout_sec == 77
    assert restored.network_policy == "deny_all"
    assert restored.env == {"SESSION_TOKEN": "present"}
    assert restored.labels == {"team": "sandbox"}
    assert restored.trust_level == TrustLevel.trusted

    cloned = restarted_service.clone_session(source.id)
    assert cloned.base_image == "python:3.12-slim"
    assert cloned.cpu_limit == 1.5
    assert cloned.memory_mb == 768
    assert cloned.timeout_sec == 77
    assert cloned.network_policy == "deny_all"
    assert cloned.env == {"SESSION_TOKEN": "present"}
    assert cloned.labels == {"team": "sandbox"}
    assert cloned.trust_level == TrustLevel.trusted


def test_cross_node_delete_invalidates_cached_session_state(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    spec = SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim")
    orch_a = SandboxOrchestrator()
    source = orch_a.create_session(
        user_id="user-55",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        body={"spec_version": "1.0", "runtime": "docker"},
    )

    orch_b = SandboxOrchestrator()
    assert orch_b.get_session(source.id) is not None
    assert orch_b.get_session_workspace_path(source.id) is not None

    assert orch_a.destroy_session(source.id) is True

    assert orch_b.get_session(source.id) is None
    assert orch_b.get_session_workspace_path(source.id) is None
    with orch_b._lock:
        assert source.id not in orch_b._sessions
        assert source.id not in orch_b._session_roots


def test_cross_service_destroy_removes_store_backed_workspace_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    creator = SandboxService()
    session = creator.create_session(
        user_id="user-cross-service-cleanup",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    workspace_path = creator._orch.get_session_workspace_path(session.id)
    assert workspace_path is not None
    workspace = Path(str(workspace_path))
    session_root = workspace.parent
    (workspace / "state.txt").write_text("cleanup me", encoding="utf-8")

    destroyer = SandboxService()

    assert destroyer.destroy_session(session.id) is True
    assert destroyer._orch.get_session(session.id) is None
    assert destroyer._orch.get_session_workspace_path(session.id) is None
    assert not session_root.exists()


def test_destroy_session_cancels_and_drains_when_active_runs_exist(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    orch = SandboxOrchestrator()
    session = orch.create_session(
        user_id="user-66",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        body={"spec_version": "1.0", "runtime": "docker"},
    )
    run = orch.enqueue_run(
        user_id="user-66",
        spec=RunSpec(
            session_id=session.id,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["python", "-c", "print('queued')"],
        ),
        spec_version="1.0",
        idem_key=None,
        body={"command": ["python", "-c", "print('queued')"], "session_id": session.id},
    )

    svc = SandboxService()
    assert svc.destroy_session(session.id) is True
    killed = orch.get_run(run.id)
    assert killed is not None
    assert killed.phase == RunPhase.killed


def test_destroy_session_cleans_snapshots_artifacts_and_usage(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-77",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc._orch.get_session_workspace_path(session.id)
    assert ws is not None
    (Path(str(ws)) / "state.txt").write_text("snapshot me", encoding="utf-8")
    snap = svc.create_snapshot(session.id)
    assert snap.get("snapshot_id")

    run = svc._orch.enqueue_run(
        user_id="user-77",
        spec=RunSpec(
            session_id=session.id,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["python", "-c", "print('queued')"],
        ),
        spec_version="1.0",
        idem_key=None,
        body={"command": ["python", "-c", "print('queued')"], "session_id": session.id},
    )
    svc._orch.store_artifacts(run.id, {"out.txt": b"hello"})
    snapshot_dir = Path(tmp_path) / "snapshots" / session.id
    artifact_dir = svc._orch._artifact_dir("user-77", run.id)  # type: ignore[attr-defined]
    assert snapshot_dir.exists()
    assert artifact_dir.exists()
    assert svc._orch._store.get_user_artifact_bytes("user-77") == 5  # type: ignore[attr-defined]

    assert svc.destroy_session(session.id) is True

    killed = svc._orch.get_run(run.id)
    assert killed is not None
    assert killed.phase == RunPhase.killed
    assert not snapshot_dir.exists()
    assert not artifact_dir.exists()
    assert svc._orch._store.get_user_artifact_bytes("user-77") == 0  # type: ignore[attr-defined]


def test_destroy_session_removes_session_root_but_keeps_workspace_lock_file(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-lock-cleanup",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc._orch.get_session_workspace_path(session.id)
    assert ws is not None
    workspace_path = Path(str(ws))
    session_root = workspace_path.parent
    lock_path = Path(svc._workspace_operation_lock_path(session.id))

    (workspace_path / "state.txt").write_text("cleanup me", encoding="utf-8")
    svc.create_snapshot(session.id)

    assert session_root.exists()
    assert lock_path.exists()

    assert svc.destroy_session(session.id) is True

    assert not session_root.exists()
    assert lock_path.exists()


def test_create_snapshot_rejects_active_session_runs(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-snapshot-busy",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )

    svc._orch.enqueue_run(
        user_id="user-snapshot-busy",
        spec=RunSpec(
            session_id=session.id,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["python", "-c", "print('queued')"],
        ),
        spec_version="1.0",
        idem_key=None,
        body={"command": ["python", "-c", "print('queued')"], "session_id": session.id},
    )

    with pytest.raises(SessionActiveRunsConflict) as exc_info:
        svc.create_snapshot(session.id)

    assert exc_info.value.session_id == session.id
    assert exc_info.value.active_runs == 1


def test_restore_snapshot_rejects_active_session_runs(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-restore",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc._orch.get_session_workspace_path(session.id)
    assert ws is not None
    (Path(str(ws)) / "state.txt").write_text("before snapshot", encoding="utf-8")
    snap = svc.create_snapshot(session.id)

    svc._orch.enqueue_run(
        user_id="user-restore",
        spec=RunSpec(
            session_id=session.id,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["python", "-c", "print('queued')"],
        ),
        spec_version="1.0",
        idem_key=None,
        body={"command": ["python", "-c", "print('queued')"], "session_id": session.id},
    )

    with pytest.raises(SessionActiveRunsConflict) as exc_info:
        svc.restore_snapshot(session.id, str(snap["snapshot_id"]))

    assert exc_info.value.session_id == session.id
    assert exc_info.value.active_runs == 1


def test_clone_session_rejects_active_session_runs(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-clone-busy",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc._orch.get_session_workspace_path(session.id)
    assert ws is not None
    (Path(str(ws)) / "notes.txt").write_text("clone state", encoding="utf-8")

    svc._orch.enqueue_run(
        user_id="user-clone-busy",
        spec=RunSpec(
            session_id=session.id,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            command=["python", "-c", "print('queued')"],
        ),
        spec_version="1.0",
        idem_key=None,
        body={"command": ["python", "-c", "print('queued')"], "session_id": session.id},
    )

    with pytest.raises(SessionActiveRunsConflict) as exc_info:
        svc.clone_session(session.id)

    assert exc_info.value.session_id == session.id
    assert exc_info.value.active_runs == 1


def test_session_backed_start_run_waits_for_snapshot_workspace_operation(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-run-lock",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc._orch.get_session_workspace_path(session.id)
    assert ws is not None
    (Path(str(ws)) / "state.txt").write_text("snapshot me", encoding="utf-8")

    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    snapshot_errors: list[BaseException] = []
    run_errors: list[BaseException] = []
    enqueue_entered = threading.Event()
    run_result: dict[str, object] = {}

    def _blocked_snapshot(session_id: str, workspace_path: str) -> dict:
        snapshot_entered.set()
        release_snapshot.wait(timeout=2.0)
        return {
            "snapshot_id": "snap-lock",
            "created_at": "2026-03-08T00:00:00+00:00",
            "size_bytes": 0,
        }

    original_enqueue = svc._orch.enqueue_run

    def _tracked_enqueue(*args, **kwargs):
        enqueue_entered.set()
        return original_enqueue(*args, **kwargs)

    monkeypatch.setattr(svc._snapshots, "create_snapshot", _blocked_snapshot)
    monkeypatch.setattr(svc._snapshots, "enforce_quota", lambda *args, **kwargs: [])
    monkeypatch.setattr(svc._orch, "enqueue_run", _tracked_enqueue)

    def _run_snapshot() -> None:
        try:
            svc.create_snapshot(session.id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            snapshot_errors.append(exc)

    def _run_start() -> None:
        try:
            run_result["status"] = svc.start_run_scaffold(
                user_id="user-run-lock",
                spec=RunSpec(
                    session_id=session.id,
                    runtime=RuntimeType.docker,
                    base_image="python:3.11-slim",
                    command=["python", "-c", "print('queued')"],
                ),
                spec_version="1.0",
                idem_key=None,
                raw_body={"command": ["python", "-c", "print('queued')"], "session_id": session.id},
            )
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            run_errors.append(exc)

    snapshot_thread = threading.Thread(target=_run_snapshot, daemon=True)
    snapshot_thread.start()
    assert snapshot_entered.wait(timeout=1.0)

    run_attempted = threading.Event()

    def _run_start_with_signal() -> None:
        run_attempted.set()
        _run_start()

    run_thread = threading.Thread(target=_run_start_with_signal, daemon=True)
    run_thread.start()
    assert run_attempted.wait(timeout=1.0)
    assert not enqueue_entered.is_set(), "session-backed run enqueue should wait for workspace operations"

    release_snapshot.set()
    snapshot_thread.join(timeout=1.0)
    run_thread.join(timeout=1.0)

    assert not snapshot_errors
    assert not run_errors
    assert enqueue_entered.is_set()
    status = run_result.get("status")
    assert status is not None
    assert status.phase == RunPhase.queued  # type: ignore[union-attr]


def test_session_backed_start_run_waits_for_snapshot_workspace_operation_across_services(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")

    session_id = "sess-cross-worker-run"
    workspace_root = tmp_path / session_id / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "state.txt").write_text("snapshot me", encoding="utf-8")

    svc_snapshot = SandboxService()
    svc_run = SandboxService()

    monkeypatch.setattr(svc_snapshot._orch, "get_session_workspace_path", lambda _sid, **kwargs: str(workspace_root))
    monkeypatch.setattr(svc_run._orch, "get_session_workspace_path", lambda _sid, **kwargs: str(workspace_root))
    monkeypatch.setattr(svc_run._orch, "get_session", lambda _sid, **kwargs: SimpleNamespace(id=session_id))

    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    snapshot_errors: list[BaseException] = []
    run_errors: list[BaseException] = []
    enqueue_entered = threading.Event()
    run_result: dict[str, object] = {}

    def _blocked_snapshot(session_id: str, workspace_path: str) -> dict:
        snapshot_entered.set()
        release_snapshot.wait(timeout=2.0)
        return {
            "snapshot_id": "snap-cross-worker",
            "created_at": "2026-03-08T00:00:00+00:00",
            "size_bytes": 0,
        }

    original_enqueue = svc_run._orch.enqueue_run

    def _tracked_enqueue(*args, **kwargs):
        enqueue_entered.set()
        return original_enqueue(*args, **kwargs)

    monkeypatch.setattr(svc_snapshot._snapshots, "create_snapshot", _blocked_snapshot)
    monkeypatch.setattr(svc_snapshot._snapshots, "enforce_quota", lambda *args, **kwargs: [])
    monkeypatch.setattr(svc_run._orch, "enqueue_run", _tracked_enqueue)

    def _run_snapshot() -> None:
        try:
            svc_snapshot.create_snapshot(session_id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            snapshot_errors.append(exc)

    def _run_start() -> None:
        try:
            run_result["status"] = svc_run.start_run_scaffold(
                user_id="user-run-lock",
                spec=RunSpec(
                    session_id=session_id,
                    runtime=RuntimeType.docker,
                    base_image="python:3.11-slim",
                    command=["python", "-c", "print('queued')"],
                ),
                spec_version="1.0",
                idem_key=None,
                raw_body={"command": ["python", "-c", "print('queued')"], "session_id": session_id},
            )
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            run_errors.append(exc)

    snapshot_thread = threading.Thread(target=_run_snapshot, daemon=True)
    snapshot_thread.start()
    assert snapshot_entered.wait(timeout=1.0)

    run_attempted = threading.Event()

    def _run_start_with_signal() -> None:
        run_attempted.set()
        _run_start()

    run_thread = threading.Thread(target=_run_start_with_signal, daemon=True)
    run_thread.start()
    assert run_attempted.wait(timeout=1.0)
    assert not enqueue_entered.is_set(), "cross-service session-backed enqueue should wait for workspace operations"

    release_snapshot.set()
    snapshot_thread.join(timeout=1.0)
    run_thread.join(timeout=1.0)

    assert not snapshot_errors
    assert not run_errors
    assert enqueue_entered.is_set()
    status = run_result.get("status")
    assert status is not None
    assert status.phase == RunPhase.queued  # type: ignore[union-attr]


def test_list_snapshots_waits_for_cross_service_snapshot_create(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc_create = SandboxService()
    svc_list = SandboxService()
    session = svc_create.create_session(
        user_id="user-snapshot-list-lock",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc_create._orch.get_session_workspace_path(session.id)
    assert ws is not None
    workspace = Path(str(ws))
    (workspace / "state.txt").write_text("before", encoding="utf-8")
    first = svc_create.create_snapshot(session.id)
    assert first.get("snapshot_id")

    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    create_errors: list[BaseException] = []
    list_errors: list[BaseException] = []
    list_result: dict[str, object] = {}
    base_create = svc_create._snapshots.create_snapshot

    def _blocked_create(session_id: str, workspace_path: str) -> dict:
        snapshot_entered.set()
        release_snapshot.wait(timeout=2.0)
        return base_create(session_id, workspace_path)

    monkeypatch.setattr(svc_create._snapshots, "create_snapshot", _blocked_create)
    (workspace / "state.txt").write_text("after", encoding="utf-8")

    def _run_create() -> None:
        try:
            list_result["created"] = svc_create.create_snapshot(session.id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            create_errors.append(exc)

    list_called = threading.Event()
    original_list_snapshots = svc_list._snapshots.list_snapshots

    def _tracked_list_snapshots(session_id: str):
        list_called.set()
        return original_list_snapshots(session_id)

    monkeypatch.setattr(svc_list._snapshots, "list_snapshots", _tracked_list_snapshots)

    def _run_list() -> None:
        try:
            list_result["snapshots"] = svc_list.list_snapshots(session.id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            list_errors.append(exc)

    create_thread = threading.Thread(target=_run_create, daemon=True)
    create_thread.start()
    assert snapshot_entered.wait(timeout=1.0)

    list_attempted = threading.Event()

    def _run_list_with_signal() -> None:
        list_attempted.set()
        _run_list()

    list_thread = threading.Thread(target=_run_list_with_signal, daemon=True)
    list_thread.start()
    assert list_attempted.wait(timeout=1.0)
    assert not list_called.is_set(), "snapshot listing should not reach the underlying store while create holds the lock"
    assert "snapshots" not in list_result, "cross-service snapshot listing should wait for in-progress create"

    release_snapshot.set()
    create_thread.join(timeout=1.0)
    list_thread.join(timeout=1.0)

    assert not create_errors
    assert not list_errors
    snapshots = list_result.get("snapshots")
    assert isinstance(snapshots, list)
    assert len(snapshots) == 2


def test_get_snapshot_info_waits_for_cross_service_snapshot_create(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc_create = SandboxService()
    svc_info = SandboxService()
    session = svc_create.create_session(
        user_id="user-snapshot-info-lock",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc_create._orch.get_session_workspace_path(session.id)
    assert ws is not None
    workspace = Path(str(ws))
    (workspace / "state.txt").write_text("before", encoding="utf-8")
    first = svc_create.create_snapshot(session.id)
    snapshot_id = str(first["snapshot_id"])

    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    create_errors: list[BaseException] = []
    info_errors: list[BaseException] = []
    info_result: dict[str, object] = {}
    base_create = svc_create._snapshots.create_snapshot

    def _blocked_create(session_id: str, workspace_path: str) -> dict:
        snapshot_entered.set()
        release_snapshot.wait(timeout=2.0)
        return base_create(session_id, workspace_path)

    monkeypatch.setattr(svc_create._snapshots, "create_snapshot", _blocked_create)
    (workspace / "state.txt").write_text("after", encoding="utf-8")

    def _run_create() -> None:
        try:
            info_result["created"] = svc_create.create_snapshot(session.id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            create_errors.append(exc)

    info_called = threading.Event()
    original_get_snapshot_info = svc_info._snapshots.get_snapshot_info

    def _tracked_get_snapshot_info(session_id: str, snapshot_id_arg: str):
        info_called.set()
        return original_get_snapshot_info(session_id, snapshot_id_arg)

    monkeypatch.setattr(svc_info._snapshots, "get_snapshot_info", _tracked_get_snapshot_info)

    def _run_info() -> None:
        try:
            info_result["snapshot"] = svc_info.get_snapshot_info(session.id, snapshot_id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            info_errors.append(exc)

    create_thread = threading.Thread(target=_run_create, daemon=True)
    create_thread.start()
    assert snapshot_entered.wait(timeout=1.0)

    info_attempted = threading.Event()

    def _run_info_with_signal() -> None:
        info_attempted.set()
        _run_info()

    info_thread = threading.Thread(target=_run_info_with_signal, daemon=True)
    info_thread.start()
    assert info_attempted.wait(timeout=1.0)
    assert not info_called.is_set(), "snapshot info should not reach the underlying store while create holds the lock"
    assert "snapshot" not in info_result, "cross-service snapshot info should wait for in-progress create"

    release_snapshot.set()
    create_thread.join(timeout=1.0)
    info_thread.join(timeout=1.0)

    assert not create_errors
    assert not info_errors
    snapshot_info = info_result.get("snapshot")
    assert isinstance(snapshot_info, dict)
    assert snapshot_info.get("snapshot_id") == snapshot_id


def test_delete_snapshot_waits_for_cross_service_snapshot_create(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    svc_create = SandboxService()
    svc_delete = SandboxService()
    session = svc_create.create_session(
        user_id="user-snapshot-delete-lock",
        spec=SessionSpec(runtime=RuntimeType.docker, base_image="python:3.11-slim"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "docker"},
    )
    ws = svc_create._orch.get_session_workspace_path(session.id)
    assert ws is not None
    workspace = Path(str(ws))
    (workspace / "state.txt").write_text("before", encoding="utf-8")
    first = svc_create.create_snapshot(session.id)
    snapshot_id = str(first["snapshot_id"])

    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    create_errors: list[BaseException] = []
    delete_errors: list[BaseException] = []
    delete_result: dict[str, object] = {}
    base_create = svc_create._snapshots.create_snapshot

    def _blocked_create(session_id: str, workspace_path: str) -> dict:
        snapshot_entered.set()
        release_snapshot.wait(timeout=2.0)
        return base_create(session_id, workspace_path)

    monkeypatch.setattr(svc_create._snapshots, "create_snapshot", _blocked_create)
    (workspace / "state.txt").write_text("after", encoding="utf-8")

    def _run_create() -> None:
        try:
            delete_result["created"] = svc_create.create_snapshot(session.id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            create_errors.append(exc)

    delete_called = threading.Event()
    original_delete_snapshot = svc_delete._snapshots.delete_snapshot

    def _tracked_delete_snapshot(session_id: str, snapshot_id_arg: str):
        delete_called.set()
        return original_delete_snapshot(session_id, snapshot_id_arg)

    monkeypatch.setattr(svc_delete._snapshots, "delete_snapshot", _tracked_delete_snapshot)

    def _run_delete() -> None:
        try:
            delete_result["deleted"] = svc_delete.delete_snapshot(session.id, snapshot_id)
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            delete_errors.append(exc)

    create_thread = threading.Thread(target=_run_create, daemon=True)
    create_thread.start()
    assert snapshot_entered.wait(timeout=1.0)

    delete_attempted = threading.Event()

    def _run_delete_with_signal() -> None:
        delete_attempted.set()
        _run_delete()

    delete_thread = threading.Thread(target=_run_delete_with_signal, daemon=True)
    delete_thread.start()
    assert delete_attempted.wait(timeout=1.0)
    assert not delete_called.is_set(), "snapshot delete should not reach the underlying store while create holds the lock"
    assert "deleted" not in delete_result, "cross-service snapshot delete should wait for in-progress create"

    release_snapshot.set()
    create_thread.join(timeout=1.0)
    delete_thread.join(timeout=1.0)

    assert not create_errors
    assert not delete_errors
    assert delete_result.get("deleted") is True
    assert svc_create.get_snapshot_info(session.id, snapshot_id) is None


def test_session_backed_start_run_rejects_stale_cached_session_when_store_lookup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")

    svc = SandboxService()
    session_id = "sess-stale-run-cache"
    workspace_root = tmp_path / session_id / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    cached_session = Session(
        id=session_id,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        expires_at=None,
    )
    with svc._orch._lock:
        svc._orch._sessions[session_id] = cached_session
        svc._orch._session_roots[session_id] = str(workspace_root)

    monkeypatch.setattr(svc._orch, "_prune_expired_sessions", lambda: None)

    def _raise_store_lookup(_sid: str):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(svc._orch._store, "get_session", _raise_store_lookup)

    enqueue_called = {"value": False}

    def _tracked_enqueue(*args, **kwargs):
        enqueue_called["value"] = True
        return RunStatus(
            id="run-stale-cache",
            phase=RunPhase.queued,
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
        )

    monkeypatch.setattr(svc._orch, "enqueue_run", _tracked_enqueue)

    with pytest.raises(ValueError, match="session_not_found"):
        svc.start_run_scaffold(
            user_id="user-stale-run",
            spec=RunSpec(
                session_id=session_id,
                runtime=RuntimeType.docker,
                base_image="python:3.11-slim",
                command=["python", "-c", "print('queued')"],
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"command": ["python", "-c", "print('queued')"], "session_id": session_id},
        )

    assert enqueue_called["value"] is False


def test_session_tenancy_metadata_roundtrips_across_orchestrators(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    spec = SessionSpec(
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        persona_id="persona-123",
        workspace_id="workspace-abc",
        workspace_group_id="wg-team-1",
        scope_snapshot_id="scope-v1",
    )
    orch_a = SandboxOrchestrator()
    source = orch_a.create_session(
        user_id="user-88",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        body={"spec_version": "1.0", "runtime": "docker"},
    )

    assert source.persona_id == "persona-123"
    assert source.workspace_id == "workspace-abc"
    assert source.workspace_group_id == "wg-team-1"
    assert source.scope_snapshot_id == "scope-v1"

    orch_b = SandboxOrchestrator()
    restored = orch_b.get_session(source.id)
    assert restored is not None
    assert restored.persona_id == "persona-123"
    assert restored.workspace_id == "workspace-abc"
    assert restored.workspace_group_id == "wg-team-1"
    assert restored.scope_snapshot_id == "scope-v1"


def test_run_tenancy_metadata_persists_and_lists(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    orch = SandboxOrchestrator()
    spec = RunSpec(
        session_id="session-tenancy-1",
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print('ok')"],
        persona_id="persona-xyz",
        workspace_id="workspace-7",
        workspace_group_id="group-A",
        scope_snapshot_id="scope-2026-02-22",
    )
    status = orch.enqueue_run(
        user_id="user-901",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        body={"command": ["python", "-c", "print('ok')"]},
    )

    stored = orch.get_run(status.id)
    assert stored is not None
    assert stored.session_id == "session-tenancy-1"
    assert stored.persona_id == "persona-xyz"
    assert stored.workspace_id == "workspace-7"
    assert stored.workspace_group_id == "group-A"
    assert stored.scope_snapshot_id == "scope-2026-02-22"

    listed = orch.list_runs(user_id="user-901", limit=500, offset=0)
    row = next((item for item in listed if item.get("id") == status.id), None)
    if row is None:
        raise AssertionError(f"Run {status.id} missing from list_runs(user_id='user-901'): {listed!r}")
    assert row.get("session_id") == "session-tenancy-1"
    assert row.get("persona_id") == "persona-xyz"
    assert row.get("workspace_id") == "workspace-7"
    assert row.get("workspace_group_id") == "group-A"
    assert row.get("scope_snapshot_id") == "scope-2026-02-22"


def test_acp_control_metadata_roundtrips_across_store_instances(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    store_a = get_store()
    store_a.put_acp_session_control(
        session_id="acp-session-1",
        user_id="user-500",
        sandbox_session_id="sandbox-session-1",
        run_id="run-1",
        ssh_host="127.0.0.1",
        ssh_port=4122,
        ssh_user="acp",
        ssh_private_key="PRIVATE-KEY",
        persona_id="persona-1",
        workspace_id="workspace-1",
        workspace_group_id="wg-1",
        scope_snapshot_id="scope-1",
    )

    store_b = get_store()
    row = store_b.get_acp_session_control("acp-session-1")
    assert row is not None
    assert row.get("user_id") == "user-500"
    assert row.get("sandbox_session_id") == "sandbox-session-1"
    assert row.get("run_id") == "run-1"
    assert row.get("ssh_user") == "acp"
    assert row.get("persona_id") == "persona-1"
    assert row.get("workspace_id") == "workspace-1"
    assert row.get("workspace_group_id") == "wg-1"
    assert row.get("scope_snapshot_id") == "scope-1"

    assert store_b.delete_acp_session_control("acp-session-1") is True
    assert store_b.get_acp_session_control("acp-session-1") is None
