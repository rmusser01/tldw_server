from __future__ import annotations

import asyncio
import builtins
import contextlib
import io
import os
import shutil
import tarfile
import threading
import zipfile
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
import pytest

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


def _user(uid: int) -> User:
    return User(id=uid, username=f"user-{uid}", is_active=True, is_admin=False)


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


@pytest.fixture(autouse=True)
def _sandbox_upload_docker_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_docker_preflight_available(monkeypatch)


def _create_session(client: TestClient) -> str:
    """Helper to create a sandbox session and return its ID."""
    body = {
        "spec_version": "1.0",
        "runtime": "docker",
        "base_image": "python:3.11-slim",
        "timeout_sec": 60,
    }
    r = client.post("/api/v1/sandbox/sessions", json=body)
    assert r.status_code == 200
    return r.json()["id"]


def test_session_upload_creates_workspace(monkeypatch) -> None:
    os.environ.setdefault("TEST_MODE", "1")
    os.environ.setdefault("SANDBOX_ENABLE_EXECUTION", "false")
    os.environ.setdefault("SANDBOX_BACKGROUND_EXECUTION", "true")

    client = TestClient(app)
    app.dependency_overrides[get_request_user] = lambda: _user(1)
    try:
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "timeout_sec": 60,
        }
        r = client.post("/api/v1/sandbox/sessions", json=body)
        assert r.status_code == 200
        session_id = r.json()["id"]

        files = [("files", ("hello.txt", b"hello"))]
        r2 = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
        assert r2.status_code == 200
        payload = r2.json()
        assert payload.get("bytes_received") == 5
        assert payload.get("file_count") == 1
    finally:
        app.dependency_overrides.pop(get_request_user, None)


@pytest.mark.asyncio
async def test_upload_files_streams_plain_upload_without_unbounded_read(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))

    upload = UploadFile(filename="hello.txt", file=io.BytesIO(b"hello world"))
    original_read = upload.file.read

    def _guarded_read(size: int = -1):
        if size in (-1, None):
            raise AssertionError("plain upload should be read in bounded chunks")
        return original_read(size)

    upload.file.read = _guarded_read  # type: ignore[assignment]

    response = await sb.upload_files(
        request=None,  # type: ignore[arg-type]
        session_id="sess-plain",
        files=[upload],
        current_user=_user(1),
        audit_service=None,
    )

    assert response.bytes_received == 11
    assert response.file_count == 1
    assert (tmp_path / "hello.txt").read_bytes() == b"hello world"


@pytest.mark.asyncio
async def test_upload_files_offloads_filesystem_work_to_threads(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))

    upload = UploadFile(filename="threaded.txt", file=io.BytesIO(b"threaded upload"))
    original_to_thread = sb.asyncio.to_thread
    calls: list[str] = []

    async def _tracked_to_thread(func, /, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(sb.asyncio, "to_thread", _tracked_to_thread)

    response = await sb.upload_files(
        request=None,  # type: ignore[arg-type]
        session_id="sess-offload",
        files=[upload],
        current_user=_user(1),
        audit_service=None,
    )

    assert response.file_count == 1
    assert "_workspace_usage_bytes_sync" in calls
    assert "_prepare_temp_target_sync" in calls
    assert "_write_fd_chunk_sync" in calls
    assert "_finalize_temp_target_sync" in calls


@pytest.mark.asyncio
async def test_upload_files_streams_tar_members_without_full_member_read(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="safe.txt")
        data = b"streamed tar content"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    tar_buffer.seek(0)

    original_extractfile = tarfile.TarFile.extractfile

    class _GuardedReader:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def read(self, size: int = -1):
            if size in (-1, None):
                raise AssertionError("tar members should be copied in bounded chunks")
            return self._wrapped.read(size)

        def close(self):
            return self._wrapped.close()

    def _guarded_extractfile(self, member, *args, **kwargs):
        reader = original_extractfile(self, member, *args, **kwargs)
        if reader is None:
            return None
        return _GuardedReader(reader)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", _guarded_extractfile)

    upload = UploadFile(filename="archive.tar.gz", file=tar_buffer)
    response = await sb.upload_files(
        request=None,  # type: ignore[arg-type]
        session_id="sess-tar",
        files=[upload],
        current_user=_user(1),
        audit_service=None,
    )

    assert response.file_count == 1
    assert (tmp_path / "safe.txt").read_bytes() == b"streamed tar content"


@pytest.mark.asyncio
async def test_upload_files_streams_zip_members_without_zipfile_read(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("safe.txt", "streamed zip content")
    zip_buffer.seek(0)

    def _forbidden_read(self, name, pwd=None):
        raise AssertionError("zip uploads should stream via ZipFile.open, not ZipFile.read")

    monkeypatch.setattr(zipfile.ZipFile, "read", _forbidden_read)

    upload = UploadFile(filename="archive.zip", file=zip_buffer)
    response = await sb.upload_files(
        request=None,  # type: ignore[arg-type]
        session_id="sess-zip",
        files=[upload],
        current_user=_user(1),
        audit_service=None,
    )

    assert response.file_count == 1
    assert (tmp_path / "safe.txt").read_text(encoding="utf-8") == "streamed zip content"


@pytest.mark.asyncio
async def test_upload_files_enforces_cap_against_existing_workspace_bytes(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))
    monkeypatch.setenv("SANDBOX_WORKSPACE_CAP_MB", "1")

    first = UploadFile(filename="first.bin", file=io.BytesIO(b"a" * (700 * 1024)))
    second = UploadFile(filename="second.bin", file=io.BytesIO(b"b" * (700 * 1024)))

    first_response = await sb.upload_files(
        request=None,  # type: ignore[arg-type]
        session_id="sess-cap",
        files=[first],
        current_user=_user(1),
        audit_service=None,
    )
    assert first_response.file_count == 1

    with pytest.raises(HTTPException) as exc_info:
        await sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id="sess-cap",
            files=[second],
            current_user=_user(1),
            audit_service=None,
        )

    assert exc_info.value.status_code == 413
    assert (tmp_path / "first.bin").exists()
    assert not (tmp_path / "second.bin").exists()


@pytest.mark.asyncio
async def test_upload_files_offloads_blocking_workspace_filesystem_calls(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))

    workspace_root = os.path.abspath(str(tmp_path))
    main_thread_id = threading.get_ident()

    original_walk = sb.os.walk
    original_makedirs = sb.os.makedirs
    original_mkstemp = sb.tempfile.mkstemp
    original_replace = sb.os.replace
    original_unlink = sb.os.unlink
    original_open = builtins.open

    def _is_workspace_path(path: object) -> bool:
        try:
            candidate = os.path.abspath(os.fspath(path))
        except TypeError:
            return False
        return candidate.startswith(workspace_root)

    def _assert_worker_thread(operation: str, path: object) -> None:
        if _is_workspace_path(path) and threading.get_ident() == main_thread_id:
            raise AssertionError(f"{operation} should be offloaded from the event loop thread")

    def _guarded_walk(path: str, *args, **kwargs):
        _assert_worker_thread("os.walk", path)
        return original_walk(path, *args, **kwargs)

    def _guarded_makedirs(path: str, *args, **kwargs):
        _assert_worker_thread("os.makedirs", path)
        return original_makedirs(path, *args, **kwargs)

    def _guarded_mkstemp(*args, **kwargs):
        directory = kwargs.get("dir")
        if directory is not None:
            _assert_worker_thread("tempfile.mkstemp", directory)
        return original_mkstemp(*args, **kwargs)

    def _guarded_replace(src: str, dst: str, *args, **kwargs):
        _assert_worker_thread("os.replace", dst)
        return original_replace(src, dst, *args, **kwargs)

    def _guarded_unlink(path: str, *args, **kwargs):
        _assert_worker_thread("os.unlink", path)
        return original_unlink(path, *args, **kwargs)

    def _guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x")):
            _assert_worker_thread("open", file)
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(sb.os, "walk", _guarded_walk)
    monkeypatch.setattr(sb.os, "makedirs", _guarded_makedirs)
    monkeypatch.setattr(sb.tempfile, "mkstemp", _guarded_mkstemp)
    monkeypatch.setattr(sb.os, "replace", _guarded_replace)
    monkeypatch.setattr(sb.os, "unlink", _guarded_unlink)
    monkeypatch.setattr(builtins, "open", _guarded_open)

    upload = UploadFile(filename="threaded.txt", file=io.BytesIO(b"thread-safe upload"))

    response = await sb.upload_files(
        request=None,  # type: ignore[arg-type]
        session_id="sess-threaded",
        files=[upload],
        current_user=_user(1),
        audit_service=None,
    )

    assert response.bytes_received == len(b"thread-safe upload")
    assert (tmp_path / "threaded.txt").read_bytes() == b"thread-safe upload"


@pytest.mark.asyncio
async def test_concurrent_uploads_are_serialized_per_session(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))
    monkeypatch.setenv("SANDBOX_WORKSPACE_CAP_MB", "1")

    first = UploadFile(filename="first.bin", file=io.BytesIO(b"a" * (700 * 1024)))
    second = UploadFile(filename="second.bin", file=io.BytesIO(b"b" * (700 * 1024)))

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    original_first_read = first.file.read
    original_second_read = second.file.read

    def _first_read(size: int = -1):
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(timeout=1.0), "first upload was never released"
        return original_first_read(size)

    def _second_read(size: int = -1):
        second_entered.set()
        return original_second_read(size)

    first.file.read = _first_read  # type: ignore[assignment]
    second.file.read = _second_read  # type: ignore[assignment]

    first_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id="sess-concurrent",
            files=[first],
            current_user=_user(1),
            audit_service=None,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(first_entered.wait), timeout=1.0)

    second_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id="sess-concurrent",
            files=[second],
            current_user=_user(1),
            audit_service=None,
        )
    )

    await asyncio.sleep(0.05)
    assert not second_entered.is_set(), "second upload should wait for the session lock"

    release_first.set()
    first_response = await asyncio.wait_for(first_task, timeout=2.0)
    assert first_response.file_count == 1

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(second_task, timeout=2.0)

    assert exc_info.value.status_code == 413
    assert second_entered.is_set()
    assert (tmp_path / "first.bin").exists()
    assert not (tmp_path / "second.bin").exists()


@pytest.mark.asyncio
async def test_create_snapshot_waits_for_shared_workspace_lock_even_when_upload_targets_legacy_lock_name(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    svc = SandboxService()
    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb, "_service", svc)
    monkeypatch.setattr(svc, "get_session_workspace_path", lambda session_id: str(tmp_path))
    monkeypatch.setattr(svc._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))
    monkeypatch.setattr(svc._snapshots, "enforce_quota", lambda *args, **kwargs: [])

    snapshot_entered = threading.Event()
    snapshot_errors: list[BaseException] = []

    def _fake_create_snapshot(session_id: str, workspace_path: str) -> dict:
        snapshot_entered.set()
        return {
            "snapshot_id": "snap-test",
            "created_at": "2026-03-07T00:00:00+00:00",
            "size_bytes": 0,
        }

    monkeypatch.setattr(svc._snapshots, "create_snapshot", _fake_create_snapshot)

    upload = UploadFile(filename=".sandbox-upload.lock", file=io.BytesIO(b"legacy lock payload"))
    upload_entered = asyncio.Event()
    release_upload = asyncio.Event()
    original_read = upload.read

    async def _blocked_read(size: int = -1):
        if not upload_entered.is_set():
            upload_entered.set()
            await release_upload.wait()
        return await original_read(size)

    upload.read = _blocked_read  # type: ignore[assignment]

    upload_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id="sess-shared-lock",
            files=[upload],
            current_user=_user(1),
            audit_service=None,
        )
    )
    await asyncio.wait_for(upload_entered.wait(), timeout=1.0)

    def _run_snapshot() -> None:
        try:
            svc.create_snapshot("sess-shared-lock")
        except BaseException as exc:  # pragma: no cover - assertion path below
            snapshot_errors.append(exc)

    snapshot_thread = threading.Thread(target=_run_snapshot, daemon=True)
    snapshot_thread.start()

    await asyncio.sleep(0.05)
    assert not snapshot_entered.is_set(), "snapshot should wait until the upload releases the workspace lock"

    release_upload.set()
    upload_response = await asyncio.wait_for(upload_task, timeout=2.0)
    snapshot_thread.join(timeout=1.0)

    assert not snapshot_errors
    assert snapshot_entered.is_set()
    assert upload_response.file_count == 1
    assert (tmp_path / ".sandbox-upload.lock").read_bytes() == b"legacy lock payload"


@pytest.mark.asyncio
async def test_destroy_session_waits_for_shared_workspace_lock_during_upload(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    svc = SandboxService()
    workspace_root = tmp_path / "session-destroy" / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb, "_service", svc)
    monkeypatch.setattr(svc, "get_session_workspace_path", lambda session_id: str(workspace_root))
    monkeypatch.setattr(svc._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(workspace_root))
    monkeypatch.setattr(svc._snapshots, "cleanup_session_snapshots", lambda session_id: None)

    destroy_entered = threading.Event()
    destroy_errors: list[BaseException] = []

    def _fake_destroy_session(session_id: str, remove_workspace_tree: bool = True) -> bool:
        destroy_entered.set()
        return True

    monkeypatch.setattr(svc._orch, "destroy_session", _fake_destroy_session)

    upload = UploadFile(filename="payload.txt", file=io.BytesIO(b"delete waits"))
    upload_entered = asyncio.Event()
    release_upload = asyncio.Event()
    original_read = upload.read

    async def _blocked_read(size: int = -1):
        if not upload_entered.is_set():
            upload_entered.set()
            await release_upload.wait()
        return await original_read(size)

    upload.read = _blocked_read  # type: ignore[assignment]

    upload_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id="sess-destroy-lock",
            files=[upload],
            current_user=_user(1),
            audit_service=None,
        )
    )
    await asyncio.wait_for(upload_entered.wait(), timeout=1.0)

    def _run_destroy() -> None:
        try:
            svc.destroy_session("sess-destroy-lock")
        except BaseException as exc:  # pragma: no cover - asserted via errors list
            destroy_errors.append(exc)

    destroy_thread = threading.Thread(target=_run_destroy, daemon=True)
    destroy_thread.start()

    await asyncio.sleep(0.05)
    assert not destroy_entered.is_set(), "destroy should wait until the upload releases the workspace lock"

    release_upload.set()
    upload_response = await asyncio.wait_for(upload_task, timeout=2.0)
    destroy_thread.join(timeout=1.0)

    assert not destroy_errors
    assert destroy_entered.is_set()
    assert upload_response.file_count == 1


@pytest.mark.asyncio
async def test_upload_files_returns_404_if_session_is_deleted_while_waiting_for_lock(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    svc = SandboxService()
    session_root = tmp_path / "session-deleted"
    workspace_root = session_root / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(svc._orch, "_workspace_root", lambda: tmp_path.resolve())

    state = {"exists": True}

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb, "_service", svc)
    monkeypatch.setattr(
        svc._orch,
        "get_session_workspace_path",
        lambda session_id, **kwargs: str(workspace_root) if state["exists"] else None,
    )
    monkeypatch.setattr(svc._snapshots, "cleanup_session_snapshots", lambda session_id: None)

    destroy_entered = threading.Event()
    release_destroy = threading.Event()
    destroy_errors: list[BaseException] = []

    def _fake_destroy_session(session_id: str, remove_workspace_tree: bool = True) -> bool:
        destroy_entered.set()
        release_destroy.wait(timeout=2.0)
        state["exists"] = False
        return True

    monkeypatch.setattr(svc._orch, "destroy_session", _fake_destroy_session)

    upload = UploadFile(filename="payload.txt", file=io.BytesIO(b"should not survive delete"))
    upload_started = asyncio.Event()
    original_read = upload.read

    async def _tracked_read(size: int = -1):
        upload_started.set()
        return await original_read(size)

    upload.read = _tracked_read  # type: ignore[assignment]

    def _run_destroy() -> None:
        try:
            svc.destroy_session("sess-deleted")
        except BaseException as exc:  # pragma: no cover - asserted below
            destroy_errors.append(exc)

    destroy_thread = threading.Thread(target=_run_destroy, daemon=True)
    destroy_thread.start()

    assert destroy_entered.wait(timeout=1.0)

    upload_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id="sess-deleted",
            files=[upload],
            current_user=_user(1),
            audit_service=None,
        )
    )

    await asyncio.sleep(0.05)
    assert not upload_started.is_set(), "upload should wait behind destroy before reading the body"

    release_destroy.set()
    destroy_thread.join(timeout=1.0)

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(upload_task, timeout=2.0)

    assert not destroy_errors
    assert exc_info.value.status_code == 404
    assert not upload_started.is_set()
    assert not session_root.exists()


@pytest.mark.asyncio
async def test_upload_files_does_not_recreate_deleted_session_root_across_services(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    from tldw_Server_API.app.core.Sandbox.service import SandboxService
    from tldw_Server_API.app.core.Sandbox import service as svc_mod

    svc_delete = SandboxService()
    svc_upload = SandboxService()
    session_id = "sess-cross-worker-delete"
    session_root = tmp_path / session_id
    workspace_root = session_root / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    state = {"exists": True}

    def _workspace_lookup(_session_id: str, **kwargs) -> str | None:
        return str(workspace_root) if state["exists"] else None

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb, "_service", svc_upload)
    monkeypatch.setattr(svc_delete._orch, "get_session_workspace_path", _workspace_lookup)
    monkeypatch.setattr(svc_upload._orch, "get_session_workspace_path", _workspace_lookup)

    original_acquire = svc_mod._acquire_workspace_file_lock
    acquire_calls = {"count": 0}
    waiter_opened = threading.Event()
    delete_holding_lock = threading.Event()
    delete_errors: list[BaseException] = []

    def _instrumented_acquire(lock_path: str):
        acquire_calls["count"] += 1
        if acquire_calls["count"] != 2:
            return original_acquire(lock_path)

        if svc_mod._SANDBOX_SERVICE_HAS_FCNTL:
            handle = open(lock_path, "a", encoding="utf-8")
            waiter_opened.set()
            svc_mod.fcntl.flock(handle.fileno(), svc_mod.fcntl.LOCK_EX)
            return ("fcntl", handle)

        if svc_mod._SANDBOX_SERVICE_HAS_MSVCRT:
            handle = open(lock_path, "a+b")
            waiter_opened.set()
            with contextlib.suppress(Exception):
                handle.seek(0)
            svc_mod.msvcrt.locking(handle.fileno(), svc_mod.msvcrt.LK_LOCK, 1)
            return ("msvcrt", handle)

        waiter_opened.set()
        return original_acquire(lock_path)

    monkeypatch.setattr(svc_mod, "_acquire_workspace_file_lock", _instrumented_acquire)

    def _run_delete() -> None:
        try:
            with svc_delete._workspace_operation_lock(session_id, str(workspace_root)) as ws:
                delete_holding_lock.set()
                assert waiter_opened.wait(timeout=1.0)
                state["exists"] = False
            shutil.rmtree(Path(ws).parent, ignore_errors=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            delete_errors.append(exc)

    delete_thread = threading.Thread(target=_run_delete, daemon=True)
    delete_thread.start()
    assert delete_holding_lock.wait(timeout=1.0)

    upload = UploadFile(filename="payload.txt", file=io.BytesIO(b"cross worker upload"))
    upload_started = asyncio.Event()
    original_read = upload.read

    async def _tracked_read(size: int = -1):
        upload_started.set()
        return await original_read(size)

    upload.read = _tracked_read  # type: ignore[assignment]

    upload_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id=session_id,
            files=[upload],
            current_user=_user(1),
            audit_service=None,
        )
    )

    await asyncio.to_thread(waiter_opened.wait, 1.0)
    delete_thread.join(timeout=1.0)

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(upload_task, timeout=2.0)

    assert not delete_errors
    assert exc_info.value.status_code == 404
    assert not upload_started.is_set()
    assert not session_root.exists()


@pytest.mark.asyncio
async def test_upload_files_rejects_stale_cached_workspace_when_store_lookup_fails(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    from tldw_Server_API.app.core.Sandbox.service import SandboxService
    from tldw_Server_API.app.core.Sandbox import service as svc_mod

    svc_delete = SandboxService()
    svc_upload = SandboxService()
    session_id = "sess-store-failure-delete"
    session_root = tmp_path / session_id
    workspace_root = session_root / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb, "_service", svc_upload)
    monkeypatch.setattr(svc_delete._orch, "get_session_workspace_path", lambda _sid, **kwargs: str(workspace_root))
    monkeypatch.setattr(svc_upload._orch, "_prune_expired_sessions", lambda: None)

    with svc_upload._orch._lock:
        svc_upload._orch._session_roots[session_id] = str(workspace_root)

    def _raise_store_lookup(_sid: str):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(svc_upload._orch._store, "get_session", _raise_store_lookup)

    original_acquire = svc_mod._acquire_workspace_file_lock
    acquire_calls = {"count": 0}
    waiter_opened = threading.Event()
    delete_holding_lock = threading.Event()
    delete_errors: list[BaseException] = []

    def _instrumented_acquire(lock_path: str):
        acquire_calls["count"] += 1
        if acquire_calls["count"] != 2:
            return original_acquire(lock_path)

        if svc_mod._SANDBOX_SERVICE_HAS_FCNTL:
            handle = open(lock_path, "a", encoding="utf-8")
            waiter_opened.set()
            svc_mod.fcntl.flock(handle.fileno(), svc_mod.fcntl.LOCK_EX)
            return ("fcntl", handle)

        if svc_mod._SANDBOX_SERVICE_HAS_MSVCRT:
            handle = open(lock_path, "a+b")
            waiter_opened.set()
            with contextlib.suppress(Exception):
                handle.seek(0)
            svc_mod.msvcrt.locking(handle.fileno(), svc_mod.msvcrt.LK_LOCK, 1)
            return ("msvcrt", handle)

        waiter_opened.set()
        return original_acquire(lock_path)

    monkeypatch.setattr(svc_mod, "_acquire_workspace_file_lock", _instrumented_acquire)

    def _run_delete() -> None:
        try:
            with svc_delete._workspace_operation_lock(session_id, str(workspace_root)) as ws:
                delete_holding_lock.set()
                assert waiter_opened.wait(timeout=1.0)
            shutil.rmtree(Path(ws).parent, ignore_errors=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            delete_errors.append(exc)

    delete_thread = threading.Thread(target=_run_delete, daemon=True)
    delete_thread.start()
    assert delete_holding_lock.wait(timeout=1.0)

    upload = UploadFile(filename="payload.txt", file=io.BytesIO(b"stale cache upload"))
    upload_started = asyncio.Event()
    original_read = upload.read

    async def _tracked_read(size: int = -1):
        upload_started.set()
        return await original_read(size)

    upload.read = _tracked_read  # type: ignore[assignment]

    upload_task = asyncio.create_task(
        sb.upload_files(
            request=None,  # type: ignore[arg-type]
            session_id=session_id,
            files=[upload],
            current_user=_user(1),
            audit_service=None,
        )
    )

    await asyncio.to_thread(waiter_opened.wait, 1.0)
    delete_thread.join(timeout=1.0)

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(upload_task, timeout=2.0)

    assert not delete_errors
    assert exc_info.value.status_code == 404
    assert not upload_started.is_set()
    assert not session_root.exists()


# =============================================================================
# Security Tests: Path Traversal Prevention
# =============================================================================


class TestPathTraversalPrevention:
    """Test cases verifying path traversal attacks are blocked in file uploads."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up test environment."""
        os.environ.setdefault("TEST_MODE", "1")
        os.environ.setdefault("SANDBOX_ENABLE_EXECUTION", "false")
        os.environ.setdefault("SANDBOX_BACKGROUND_EXECUTION", "true")

    def test_traversal_filename_dotdot_blocked(self) -> None:
        """Files with ../ in name should be skipped (not cause traversal)."""
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Attempt upload with path traversal in filename
            files = [("files", ("../../../etc/passwd", b"malicious content"))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            # Should succeed but skip the malicious file (file_count = 0)
            assert r.status_code == 200
            payload = r.json()
            assert payload.get("file_count") == 0
            assert payload.get("bytes_received") == 0
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_traversal_filename_absolute_path_blocked(self) -> None:
        """Files with absolute paths should be skipped."""
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Attempt upload with absolute path in filename
            files = [("files", ("/etc/passwd", b"malicious content"))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            # Should succeed but skip the malicious file
            assert r.status_code == 200
            payload = r.json()
            assert payload.get("file_count") == 0
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_traversal_tar_dotdot_blocked(self) -> None:
        """Tar files containing ../ paths should have those entries skipped."""
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Create a tar file with a path traversal attempt
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tf:
                # Add a safe file
                safe_info = tarfile.TarInfo(name="safe.txt")
                safe_content = b"safe content"
                safe_info.size = len(safe_content)
                tf.addfile(safe_info, io.BytesIO(safe_content))

                # Add a malicious file with path traversal
                evil_info = tarfile.TarInfo(name="../../../tmp/evil.txt")
                evil_content = b"evil content"
                evil_info.size = len(evil_content)
                tf.addfile(evil_info, io.BytesIO(evil_content))

            tar_buffer.seek(0)
            files = [("files", ("archive.tar.gz", tar_buffer.read()))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            assert r.status_code == 200
            payload = r.json()
            # Only the safe file should be extracted
            assert payload.get("file_count") == 1
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_traversal_tar_symlink_blocked(self) -> None:
        """Symlinks in tar files should be skipped."""
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Create a tar file with a symlink
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tf:
                # Add a safe file
                safe_info = tarfile.TarInfo(name="safe.txt")
                safe_content = b"safe content"
                safe_info.size = len(safe_content)
                tf.addfile(safe_info, io.BytesIO(safe_content))

                # Add a symlink to /etc/passwd
                link_info = tarfile.TarInfo(name="evil_link")
                link_info.type = tarfile.SYMTYPE
                link_info.linkname = "/etc/passwd"
                tf.addfile(link_info)

            tar_buffer.seek(0)
            files = [("files", ("archive.tar.gz", tar_buffer.read()))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            assert r.status_code == 200
            payload = r.json()
            # Only the safe file should be extracted, symlink should be skipped
            assert payload.get("file_count") == 1
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_traversal_zip_dotdot_blocked(self) -> None:
        """Zip files containing ../ paths should have those entries skipped."""
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Create a zip file with a path traversal attempt
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                # Add a safe file
                zf.writestr("safe.txt", "safe content")
                # Add a malicious file with path traversal
                zf.writestr("../../../tmp/evil.txt", "evil content")

            zip_buffer.seek(0)
            files = [("files", ("archive.zip", zip_buffer.read()))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            assert r.status_code == 200
            payload = r.json()
            # Only the safe file should be extracted
            assert payload.get("file_count") == 1
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_traversal_zip_symlink_blocked(self) -> None:
        """Symlinks in zip files should be skipped.

        Zip files can represent symlinks via the external_attr field.
        The Unix symlink mode 0o120000 is stored in the high 16 bits.
        """
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Create a zip file with a symlink
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                # Add a safe file
                zf.writestr("safe.txt", "safe content")

                # Add a symlink entry
                # Unix symlink mode is 0o120000, stored in high 16 bits of external_attr
                symlink_info = zipfile.ZipInfo("evil_symlink")
                # Set symlink mode: 0o120777 (symlink with all perms) in high 16 bits
                symlink_info.external_attr = (0o120777 << 16)
                zf.writestr(symlink_info, "/etc/passwd")

            zip_buffer.seek(0)
            files = [("files", ("archive.zip", zip_buffer.read()))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            assert r.status_code == 200
            payload = r.json()
            # Only the safe file should be extracted, symlink should be skipped
            assert payload.get("file_count") == 1
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_safe_join_rejects_prefix_confusion(self) -> None:
        """Test that safe_join properly rejects prefix confusion attacks.

        A prefix confusion attack happens when:
        - base_dir = '/sandbox/user1'
        - name results in '/sandbox/user1_evil/file.txt'
        - A naive startswith() check would pass this
        """
        from tldw_Server_API.app.core.Utils.path_utils import safe_join

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "user1")
            os.makedirs(base)

            # Create a sibling directory that shares the prefix
            evil_sibling = os.path.join(tmpdir, "user1_evil")
            os.makedirs(evil_sibling)

            # Attempt to escape to sibling via traversal
            # This simulates: /sandbox/user1 + ../user1_evil/pwned.txt
            result = safe_join(base, "../user1_evil/pwned.txt")
            assert result is None, "safe_join should reject prefix confusion attacks"

    def test_safe_join_rejects_symlink_escape(self) -> None:
        """Test that safe_join rejects paths that resolve through symlinks."""
        from tldw_Server_API.app.core.Utils.path_utils import safe_join

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "workspace")
            os.makedirs(base)

            # Create target directory outside workspace
            target = os.path.join(tmpdir, "outside")
            os.makedirs(target)

            # Create a symlink inside workspace pointing outside
            symlink = os.path.join(base, "escape")
            os.symlink(target, symlink)

            # Attempt to access file through symlink
            result = safe_join(base, "escape/secret.txt")
            assert result is None, "safe_join should reject symlink escapes"

    def test_multiple_traversal_patterns(self) -> None:
        """Test various path traversal patterns are all blocked."""
        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # These traversal patterns use forward slashes which work on all platforms
            traversal_patterns = [
                "../secret.txt",
                "foo/../../../secret.txt",
                "foo/bar/../../../../../../etc/passwd",
                "./../secret.txt",
            ]

            for pattern in traversal_patterns:
                files = [("files", (pattern, b"test content"))]
                r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
                assert r.status_code == 200
                payload = r.json()
                # All traversal attempts should be blocked
                assert payload.get("file_count") == 0, f"Pattern {pattern!r} should be blocked"
        finally:
            app.dependency_overrides.pop(get_request_user, None)

    def test_windows_style_traversal_blocked_on_windows(self) -> None:
        """Test that Windows-style backslash traversal is blocked.

        Note: On Unix systems, backslashes are valid filename characters,
        so this test verifies proper handling on the current platform.
        """
        import sys

        client = TestClient(app)
        app.dependency_overrides[get_request_user] = lambda: _user(1)
        try:
            session_id = _create_session(client)

            # Windows-style traversal pattern
            pattern = "..\\secret.txt"
            files = [("files", (pattern, b"test content"))]
            r = client.post(f"/api/v1/sandbox/sessions/{session_id}/files", files=files)
            assert r.status_code == 200
            payload = r.json()

            if sys.platform == "win32":
                # On Windows, backslash is a path separator, so this should be blocked
                assert payload.get("file_count") == 0, "Windows traversal should be blocked on Windows"
            else:
                # On Unix, backslash is a valid filename character
                # The file is written with the literal name "..\\secret.txt"
                # which is safe (doesn't traverse) but allowed
                pass  # Either outcome is acceptable on Unix
        finally:
            app.dependency_overrides.pop(get_request_user, None)
