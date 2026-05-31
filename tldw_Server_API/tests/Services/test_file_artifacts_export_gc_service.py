from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import StoragePathValidationError
from tldw_Server_API.app.services import file_artifacts_export_gc_service as service


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


def test_enumerate_user_ids_base_dir_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_user_db_base_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/files-gc-base sk-live-base")),
    )

    assert service._enumerate_user_ids() == []
    assert logger.debugs == ["files_export_gc: failed to resolve user db base dir"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-base" not in rendered
    assert "sk-live-base" not in rendered


def test_enumerate_user_ids_single_user_fallback_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.DatabasePaths, "get_user_db_base_dir", lambda: tmp_path)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_single_user_id",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/files-gc-single sk-live-single")),
    )

    assert service._enumerate_user_ids() == []
    assert logger.debugs == ["files_export_gc: failed to derive single user id"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-single" not in rendered
    assert "sk-live-single" not in rendered


@pytest.mark.asyncio
async def test_purge_expired_exports_invalid_export_path_warning_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    row = {
        "id": 12,
        "export_storage_path": "exports/file.csv",
        "export_format": "csv",
        "export_bytes": 4,
        "export_content_type": "text/csv",
        "export_job_id": "job-1",
    }
    cdb = SimpleNamespace(
        list_file_artifacts_expired_exports=lambda now_iso: [row],
        resolve_temp_output_storage_path=lambda storage_path: (_ for _ in ()).throw(
            StoragePathValidationError("bad /tmp/files-gc-path sk-live-path")
        ),
        update_file_artifact_export=lambda *args, **kwargs: None,
    )

    class _CollectionsContext:
        def __enter__(self):
            return cdb

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.CollectionsDatabase, "for_user", lambda user_id: _CollectionsContext())

    cleared, files_deleted = await service._purge_expired_exports_for_user(user_id=7, now_iso="2026-01-01T00:00:00+00:00")

    assert (cleared, files_deleted) == (1, 0)
    assert logger.warnings == ["files_export_gc: invalid export path for file 12"]
    assert logger.binds == [{"error_type": "StoragePathValidationError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-path" not in rendered
    assert "sk-live-path" not in rendered


@pytest.mark.asyncio
async def test_purge_expired_exports_delete_file_warning_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger = _LoggerStub()
    row = {
        "id": 12,
        "export_storage_path": "exports/file.csv",
        "export_format": "csv",
        "export_bytes": 4,
        "export_content_type": "text/csv",
        "export_job_id": "job-1",
    }
    outputs_dir = tmp_path / "outputs"
    target = outputs_dir / "exports" / "file.csv"
    target.parent.mkdir(parents=True)
    target.write_text("payload")
    original_unlink = Path.unlink

    def _patched_unlink(self, *args, **kwargs):
        if self == target:
            raise PermissionError("cannot unlink /tmp/files-gc-delete sk-live-delete")
        return original_unlink(self, *args, **kwargs)

    cdb = SimpleNamespace(
        list_file_artifacts_expired_exports=lambda now_iso: [row],
        resolve_temp_output_storage_path=lambda storage_path: "exports/file.csv",
        update_file_artifact_export=lambda *args, **kwargs: None,
    )

    class _CollectionsContext:
        def __enter__(self):
            return cdb

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.CollectionsDatabase, "for_user", lambda user_id: _CollectionsContext())
    monkeypatch.setattr(service.DatabasePaths, "get_user_temp_outputs_dir", lambda user_id: outputs_dir)
    monkeypatch.setattr(Path, "unlink", _patched_unlink)

    cleared, files_deleted = await service._purge_expired_exports_for_user(user_id=7, now_iso="2026-01-01T00:00:00+00:00")

    assert (cleared, files_deleted) == (1, 0)
    assert logger.warnings == ["files_export_gc: failed to delete export file for 12"]
    assert logger.binds == [{"error_type": "PermissionError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-delete" not in rendered
    assert "sk-live-delete" not in rendered


@pytest.mark.asyncio
async def test_purge_expired_exports_clear_state_warning_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    row = {
        "id": 12,
        "export_storage_path": None,
        "export_format": "csv",
        "export_bytes": 4,
        "export_content_type": "text/csv",
        "export_job_id": "job-1",
    }
    cdb = SimpleNamespace(
        list_file_artifacts_expired_exports=lambda now_iso: [row],
        update_file_artifact_export=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("cannot clear /tmp/files-gc-clear sk-live-clear")
        ),
    )

    class _CollectionsContext:
        def __enter__(self):
            return cdb

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.CollectionsDatabase, "for_user", lambda user_id: _CollectionsContext())

    cleared, files_deleted = await service._purge_expired_exports_for_user(user_id=7, now_iso="2026-01-01T00:00:00+00:00")

    assert (cleared, files_deleted) == (0, 0)
    assert logger.warnings == ["files_export_gc: failed to clear export state for 12"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-clear" not in rendered
    assert "sk-live-clear" not in rendered


@pytest.mark.asyncio
async def test_start_scheduler_invalid_interval_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _LoggerStub()
    created = {}
    monkeypatch.setenv("FILES_EXPORT_GC_ENABLED", "true")
    monkeypatch.setenv("FILES_EXPORT_GC_INTERVAL_SEC", "bad /tmp/files-gc-interval sk-live-interval")
    monkeypatch.setattr(service, "logger", logger)

    def _fake_create_task(coro, *, name=None):
        created["name"] = name
        coro.close()
        return SimpleNamespace(name=name)

    monkeypatch.setattr(service.asyncio, "create_task", _fake_create_task)

    task = await service.start_file_artifacts_export_gc_scheduler()

    assert task is not None
    assert created == {"name": "file_artifacts_export_gc"}
    assert logger.debugs == ["files_export_gc: invalid FILES_EXPORT_GC_INTERVAL_SEC; using default"]
    assert logger.binds == [{"error_type": "ValueError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-interval" not in rendered
    assert "sk-live-interval" not in rendered


@pytest.mark.asyncio
async def test_start_scheduler_run_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _LoggerStub()
    monkeypatch.setenv("FILES_EXPORT_GC_ENABLED", "true")
    monkeypatch.setenv("FILES_EXPORT_GC_INTERVAL_SEC", "60")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service,
        "_enumerate_user_ids",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/files-gc-run sk-live-run")),
    )

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)

    task = await service.start_file_artifacts_export_gc_scheduler()
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert "files_export_gc: run failed" in logger.debugs
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/files-gc-run" not in rendered
    assert "sk-live-run" not in rendered
